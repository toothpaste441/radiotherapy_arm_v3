"""Fail-closed educational multi-node robot motion around a synthetic brain target.

This model is NOT a treatment simulator, and an incomplete/failed run is reported
as failed. No radiation is modelled or delivered.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
from src.brain_scene import create_brain_scene, set_beam_markers
from src.beam_ik import candidate_beam_directions, solve_beam_ik
from src.kinematics import RobotKinematics
from src.path_planning import rrt_connect
from src.robust_collision import DetailedCollisionChecker
from src.stable_joint_control import ComputedTorquePD, ArrivalGate
from src.mujoco_compat import fill_dense_mass_matrix
from src.trajectory_generation import make_quintic_trajectory


def beam_metrics(point, direction, tumour):
    vector=np.asarray(tumour)-np.asarray(point)
    direction=np.asarray(direction)/np.linalg.norm(direction)
    along=float(vector@direction)
    dist=float(np.linalg.norm(vector-along*direction))
    angle=float(np.degrees(np.arccos(np.clip(along/max(np.linalg.norm(vector),1e-12),-1.,1.))))
    return dist,angle,along>0


def pose_geometry_validation(kin, data, qadr, joint_angles, *, tolerance_pos=1e-5, tolerance_axis=1e-5):
    """Compare independently computed FK and beam axis with actual MuJoCo transforms."""
    import mujoco
    data.qpos[qadr]=joint_angles
    data.qvel[:]=0
    mujoco.mj_forward(kin.model,data)
    point,axis,_,_=kin.beam_state(joint_angles)
    geom_id=kin.tool_geom_id
    actual_point=data.geom_xpos[geom_id].copy()
    actual_axis=data.geom_xmat[geom_id].reshape(3,3)[:,0].copy()
    error_p=float(np.linalg.norm(point-actual_point))
    error_a=float(np.linalg.norm(axis-actual_axis))
    if error_p>tolerance_pos or error_a>tolerance_axis:
        raise RuntimeError('Independent FK / MuJoCo mismatch: '
                           f'point={error_p:.8g} m, direction={error_a:.8g}')
    return {'fk_tool_position_error_m':error_p,'beam_axis_vector_error':error_a}


def run_cyberknife_demo(project_root: Path, *, view=False, nodes=7, node_hold_s=2.,
                        max_joint_speed=0.22, max_settle_s=7.0, zero_gravity=False,
                        record=False):
    """Plan, execute and validate each treatment-*style* node; stop at first failure.

    `zero_gravity=True` is a gravity-free dynamics diagnostic (not a model of a
    real arm). The default retains gravity from the MuJoCo XML.
    """
    import mujoco
    from src.simulation import HOME_ANGLES, TORQUE_LIMITS, CONTROL_PERIOD
    if nodes<1 or nodes>13 or node_hold_s<=0 or max_joint_speed<=0 or max_settle_s<=0:
        raise ValueError('Invalid node or time arguments')
    root=Path(project_root)
    output=root/'results'/'cyberknife_motion'
    output.mkdir(parents=True,exist_ok=True)
    base_model=mujoco.MjModel.from_xml_path(str(root/'cyberknife_6dof.xml'))
    k0=RobotKinematics(base_model)
    home=HOME_ANGLES.copy()
    reference_q=np.clip(home+[.45,-.38,.32,.30,-.22,.18],
                        k0.joint_limits[:,0]+.03,k0.joint_limits[:,1]-.03)
    point0,axis0,_,_=k0.beam_state(reference_q)
    scene=create_brain_scene(root/'cyberknife_6dof.xml',root/'brain_patient_scene.xml',point0,axis0)
    checker=DetailedCollisionChecker(k0,scene.obstacles,link_radius=.012,clearance=.012,interpolation_step=.035)
    # Validate the starting configuration before constructing any simulated movement.
    start_ok,source,start_clearance=checker.check(home)
    if not start_ok:
        raise RuntimeError(f'Home configuration violates collision proxy {source}')
    viewpoints=[]
    rejected_candidates=[]
    # Reject infeasible viewpoint candidates BEFORE planning. A beam direction
    # can be IK-reachable yet cause a link to pass through the visible pillow.
    # Continue through the available illustrative directions to find `nodes`
    # feasible viewpoints; never silently accept a collision for a fixed count.
    for candidate_number,(yaw,pitch,axis) in enumerate(candidate_beam_directions(axis0),start=1):
        desired_point=scene.brain_target-.26*axis
        qseed=reference_q if not viewpoints else viewpoints[-1]['q']
        ik=solve_beam_ik(k0,desired_point,axis,qseed)
        if not ik.converged:
            ik=solve_beam_ik(k0,desired_point,axis,reference_q)
        if not ik.converged:
            rejected_candidates.append(dict(candidate=candidate_number,
                                            reason='position+axis IK failed'))
            continue
        okay,which,clearance=checker.check(ik.joint_angles)
        if not okay:
            rejected_candidates.append(dict(candidate=candidate_number,
                                            reason=f'proxy collision with {which}'))
            continue
        viewpoints.append(dict(index=len(viewpoints)+1,candidate=candidate_number,
                               yaw_offset_deg=yaw,pitch_offset_deg=pitch,
                               q=ik.joint_angles.copy(),desired_position=desired_point,
                               desired_axis=axis.copy(),clearance_m=clearance,
                               ik_position_error_mm=ik.position_error_m*1000,
                               ik_axis_error_deg=ik.pointing_error_deg))
        if len(viewpoints)>=nodes:
            break
    if len(viewpoints)<nodes:
        raise RuntimeError(f'Only {len(viewpoints)}/{nodes} illustrative beam directions '
                           f'are feasible under the current collision model: '
                           f'{rejected_candidates}')
    sections=[]
    q_from=home
    for i,node in enumerate(viewpoints):
        before=checker.checks
        plan=rrt_connect(q_from,node['q'],checker,max_iterations=5000,seed=23+i)
        if not plan.success:
            raise RuntimeError(f'No proxy-clear route to node {i+1}')
        trajectory=make_quintic_trajectory(plan.waypoints,max_joint_speed=max_joint_speed)
        sample_step=max(0.002,trajectory.duration/1000)
        for t in np.arange(0.,trajectory.duration+sample_step/2,sample_step):
            if not checker.valid(trajectory.evaluate(min(t,trajectory.duration))[0]):
                raise RuntimeError(f'Node {i+1} interpolated path violates collision proxies')
        sections.append((node,plan,trajectory,checker.checks-before))
        q_from=node['q']

    model=mujoco.MjModel.from_xml_path(str(scene.xml_path))
    if zero_gravity:
        model.opt.gravity[:]=0.0
    data=mujoco.MjData(model)
    kin=RobotKinematics(model)
    qadr=kin.qpos_addresses
    vadr=kin.dof_addresses
    ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'motor{i}') for i in range(1,7)])
    if np.any(ids<0):
        raise RuntimeError('Missing motor in MuJoCo XML')
    target_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'tumor_target')
    target_mocap=int(model.body_mocapid[target_body])
    model_kin=RobotKinematics(model)
    # Read-only independent NumPy FK check against MuJoCo geometry at many poses.
    verification=[]
    for q in [home,reference_q,*[v['q'] for v in viewpoints]]:
        verification.append(pose_geometry_validation(model_kin,data,qadr,q))
    data.qpos[qadr]=home
    data.qvel[vadr]=0.
    data.mocap_pos[target_mocap]=scene.brain_target
    mujoco.mj_forward(model,data)
    substeps=round(CONTROL_PERIOD/model.opt.timestep)
    if not np.isclose(substeps*model.opt.timestep,CONTROL_PERIOD) or substeps<1:
        raise RuntimeError('Control and physics timestep mismatch')
    # Keep original torque limits; never silently increase actuator capability.
    controller=ComputedTorquePD(TORQUE_LIMITS,natural_frequency=3.0,damping_ratio=1.0)
    joint_mass=np.zeros((model.nv,model.nv))
    viewer=None
    recorder=None
    if record:
        from src.video_capture import MuJoCoMP4Recorder
        recorder=MuJoCoMP4Recorder(
            model, output/'one_cycle.mp4',
            fps=20, width=960, height=540,
            lookat=scene.head_center+[0,.35,0],
            distance=3.0, azimuth=130, elevation=-22,
        )
        # Record the starting scene before the first step.
        set_beam_markers(model,data,scene.beam_names,
                         data.geom_xpos[kin.tool_geom_id].copy(),
                         data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0].copy())
        mujoco.mj_forward(model,data)
        recorder.capture(data)
    if view:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(model,data)
        viewer.cam.lookat[:]=scene.head_center+[0,.35,0]
        viewer.cam.distance=3.0
        viewer.cam.azimuth=130
        viewer.cam.elevation=-22
    rows=[]
    per_node=[]
    failure=None
    current_target=home
    collision_violations=0
    position_threshold_mm=10.0
    ray_threshold_mm=5.0
    velocity_threshold=0.08
    # Safety-like stop is diagnostic only, not a validated safety mechanism.
    speed_abort_rad_s=3.0

    def tick(node,phase,qcmd,dqcmd,ddqcmd=None):
        nonlocal collision_violations, failure
        for _ in range(substeps):
            # Refresh force and mass terms at the *current* state. After
            # mj_step() these fields can still describe the previous step.
            mujoco.mj_forward(model, data)
            q=data.qpos[qadr].copy()
            dq=data.qvel[vadr].copy()
            bias=data.qfrc_bias[vadr].copy()
            passive=data.qfrc_passive[vadr].copy()
            fill_dense_mass_matrix(mujoco, model, data, joint_mass)
            M=joint_mass[np.ix_(vadr,vadr)]
            tau=controller.calculate(q,dq,qcmd,dqcmd,bias,M,ddqcmd,passive=passive)
            data.ctrl[ids]=tau
            data.mocap_pos[target_mocap]=scene.brain_target
            mujoco.mj_step(model,data)
            # Monitor each 2-ms physics substep rather than only 20-ms control samples.
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                failure=f'Nonfinite state at node {node["index"]} ({phase})'
                break
            now_q=data.qpos[qadr].copy()
            now_speed=float(np.max(np.abs(data.qvel[vadr])))
            if now_speed>speed_abort_rad_s:
                failure=f'Overspeed {now_speed:.3f} rad/s at node {node["index"]} ({phase})'
                break
            now_clear,hit,_=checker.check(now_q)
            if not now_clear:
                collision_violations+=1
                failure=f'Dynamic proxy intersection with {hit} at node {node["index"]} ({phase})'
                break
        q=data.qpos[qadr].copy()
        dq=data.qvel[vadr].copy()
        tip=data.geom_xpos[kin.tool_geom_id].copy()
        direction=data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0].copy()
        set_beam_markers(model,data,scene.beam_names,tip,direction)
        if recorder is not None:
            mujoco.mj_forward(model,data)
            recorder.capture(data)
        if viewer is not None:
            # Refresh world coordinates for the marker mocap bodies before drawing.
            mujoco.mj_forward(model,data)
            viewer.sync()
            time.sleep(CONTROL_PERIOD)
        miss,ang,front=beam_metrics(tip,direction,scene.brain_target)
        pos_err=float(np.linalg.norm(tip-node['desired_position'])*1000)
        speed=float(np.max(np.abs(dq)))
        q_err=float(np.max(np.abs(q-qcmd)))
        clear,hit,clearance=checker.check(q)
        # Already counted substep-level proxy intersections above.
        row=dict(time_s=float(data.time),node=node['index'],phase=phase,
                 aimpoint_error_mm=pos_err,ray_miss_mm=miss*1000,
                 angular_miss_deg=ang,tool_speed_rad_s=speed,
                 max_joint_error_rad=q_err,target_in_front=front,proxy_clear=clear,
                 min_proxy_clearance_m=clearance,
                 torque_saturated=bool(np.any(np.abs(data.ctrl[ids])>=TORQUE_LIMITS-1e-7)),
                 max_passive_force_Nm=float(np.max(np.abs(passive))),
                 max_bias_force_Nm=float(np.max(np.abs(bias))),
                 **{f'q{i+1}_rad':float(q[i]) for i in range(6)},
                 **{f'q{i+1}_command_rad':float(qcmd[i]) for i in range(6)},
                 **{f'tau{i+1}_Nm':float(data.ctrl[ids[i]]) for i in range(6)})
        rows.append(row)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            failure='Nonfinite MuJoCo state'
        elif not clear:
            failure=f"Dynamic proxy intersection with {hit} at node {node['index']} ({phase})"
        elif speed>speed_abort_rad_s:
            failure=f"Overspeed {speed:.3f} rad/s at node {node['index']} ({phase})"
        return row,q,dq

    try:
        for node,plan,traj,nchecks in sections:
            gate=ArrivalGate(required_samples=max(1,int(np.ceil(.4/CONTROL_PERIOD))),
                             joint_tol=.015,speed_tol=velocity_threshold)
            last=None
            # Physical motion; reference positions and velocities from quintic trajectory.
            for step in range(int(np.ceil(traj.duration/CONTROL_PERIOD))+1):
                qcmd,dqcmd,ddqcmd=traj.evaluate(min(step*CONTROL_PERIOD,traj.duration))
                last,q,dq=tick(node,'move',qcmd,dqcmd,ddqcmd)
                if failure: break
            arrived=False
            if not failure:
                for step in range(int(np.ceil(max_settle_s/CONTROL_PERIOD))):
                    last,q,dq=tick(node,'settling',node['q'],np.zeros(6))
                    if failure: break
                    # A viewpoint can only be accepted if BOTH physical joint
                    # settling AND actual geometric aim checks are persistent.
                    good=(last['target_in_front'] and last['proxy_clear'] and
                          last['aimpoint_error_mm']<=position_threshold_mm and
                          last['ray_miss_mm']<=ray_threshold_mm)
                    if gate.update(q,dq,node['q'],additional_ok=good):
                        arrived=True
                        break
            if not arrived and not failure:
                failure=f"Node {node['index']} did not settle/align within {max_settle_s:g} simulated seconds"
            if arrived and not failure:
                for step in range(int(np.ceil(node_hold_s/CONTROL_PERIOD))):
                    last,q,dq=tick(node,'aim_hold',node['q'],np.zeros(6))
                    if failure: break
                    if (last['aimpoint_error_mm']>position_threshold_mm or
                        last['ray_miss_mm']>ray_threshold_mm or
                        last['tool_speed_rad_s']>velocity_threshold):
                        failure=f"Node {node['index']} lost alignment during hold"
                        break
            samples=[r for r in rows if r['node']==node['index']]
            endpoint=samples[-1] if samples else {}
            per_node.append(dict(node=node['index'],beam_candidate=node['candidate'],
                                 yaw_offset_deg=node['yaw_offset_deg'],
                                 pitch_offset_deg=node['pitch_offset_deg'],
                                 planning_iterations=plan.iterations,
                                 waypoints=len(plan.waypoints),collision_checks=nchecks,
                                 planned_approach_s=float(traj.duration),
                                 geometric_ik_position_error_mm=node['ik_position_error_mm'],
                                 geometric_ik_axis_error_deg=node['ik_axis_error_deg'],
                                 achieved=bool(arrived and not failure),
                                 last_aimpoint_error_mm=endpoint.get('aimpoint_error_mm'),
                                 last_ray_miss_mm=endpoint.get('ray_miss_mm'),
                                 last_angular_miss_deg=endpoint.get('angular_miss_deg'),
                                 last_joint_speed_rad_s=endpoint.get('tool_speed_rad_s'),
                                 last_joint_error_rad=endpoint.get('max_joint_error_rad'),
                                 final_actual_joints_rad=[endpoint.get(f'q{i+1}_rad') for i in range(6)],
                                 final_command_joints_rad=[endpoint.get(f'q{i+1}_command_rad') for i in range(6)],
                                 final_max_passive_force_Nm=endpoint.get('max_passive_force_Nm'),
                                 final_max_bias_force_Nm=endpoint.get('max_bias_force_Nm'),
                                 minimum_proxy_clearance_m=min(r['min_proxy_clearance_m'] for r in samples)
                                 if samples else None))
            if failure: break
    finally:
        # Make sure the viewer is closed even when an execution gate fails.
        if viewer is not None:
            viewer.close()
        if recorder is not None:
            recorder.close()
    result=dict(task='Educational synthetic brain-pointing demo; no radiotherapy',
                requested_nodes=nodes,completed_nodes=sum(n['achieved'] for n in per_node),
                successful=(failure is None and len(per_node)==nodes),
                failure_reason=failure,
                video_file=str(output/'one_cycle.mp4') if record else None,
                video_frames=recorder.frame_count if recorder is not None else 0,
                zero_gravity_diagnostic=bool(zero_gravity),
                fk_mujoco_max_position_error_m=max(x['fk_tool_position_error_m'] for x in verification),
                fk_mujoco_max_beam_direction_error=max(x['beam_axis_vector_error'] for x in verification),
                planning_collision_model='Joint-link capsules vs head/neck/nose spheres + table/mattress/pillow/torso/pelvis/limbs boxes',
                dynamic_proxy_violation_samples=collision_violations,
                controller_type="inertia-scaled computed-torque PD with passive damping compensation (MuJoCo M/bias/passive; NumPy feedback)",
                controller_natural_frequency_rad_s=controller.natural_frequency,
                controller_damping_ratio=controller.damping_ratio,
                actuator_samples=controller.diag.samples,
                actuator_saturated_steps=controller.diag.saturated_samples,
                max_joint_speed_seen_rad_s=controller.diag.max_speed,
                max_abs_bias_torque_Nm=controller.diag.max_abs_bias,
                max_abs_passive_force_Nm=controller.diag.max_abs_passive,
                viewpoint_metrics=per_node,
                skipped_infeasible_viewpoint_candidates=rejected_candidates,
                limitations=['Approximate collision proxies; no robot self-collision or exact mesh/dynamic continuous collision proof',
                             'No clinical treatment accuracy, patient registration, radiation or dose model',
                             'Failure at any node stops the sequence; alignment is not assumed from IK alone'])
    if rows:
        with (output/'multinode_samples.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    (output/'multinode_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
        t=[r['time_s'] for r in rows]
        axes[0].plot(t,[r['ray_miss_mm'] for r in rows],label='beam ray miss mm')
        axes[0].plot(t,[r['aimpoint_error_mm'] for r in rows],label='tool position error mm')
        axes[0].legend();axes[0].set_ylabel('Error (mm)')
        axes[1].plot(t,[r['tool_speed_rad_s'] for r in rows],label='Max abs joint speed rad/s')
        axes[1].axhline(velocity_threshold,color='gray',linestyle='--',label='settling target')
        axes[1].legend();axes[1].set_ylabel('Speed (rad/s)')
        axes[2].plot(t,[r['max_joint_error_rad'] for r in rows],label='Max joint error rad')
        axes[2].plot(t,[float(r['torque_saturated']) for r in rows],label='Torque limit reached (0/1)',alpha=.7)
        axes[2].set_ylabel('Error / saturation');axes[2].set_xlabel('Simulation time (s)');axes[2].legend()
        for ax in axes:ax.grid(alpha=.3)
        fig.suptitle('Educational node aiming — geometry, settling and torque diagnostics')
        fig.savefig(output/'multinode_motion.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))
    if failure:
        print('SIMULATION FAILED (correctly stopped):',failure)
    else:
        print('All requested illustrative node criteria passed under this simulation configuration.')
    print('Results written to:',output)
    return result
