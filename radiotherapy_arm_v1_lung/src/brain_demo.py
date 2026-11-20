"""RRT approach to a *synthetic* brain target, then stationary point aiming.

Usage: python robot_simulation.py --brain-view (or --brain headless)

The beam dots indicate a geometric line between tool and tumour; radiation
physics, beam-axis orientation, collision safety and treatment plans are absent.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import numpy as np

from src.brain_scene import create_brain_scene, set_beam_markers
from src.kinematics import RobotKinematics
from src.inverse_kinematics import solve_position_ik
from src.path_planning import CollisionChecker, rrt_connect
from src.trajectory_generation import make_quintic_trajectory


def run_brain_demo(project_root: Path, *, view=False, hold_duration=5.0, seed=7):
    import mujoco
    from src.simulation import HOME_ANGLES, KP, KD, TORQUE_LIMITS, CONTROL_PERIOD
    project_root=Path(project_root)
    output=project_root/'results'/'brain_targeting'
    output.mkdir(parents=True,exist_ok=True)
    baseline=mujoco.MjModel.from_xml_path(str(project_root/'cyberknife_6dof.xml'))
    kin0=RobotKinematics(baseline)
    q_home=HOME_ANGLES.copy()
    q_goal=np.clip(q_home+np.array([0.45,-0.38,0.32,0.30,-0.22,0.18]),
                   kin0.joint_limits[:,0]+0.03,kin0.joint_limits[:,1]-0.03)
    # Reference tool orientation comes from the physical MuJoCo joint geometry;
    # by convention this educational scene treats tool-local +X as beam axis.
    goal_data=mujoco.MjData(baseline)
    goal_data.qpos[kin0.qpos_addresses]=q_goal
    mujoco.mj_forward(baseline,goal_data)
    desired_tool=goal_data.geom_xpos[kin0.tool_geom_id].copy()
    goal_axis=goal_data.geom_xmat[kin0.tool_geom_id].reshape(3,3)[:,0].copy()
    scene=create_brain_scene(project_root/'cyberknife_6dof.xml',
                             project_root/'brain_patient_scene.xml',desired_tool,
                             beam_axis_world=goal_axis)
    # The planners' capsule-link approximation is used for both endpoints
    # and all interpolated states; the visual patient geoms are contact-disabled.
    checker=CollisionChecker(kin0,scene.obstacles,link_radius=0.012,clearance=0.012)
    if not checker.valid(q_home) or not checker.valid(q_goal):
        raise RuntimeError('Patient/table proxy overlaps an approach endpoint; adjust scene coordinates')
    plan=rrt_connect(q_home,q_goal,checker,seed=seed,max_iterations=5000)
    if not plan.success:
        raise RuntimeError('RRT could not find a patient-proxy-clear approach. Revise geometry.')
    traj=make_quintic_trajectory(plan.waypoints,max_joint_speed=0.45)
    for t in np.linspace(0,traj.duration,max(100,int(traj.duration/0.01)+1)):
        if not checker.valid(traj.evaluate(t)[0]):
            raise RuntimeError('Quintic approach violates simplified patient collision proxies')

    model=mujoco.MjModel.from_xml_path(str(scene.xml_path))
    data=mujoco.MjData(model)
    kin=RobotKinematics(model)
    qpos, qvel=kin.qpos_addresses,kin.dof_addresses
    actuator_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,
                                             f'motor{i}') for i in range(1,7)])
    target_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'tumor_target')
    target_mocap=int(model.body_mocapid[target_body])
    data.qpos[qpos]=q_home
    data.mocap_pos[target_mocap]=scene.brain_target
    mujoco.mj_forward(model,data)
    home_axis=data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0]
    set_beam_markers(model,data,scene.beam_names,kin.forward(q_home),home_axis)
    mujoco.mj_forward(model,data)

    # Confirm that a *position-only* inverse kinematics problem is coherent.
    ik=solve_position_ik(kin,scene.tool_aim_point,q_goal,method='dls')
    if not ik.converged:
        raise RuntimeError(f'Treatment standoff point is not IK reachable: {ik.position_error:.5f} m')

    physics_dt=float(model.opt.timestep)
    substeps=round(CONTROL_PERIOD/physics_dt)
    if substeps<1 or not np.isclose(substeps*physics_dt,CONTROL_PERIOD):
        raise ValueError('MuJoCo timestep does not divide control period')

    viewer=None
    if view:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(model,data)
        viewer.cam.lookat[:]=scene.head_center+[0,0.35,0]
        viewer.cam.distance=3.0
        viewer.cam.azimuth=130
        viewer.cam.elevation=-22

    rows=[]
    duration=traj.duration+hold_duration
    total_steps=int(np.ceil(duration/CONTROL_PERIOD))
    try:
        for k in range(total_steps):
            t=k*CONTROL_PERIOD
            if t<traj.duration:
                phase='approach'
                desired_q,desired_dq,_=traj.evaluate(t)
            else:
                phase='brain_aim_hold'
                desired_q=ik.joint_angles
                desired_dq=np.zeros(6)
            for _ in range(substeps):
                q=data.qpos[qpos].copy()
                dq=data.qvel[qvel].copy()
                bias=data.qfrc_bias[qvel].copy()
                tau=bias+1.5*(KP*(desired_q-q)+KD*(desired_dq-dq))
                data.ctrl[actuator_ids]=np.clip(tau,-TORQUE_LIMITS,TORQUE_LIMITS)
                data.mocap_pos[target_mocap]=scene.brain_target
                mujoco.mj_step(model,data)
                tool_now=data.geom_xpos[kin.tool_geom_id]
                actual_axis=data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0]
                set_beam_markers(model,data,scene.beam_names,tool_now,actual_axis)
            actual=data.geom_xpos[kin.tool_geom_id].copy()
            reference=kin.forward(desired_q)
            aim_pos_error=float(np.linalg.norm(actual-scene.tool_aim_point))
            beam_axis=data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0]
            tumour_relative=scene.brain_target-actual
            forward_distance=float(tumour_relative@beam_axis)
            ray_miss=float(np.linalg.norm(tumour_relative-forward_distance*beam_axis))
            rows.append({'time_s':float(data.time),'phase':phase,
                         'tool_reference_error_mm':float(np.linalg.norm(actual-reference)*1000),
                         'tool_aimpoint_error_mm':aim_pos_error*1000,
                         'brain_target_distance_mm':float(np.linalg.norm(actual-scene.brain_target))*1000,
                         'beam_ray_miss_mm':ray_miss*1000,
                         'tumour_ahead_of_beam':forward_distance>0,
                         **{f'q{i+1}_rad':float(data.qpos[qpos[i]]) for i in range(6)},
                         **{f'q{i+1}_desired_rad':float(desired_q[i]) for i in range(6)},
                         **{f'tau{i+1}_Nm':float(data.ctrl[actuator_ids[i]]) for i in range(6)}})
            if viewer is not None:
                viewer.sync()
                time.sleep(CONTROL_PERIOD)
    finally:
        if viewer is not None:
            viewer.close()

    with (output/'brain_targeting_samples.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    held=np.array([r['tool_aimpoint_error_mm'] for r in rows if r['phase']=='brain_aim_hold'])
    last1=np.array([r['tool_aimpoint_error_mm'] for r in rows if r['time_s']>=duration-1.0])
    final_err=float(rows[-1]['tool_aimpoint_error_mm'])
    miss_held=np.array([r['beam_ray_miss_mm'] for r in rows if r['phase']=='brain_aim_hold'])
    beam_ahead=bool(rows[-1]['tumour_ahead_of_beam'])
    summary={'task':'Synthetic brain tumour point-aiming (not radiotherapy treatment)',
             'tumour_world_m':scene.brain_target.tolist(),
             'head_center_world_m':scene.head_center.tolist(),
             'tool_standoff_target_world_m':scene.tool_aim_point.tolist(),
             'nominal_tool_to_tumour_distance_m':float(np.linalg.norm(scene.brain_target-scene.tool_aim_point)),
             'plan':'NumPy RRT-Connect with approximate head, torso, table collision spheres',
             'path_collision_validated_in_planner':True,
             'mesh_and_dynamic_collision_validated':False,
             'reference_goal_tool_axis_world':scene.beam_axis_world_at_goal.tolist(),
             'final_beam_ray_miss_mm':float(rows[-1]['beam_ray_miss_mm']),
             'hold_rms_beam_ray_miss_mm':float(np.sqrt(np.mean(miss_held**2))),
             'tumour_in_front_of_beam_at_end':beam_ahead,
             'tool_orientation_actively_controlled':False,
             'beam_is_visual_only':True,
             'rrt_iterations':plan.iterations,
             'planning_time_s':plan.elapsed_s,
             'collision_checks':checker.checks,
             'path_waypoints':len(plan.waypoints),
             'approach_duration_s':traj.duration,
             'hold_duration_s':hold_duration,
             'hold_rms_tool_aim_error_mm':float(np.sqrt(np.mean(held**2))),
             'final_tool_aim_error_mm':final_err,
             'last_1s_rms_tool_aim_error_mm':float(np.sqrt(np.mean(last1**2))),
             'within_25mm_aimpoint_at_end':final_err<=25.0,
             'note':'Geometric point targeting only. No treatment dose, no tool angular alignment, no safety claim.'}
    (output/'brain_targeting_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savetxt(output/'brain_planned_waypoints.csv',plan.waypoints,delimiter=',',
               header=','.join(f'q{i+1}_rad' for i in range(6)),comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    tt=[r['time_s'] for r in rows]
    ax[0].plot(tt,[r['tool_reference_error_mm'] for r in rows],label='tool-to-reference error')
    ax[0].plot(tt,[r['tool_aimpoint_error_mm'] for r in rows],label='tool-to-aimpoint error',alpha=.7)
    ax[0].plot(tt,[r['beam_ray_miss_mm'] for r in rows],label='beam-axis miss distance',alpha=.7)
    ax[0].axvline(traj.duration,linestyle='--',label='approach to hold')
    ax[0].set(ylabel='Error (mm)',title='Synthetic brain targeting (position-only)')
    ax[0].legend()
    for i in range(6):
        ax[1].plot(tt,[r[f'q{i+1}_rad'] for r in rows],label=f'J{i+1}')
    ax[1].axvline(traj.duration,linestyle='--')
    ax[1].set(xlabel='Simulation time (s)',ylabel='Actual joint angle (rad)')
    ax[1].legend(ncol=6)
    fig.savefig(output/'brain_targeting.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    if not summary['within_25mm_aimpoint_at_end'] or not beam_ahead or summary['final_beam_ray_miss_mm']>10:
        print('WARNING: geometric beam aim may be inaccurate; orientation is not actively controlled.')
    if not summary['within_25mm_aimpoint_at_end']:
        print('WARNING: robot did not reach the illustrative aimpoint within 25 mm. Check torque saturation and model dynamics.')
    print(f'Brain-targeting results written to {output}')
    return summary
