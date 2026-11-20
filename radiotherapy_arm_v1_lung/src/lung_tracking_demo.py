"""Respiration-compensated virtual lung-tumour aim using a synthetic chest marker.

This is an educational demonstration only: no camera, imaging, clinical tumour
localization, radiation, dose delivery or patient safety guarantees. A synthetic
external chest marker is measured with noise and mapped to a simulated internal
lesion through an assumed, calibrated breathing-motion correlation.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
from src.beam_ik import solve_beam_ik
from src.brain_scene import set_beam_markers
from src.kinematics import RobotKinematics
from src.lung_observer import (ChestMarkerObserver,LungMotionCalibration,
                               synthetic_tumour,external_marker_height)
from src.lung_scene import create_lung_scene, relocate_robot_base
from src.mujoco_compat import fill_dense_mass_matrix
from src.vision_servo import FixedEyeToHandCamera, image_based_joint_velocity
from src.path_planning import rrt_connect
from src.robust_collision import DetailedCollisionChecker
from src.stable_joint_control import ComputedTorquePD, ArrivalGate
from src.trajectory_generation import make_quintic_trajectory


def ray_miss(point,axis,target):
    v=np.asarray(target)-np.asarray(point)
    d=np.asarray(axis)/np.linalg.norm(axis)
    proj=float(v@d)
    perpendicular=v-max(0.,proj)*d
    return float(np.linalg.norm(perpendicular)),proj>0


def _save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_lung_tracking(project_root:Path, *, view=False, compare=True, record=False,
                      duration_s=12.0, sensor_noise_mm=0.35,
                      period_s=4.0, seed=2026, vision_mode=False):
    """Approach a nominal source pose; estimate breathing from an external marker.

    compare=True performs a separate STATIC BEAM baseline run, same breathing
    waveform/model/control with compensation disabled. The viewer/video show only
    the compensated run (because a simultaneous scene is not appropriate here).
    """
    import mujoco
    from src.simulation import HOME_ANGLES,TORQUE_LIMITS,CONTROL_PERIOD
    from src.cyberknife_motion_demo import pose_geometry_validation
    if duration_s<=0 or sensor_noise_mm<0 or period_s<=0:
        raise ValueError('Invalid duration, noise or period')
    root=Path(project_root)
    output=root/'results'/'lung_tracking'
    output.mkdir(parents=True,exist_ok=True)
    robot_xml=root/'cyberknife_6dof.xml'
    robot=mujoco.MjModel.from_xml_path(str(robot_xml))
    k0=RobotKinematics(robot)
    home=HOME_ANGLES.copy()
    q_ref=np.clip(home+[.45,-.38,.32,.30,-.22,.18],
                  k0.joint_limits[:,0]+.03,k0.joint_limits[:,1]-.03)
    tool_ref,axis_ref,_,_=k0.beam_state(q_ref)
    # The target and patient are fixed in world coordinates by the original
    # (unmoved) robot reference tool pose. Moving the base MUST force new IK.
    # A deterministic placement search first checks collisions at HOME; only
    # collision-free homes proceed to goal IK, RRT and trajectory validation.
    # Do not try to solve base collisions by silently removing patient arms.
    base_offsets = (
        (0.0,0.0,0.0),
        (-0.20,0.0,0.0),(0.20,0.0,0.0),
        (-0.35,0.0,0.0),(0.35,0.0,0.0),
        (-0.50,0.0,0.0),(0.50,0.0,0.0),
        (-0.65,0.0,0.0),(0.65,0.0,0.0),
        (-0.30,-0.25,0.0),(-0.30,0.25,0.0),
        (0.30,-0.25,0.0),(0.30,0.25,0.0),
        (-0.50,-0.25,0.0),(-0.50,0.25,0.0),
        (0.50,-0.25,0.0),(0.50,0.25,0.0),
        (-0.65,-0.35,0.0),(-0.65,0.35,0.0),
        (0.65,-0.35,0.0),(0.65,0.35,0.0),
    )
    arm_offsets=((0.0,0.0),(0.12,0.12),(0.25,0.25))
    rejected=[]
    scene=checker=plan=trajectory=None
    selected_kinematics=None
    original_tool=tool_ref.copy()
    original_axis=axis_ref.copy()
    target_world=original_tool+0.26*original_axis
    for offset in base_offsets:
        for spread in arm_offsets:
            candidate_scene=create_lung_scene(
                robot_xml,root/'lung_patient_scene.xml',original_tool,original_axis,
                arm_spread_m=spread)
            candidate_scene=relocate_robot_base(candidate_scene,offset)
            # Crucial: load the *relocated* XML and rebuild NumPy kinematics.
            candidate_model=mujoco.MjModel.from_xml_path(str(candidate_scene.xml_path))
            candidate_kin=RobotKinematics(candidate_model)
            candidate_checker=DetailedCollisionChecker(
                candidate_kin,candidate_scene.obstacles,link_radius=.012,
                clearance=.012,interpolation_step=.035)
            valid,hit,_=candidate_checker.check(home)
            if not valid:
                rejected.append(f'base={offset}, arms={spread}: home near {hit}')
                continue
            # Preserve the same WORLD tumour and source goal; solve a new joint
            # configuration after translating the robot base.
            if np.allclose(offset,0.):
                goal=q_ref.copy()
            else:
                solutions=[]
                for guess in (q_ref,home, np.clip(q_ref+[0.12,-0.1,0.1,0,0,0],
                        candidate_kin.joint_limits[:,0]+.02,
                        candidate_kin.joint_limits[:,1]-.02)):
                    candidate=solve_beam_ik(candidate_kin,original_tool,original_axis,
                                            guess,max_iterations=220)
                    if candidate.converged and candidate_checker.valid(candidate.joint_angles):
                        solutions.append(candidate.joint_angles.copy())
                if not solutions:
                    rejected.append(f'base={offset}, arms={spread}: goal IK infeasible/colliding')
                    continue
                goal=min(solutions,key=lambda q:np.linalg.norm(q-home))
            valid,hit,_=candidate_checker.check(goal)
            if not valid:
                rejected.append(f'base={offset}, arms={spread}: goal near {hit}')
                continue
            candidate_plan=rrt_connect(home,goal,candidate_checker,seed=42,max_iterations=5000)
            if not candidate_plan.success:
                rejected.append(f'base={offset}, arms={spread}: no collision-free RRT path')
                continue
            candidate_trajectory=make_quintic_trajectory(
                candidate_plan.waypoints,max_joint_speed=.22)
            times=np.linspace(0,candidate_trajectory.duration,
                       max(30,int(candidate_trajectory.duration/.01)))
            if not all(candidate_checker.valid(candidate_trajectory.evaluate(t)[0])
                       for t in times):
                rejected.append(f'base={offset}, arms={spread}: trajectory collision')
                continue
            scene,checker,plan,trajectory=(candidate_scene,candidate_checker,
                                           candidate_plan,candidate_trajectory)
            selected_kinematics=candidate_kin
            q_ref=goal.copy()
            break
        if scene is not None:
            break
    if scene is None:
        raise RuntimeError(
            'No collision-clear robot-base placement, IK solution, and approach found. '
            'The patient and collision checks were kept in place. '
            'Recent rejected placements: '+'; '.join(rejected[-12:]))
    print(f'Lung scene: robot base offset = {scene.base_offset_world_m} m; '
          f'arm offsets = {scene.arm_spread_m} m; '
          f'RRT waypoints = {len(plan.waypoints)}; '
          f'rejected placement attempts = {len(rejected)}.')
    calibration=LungMotionCalibration(period_s=float(period_s))
    camera=FixedEyeToHandCamera.beside_patient(scene.chest_marker_origin) if vision_mode else None

    def execute(compensate:bool,*,show=False,write_video=False):
        model=mujoco.MjModel.from_xml_path(str(scene.xml_path))
        data=mujoco.MjData(model)
        kin=RobotKinematics(model)
        qadr=kin.qpos_addresses;vadr=kin.dof_addresses
        actuators=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'motor{i}')
                            for i in range(1,7)])
        if np.any(actuators<0):raise RuntimeError('Motor missing from model')
        target_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'tumor_target')
        chest_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'chest_observation_marker')
        target_mocap=int(model.body_mocapid[target_body])
        chest_mocap=int(model.body_mocapid[chest_body])
        surface_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'breathing_chest_surface')
        surface_mocap=int(model.body_mocapid[surface_body])
        pose_geometry_validation(kin,data,qadr,q_ref)
        data.qpos[qadr]=home
        data.qvel[vadr]=0.
        mujoco.mj_forward(model,data)
        substeps=round(CONTROL_PERIOD/model.opt.timestep)
        if substeps<1 or not np.isclose(CONTROL_PERIOD,substeps*model.opt.timestep):
            raise RuntimeError('Physics and command timesteps incompatible')
        mass=np.zeros((model.nv,model.nv))
        controller=ComputedTorquePD(TORQUE_LIMITS,natural_frequency=3.,damping_ratio=1.)
        observer=ChestMarkerObserver(scene.tumour_anchor,calibration)
        rng=np.random.default_rng(seed)
        viewer=None;recorder=None
        if show:
            from mujoco import viewer as mjviewer
            viewer=mjviewer.launch_passive(model,data)
            viewer.cam.lookat[:]=scene.torso_center+[0,.3,0]
            viewer.cam.distance=3.
            viewer.cam.azimuth=130
            viewer.cam.elevation=-22
        if write_video:
            from src.video_capture import MuJoCoMP4Recorder
            recorder=MuJoCoMP4Recorder(model,output/'lung_compensation.mp4',
                     fps=20,width=960,height=540,
                     lookat=scene.torso_center+[0,.3,0],distance=3.,
                     azimuth=130,elevation=-22)
        rows=[]
        failure=None
        last_qcmd=home.copy()
        q_goal=q_ref.copy()
        fallback_count=0
        sample_index=0
        last_image_error_px=np.nan
        last_image_command_rad_s=np.nan
        last_marker_pixel=np.array([np.nan,np.nan])
        def observe_external_chest(t):
            """Read the simulated external 3D chest-marker coordinate only.

            In a future camera-based implementation this one read would be
            replaced by tracked image feature coordinates after calibration.
            """
            u=external_marker_height(t,calibration)
            data.mocap_pos[chest_mocap]=scene.chest_marker_origin+np.array([0,0,u])
            data.mocap_pos[surface_mocap]=scene.surface_origin+np.array([0,0,u])
            if vision_mode:
                # L33 pinhole projection: image features are synthetic pixels.
                # Image noise is equivalent to sensor_noise_mm at the marker depth.
                nonlocal last_marker_pixel
                pixel=camera.project(data.mocap_pos[chest_mocap])
                marker_depth=camera.camera_xyz(scene.chest_marker_origin)[2]
                pixel_std=sensor_noise_mm/1000*camera.focal_px/marker_depth
                pixel_noisy=pixel+rng.normal(0,pixel_std,size=2)
                last_marker_pixel=pixel_noisy
                return u,camera.recover_vertical_excursion(pixel_noisy,scene.chest_marker_origin)
            measured=float(data.mocap_pos[chest_mocap,2]-scene.chest_marker_origin[2])
            return u,measured+rng.normal(0,sensor_noise_mm/1000)

        def step(phase,q_cmd,vel_cmd,acc_cmd,truth,estimate,marker_height):
            nonlocal failure,last_qcmd,sample_index
            # The external marker is the synthetic 'measurement'. This scalar is
            # generated independently from the tumour target's mocap position.
            data.mocap_pos[chest_mocap]=scene.chest_marker_origin+np.array([0,0,marker_height])
            data.mocap_pos[target_mocap]=truth
            for _ in range(substeps):
                mujoco.mj_forward(model,data)
                q=data.qpos[qadr].copy();dq=data.qvel[vadr].copy()
                fill_dense_mass_matrix(mujoco,model,data,mass)
                M=mass[np.ix_(vadr,vadr)]
                tau=controller.calculate(q,dq,q_cmd,vel_cmd,
                                         data.qfrc_bias[vadr].copy(),M,acc_cmd,
                                         passive=data.qfrc_passive[vadr].copy())
                data.ctrl[actuators]=tau
                mujoco.mj_step(model,data)
                if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                    failure=f'Nonfinite state during {phase}'
                    break
                if np.max(np.abs(data.qvel[vadr]))>3.0:
                    failure=f'Overspeed during {phase}'
                    break
                valid,hit,_=checker.check(data.qpos[qadr])
                if not valid:
                    failure=f'Proxy collision with {hit} during {phase}'
                    break
            mujoco.mj_forward(model,data)
            p=data.geom_xpos[kin.tool_geom_id].copy()
            d=data.geom_xmat[kin.tool_geom_id].reshape(3,3)[:,0].copy()
            set_beam_markers(model,data,scene.beam_names,p,d)
            mujoco.mj_forward(model,data)
            miss,front=ray_miss(p,d,truth)
            est_miss,_=ray_miss(p,d,estimate)
            q=data.qpos[qadr].copy();dq=data.qvel[vadr].copy()
            r=dict(time_s=float(data.time),phase=phase,
                   breathing_measurement_mm=float(marker_height*1000),
                   estimated_period_s=float(observer.period_estimate or 0),
                   observer_ready=int(observer.ready),
                   camera_marker_u_px=float(last_marker_pixel[0]),
                   camera_marker_v_px=float(last_marker_pixel[1]),
                   image_error_px=float(last_image_error_px),
                   ibvs_joint_speed_cmd_rad_s=float(last_image_command_rad_s),
                   estimation_error_mm=float(1000*np.linalg.norm(estimate-truth)),
                   true_beam_miss_mm=float(miss*1000),
                   estimated_beam_miss_mm=float(est_miss*1000),
                   tumour_in_front=bool(front),
                   max_joint_error_rad=float(np.max(np.abs(q-q_cmd))),
                   max_joint_speed_rad_s=float(np.max(np.abs(dq))),
                   **{f'true_{a}_m':float(truth[j]) for j,a in enumerate('xyz')},
                   **{f'estimated_{a}_m':float(estimate[j]) for j,a in enumerate('xyz')},
                   **{f'q{i+1}_rad':float(q[i]) for i in range(6)},
                   **{f'q{i+1}_cmd_rad':float(q_cmd[i]) for i in range(6)})
            rows.append(r)
            if viewer:
                viewer.sync()
                time.sleep(CONTROL_PERIOD)
            if recorder:recorder.capture(data)
            last_qcmd=q_cmd.copy()
            sample_index+=1
            return r

        try:
            # First move to the stationary reference pose using planned quintic
            # motion. At the same time, observe the chest signal to calibrate.
            for i in range(int(np.ceil(trajectory.duration/CONTROL_PERIOD))+1):
                t=float(data.time)
                u,y=observe_external_chest(t)
                observer.update(t,y)
                truth=synthetic_tumour(t,scene.tumour_anchor,calibration)
                estimate=observer.estimate(t)
                qc,vc,ac=trajectory.evaluate(min(i*CONTROL_PERIOD,trajectory.duration))
                step('approach',qc,vc,ac,truth,estimate,u)
                if failure:break
            gate=ArrivalGate(required_samples=max(1,int(.4/CONTROL_PERIOD)),
                             joint_tol=.015,speed_tol=.08)
            if not failure:
                for _ in range(int(7/CONTROL_PERIOD)):
                    t=float(data.time)
                    u,y=observe_external_chest(t)
                    observer.update(t,y)
                    truth=synthetic_tumour(t,scene.tumour_anchor,calibration)
                    estimate=observer.estimate(t)
                    step('settling',q_ref,np.zeros(6),np.zeros(6),truth,estimate,u)
                    if failure:break
                    if gate.update(data.qpos[qadr],data.qvel[vadr],q_ref):break
                else:
                    failure='Robot failed to settle at nominal beam source'
            if not failure and not observer.ready:
                failure='Chest-motion estimator did not receive enough observation history'
            # Actual compensation is ONLY during this tracking interval.
            q_goal=q_ref.copy()
            if not failure:
                for _ in range(int(np.ceil(duration_s/CONTROL_PERIOD))):
                    t=float(data.time)
                    u,y=observe_external_chest(t)
                    observer.update(t,y)
                    truth=synthetic_tumour(t,scene.tumour_anchor,calibration)
                    estimate=observer.estimate(t,prediction_horizon_s=.02)
                    if compensate:
                        desired_axis=estimate-tool_ref
                        desired_axis/=np.linalg.norm(desired_axis)
                        ik=solve_beam_ik(kin,tool_ref,desired_axis,q_goal,
                                         max_iterations=100)
                        if ik.converged and checker.valid(ik.joint_angles):
                            q_goal=ik.joint_angles.copy()
                        else:
                            fallback_count+=1  # fail closed: hold previous command
                        if vision_mode:
                            # L37 fixed eye-to-hand IBVS. Visual error is based on
                            # the *actual* virtual beam endpoint in the image and
                            # the observer's estimated tumour projection.
                            # 2-D feedback only: the 3-D IK anchors the remaining
                            # unobservable depth / beam-source standoff DOFs.
                            q_measured=data.qpos[qadr].copy()
                            try:
                                dq_img,e_img,_,_,_=image_based_joint_velocity(
                                    camera,kin,q_measured,estimate)
                                last_image_error_px=float(np.linalg.norm(e_img))
                                last_image_command_rad_s=float(np.linalg.norm(dq_img))
                                q_ibvs=np.clip(q_goal+.02*dq_img,
                                    kin.joint_limits[:,0]+.001,
                                    kin.joint_limits[:,1]-.001)
                                if checker.valid(q_ibvs):
                                    q_goal=q_ibvs
                            except ValueError:
                                # Out-of-view target: do not use invalid pixels.
                                fallback_count+=1
                    # Velocity feedforward from finite differences, with clipping
                    # against noisy instantaneous IK jumps. No acceleration FF here.
                    q_now=np.clip(q_goal,last_qcmd-.006,last_qcmd+.006)
                    v_now=np.clip((q_now-last_qcmd)/CONTROL_PERIOD,-.30,.30)
                    step('track' if compensate else 'static_baseline',
                         q_now,v_now,np.zeros(6),truth,estimate,u)
                    if failure:break
        finally:
            if viewer:viewer.close()
            if recorder:recorder.close()
        series=[r for r in rows if r['phase'] in ('track','static_baseline')]
        errs=np.array([r['true_beam_miss_mm'] for r in series],float)
        est=np.array([r['estimation_error_mm'] for r in series],float)
        result=dict(mode='compensated' if compensate else 'static_baseline',
                    visual_servoing_enabled=bool(vision_mode and compensate),
                    visual_error_last_px=(float(last_image_error_px) if np.isfinite(last_image_error_px) else None),
                    successful=(failure is None and len(series)>0),failure_reason=failure,
                    tracking_samples=len(series),
                    beam_miss_rms_mm=float(np.sqrt(np.mean(errs**2))) if len(errs) else None,
                    beam_miss_p95_mm=float(np.percentile(errs,95)) if len(errs) else None,
                    beam_miss_max_mm=float(np.max(errs)) if len(errs) else None,
                    tumour_estimation_rms_mm=float(np.sqrt(np.mean(est**2))) if len(est) else None,
                    estimated_breath_period_s=observer.period_estimate,
                    ik_fallbacks=fallback_count,
                    actuator_saturated_steps=controller.diag.saturated_samples,
                    max_joint_speed_rad_s=controller.diag.max_speed,
                    path_waypoints=len(plan.waypoints),
                    planning_iterations=plan.iterations,
                    approach_duration_s=float(trajectory.duration),
                    robot_base_offset_world_m=list(scene.base_offset_world_m),
                    patient_arm_spread_m=list(scene.arm_spread_m))
        _save_csv(output/('compensated.csv' if compensate else 'static_baseline.csv'),rows)
        return result,rows

    compensated,comp_rows=execute(True,show=view,write_video=record)
    baseline=None;baseline_rows=[]
    if compare:
        baseline,baseline_rows=execute(False)
    result=dict(task=('Synthetic fixed-camera IBVS respiratory compensation — geometric beam aiming only' if vision_mode else 'Synthetic externally observed respiratory compensation — geometric beam aiming only'),
                compensated=compensated,static_baseline=baseline,
                comparison_rms_improvement_percent=(
                    100*(baseline['beam_miss_rms_mm']-compensated['beam_miss_rms_mm'])/
                    baseline['beam_miss_rms_mm']
                    if baseline and baseline['successful'] and compensated['successful']
                    and baseline['beam_miss_rms_mm'] else None),
                calibration_assumptions=dict(period_s=period_s,
                    chest_to_tumour_correlation='assumed fixed and known',
                    marker_noise_std_mm=sensor_noise_mm,
                    internal_tumour_used_by_observer=False),
                limitations=['No real camera or ECG: camera pixels are synthetic pinhole projections with added noise',
                             'Eye-to-hand IBVS is a 2-D feature loop; 3-D depth relies on a surrogate model and IK',
                             'Synthetic internal lesion motion and fixed surrogate calibration',
                             'External marker does not directly reveal internal tumour motion',
                             'No radiation dose or clinical treatment planning',
                             'Proxy collision checks only; no exact patient mesh or self collision verification'])
    (output/'lung_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if comp_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4 if vision_mode else 3,1,figsize=(11,11 if vision_mode else 9),sharex=False,constrained_layout=True)
        for rows,label in ((comp_rows,'Compensation'),(baseline_rows,'Static beam')):
            xs=[r['time_s'] for r in rows if r['phase'] in ('track','static_baseline')]
            ys=[r['true_beam_miss_mm'] for r in rows if r['phase'] in ('track','static_baseline')]
            if xs:ax[0].plot(np.array(xs)-xs[0],ys,label=label)
        ax[0].set_ylabel('Actual beam-ray miss (mm)');ax[0].legend()
        active=[r for r in comp_rows if r['phase']=='track']
        if active:
            xx=np.array([r['time_s'] for r in active]);xx-=xx[0]
            ax[1].plot(xx,[r['breathing_measurement_mm'] for r in active],label='External chest marker')
            ax[1].set_ylabel('Chest motion (mm)');ax[1].legend()
            ax[2].plot(xx,[r['estimation_error_mm'] for r in active],label='Tumour estimate error')
            ax[2].set_ylabel('Estimation error (mm)');ax[2].set_xlabel('Tracking time (s)');ax[2].legend()
            if vision_mode:
                ax[3].plot(xx,[r['image_error_px'] for r in active],
                           label='Projected beam–estimated target image error')
                ax[3].set_ylabel('Image error (px)')
                ax[3].set_xlabel('Tracking time (s)')
                ax[3].legend()
        for a in ax:a.grid(alpha=.3)
        fig.suptitle('Synthetic '+('eye-to-hand IBVS' if vision_mode else 'chest-marker observer')+' and beam-axis tracking')
        fig.savefig(output/'lung_tracking_results.png',dpi=160)
        plt.close(fig)
    print(json.dumps(result,indent=2))
    print('Lung tracking results written to:',output)
    return result
