"""Integrated home -> RRT-Connect -> quintic approach -> respiratory tracking.

Run from project root: python robot_simulation.py --plan
Optional viewing:       python robot_simulation.py --plan --plan-view
Output: results/planning/{approach_and_tracking.csv,planning_summary.json,planning.png}
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.kinematics import RobotKinematics
from src.inverse_kinematics import solve_position_ik
from src.path_planning import CollisionChecker, SphereObstacle, rrt_connect
from src.trajectory_generation import make_quintic_trajectory
from src.tumor_trajectory import BreathingParameters, tumor_displacement, tumor_position


def _choose_obstacle(kin, q0, q1):
    """Place a demonstration obstacle near the nominal tool route.

    The obstacle is an illustrative sphere, not patient geometry. If it cannot
    be placed without invalidating either endpoint, fail rather than pretend
    collision checking succeeded.
    """
    p0, p1 = kin.forward(q0), kin.forward(q1)
    sep = np.linalg.norm(p1-p0)
    if sep < 0.08:
        raise ValueError("Treatment joint configuration produced insufficient tool separation")
    center = (p0+p1)/2
    for radius in (0.15*sep, 0.10*sep, 0.06*sep):
        obstacle = SphereObstacle(center, float(radius), "virtual_patient_obstacle")
        checker = CollisionChecker(kin, [obstacle], link_radius=0.012, clearance=0.005)
        if checker.valid(q0) and checker.valid(q1):
            return obstacle, checker
    raise ValueError("Demonstration obstacle intersects home or treatment configuration")


def _make_goal(kin, home):
    # Example reachable joint-space treatment pose, within joint limits.
    proposed = home + np.array([0.45, -0.38, 0.32, 0.30, -0.22, 0.18])
    limits = kin.joint_limits
    return np.clip(proposed, limits[:,0]+0.03, limits[:,1]-0.03)


def run_planning_demo(project_root: Path, *, view=False, tracking_duration=8.0, seed=7):
    import mujoco
    import xml.etree.ElementTree as ET
    from src.simulation import HOME_ANGLES, KP, KD, TORQUE_LIMITS, CONTROL_PERIOD
    root = Path(project_root)
    model = mujoco.MjModel.from_xml_path(str(root / "cyberknife_6dof.xml"))
    data = mujoco.MjData(model)
    kin = RobotKinematics(model)
    output_dir = root / "results" / "planning"
    output_dir.mkdir(parents=True, exist_ok=True)
    q0 = HOME_ANGLES.copy()
    q_goal = _make_goal(kin, q0)
    obstacle, checker = _choose_obstacle(kin, q0, q_goal)
    plan = rrt_connect(q0, q_goal, checker, seed=seed)
    # Create a SEPARATE XML so the virtual obstacle is visible during planning
    # without changing the baseline breathing experiments. Collision detection
    # remains the NumPy capsule checker; the sphere is visual-only in MuJoCo.
    xml_tree = ET.parse(root / "cyberknife_6dof.xml")
    world = xml_tree.getroot().find("worldbody")
    ET.SubElement(world, "geom", name="virtual_patient_obstacle", type="sphere",
                  pos=" ".join(f"{a:.9f}" for a in obstacle.center),
                  size=f"{obstacle.radius:.9f}", rgba="0.9 0.55 0.12 0.6",
                  contype="0", conaffinity="0")
    scene_path = root / "planning_scene.xml"
    xml_tree.write(scene_path, encoding="unicode")
    if not plan.success:
        raise RuntimeError("RRT-Connect failed: increase iterations or revise obstacle/goal")
    trajectory = make_quintic_trajectory(plan.waypoints, max_joint_speed=0.5)
    # Same articulated robot with an additional visual marker. Reload only
    # after planning so the baseline model and experiment results are untouched.
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    kin = RobotKinematics(model)
    # Audit every interpolated trajectory state against the approximate obstacle.
    for t in np.linspace(0, trajectory.duration, max(50, int(trajectory.duration/0.02)+1)):
        if not checker.valid(trajectory.evaluate(t)[0]):
            raise RuntimeError("Smoothed quintic trajectory violates planning collision model")

    qidx, vidx = kin.qpos_addresses, kin.dof_addresses
    actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                          f"motor{i}") for i in range(1,7)])
    mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tumor_target")
    mocap_id = int(model.body_mocapid[mocap_body])
    data.qpos[qidx] = q0
    mujoco.mj_forward(model, data)
    params = BreathingParameters()
    target_origin = kin.forward(q_goal)-tumor_displacement(0.0, params)
    data.mocap_pos[mocap_id] = kin.forward(q_goal)
    mujoco.mj_forward(model, data)

    viewer = None
    if view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        viewer.cam.lookat[:] = kin.forward(q0)
        viewer.cam.distance = 3.0

    rows=[]
    physics_dt = float(model.opt.timestep)
    steps = round(CONTROL_PERIOD/physics_dt)
    if steps < 1 or not np.isclose(steps*physics_dt, CONTROL_PERIOD):
        raise ValueError("Physics timestep must divide 20ms control period")
    total_steps = int(np.ceil((trajectory.duration+tracking_duration)/CONTROL_PERIOD))
    command = q_goal.copy()
    try:
        for sample in range(total_steps):
            t = sample*CONTROL_PERIOD
            phase = "approach" if t < trajectory.duration else "tracking"
            if phase == "approach":
                q_des, qd_des, _ = trajectory.evaluate(t)
                target = kin.forward(q_goal)
            else:
                breathing_t = t-trajectory.duration
                target = tumor_position(breathing_t, target_origin, params)
                result = solve_position_ik(kin, target, command, damping=0.04)
                if not result.converged:
                    raise RuntimeError(f"Tracking IK did not converge at t={t:.2f}s: {result.position_error:.5f}m")
                command = result.joint_angles
                q_des, qd_des = command, np.zeros(6)
            data.mocap_pos[mocap_id] = target
            for _ in range(steps):
                q=data.qpos[qidx].copy()
                qd=data.qvel[vidx].copy()
                bias=data.qfrc_bias[vidx].copy()
                torque=bias+1.5*(KP*(q_des-q)+KD*(qd_des-qd))
                data.ctrl[actuators]=np.clip(torque,-TORQUE_LIMITS,TORQUE_LIMITS)
                mujoco.mj_step(model,data)
            actual = data.geom_xpos[kin.tool_geom_id].copy()
            actual_q=data.qpos[qidx].copy()
            reference = kin.forward(q_des) if phase == "approach" else target
            rows.append({"time_s":float(data.time), "phase":phase,
                         "tracking_error_mm":1000*float(np.linalg.norm(actual-reference)),
                         **{f"q{i+1}_rad":float(actual_q[i]) for i in range(6)},
                         **{f"q{i+1}_desired_rad":float(q_des[i]) for i in range(6)},
                         **{f"torque{i+1}_Nm":float(data.ctrl[actuators[i]]) for i in range(6)}})
            if viewer is not None:
                viewer.sync()
                time.sleep(CONTROL_PERIOD)
    finally:
        if viewer is not None:
            viewer.close()

    with (output_dir / "approach_and_tracking.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    approach = np.asarray([r["tracking_error_mm"] for r in rows if r["phase"]=="approach"])
    tracking = np.asarray([r["tracking_error_mm"] for r in rows if r["phase"]=="tracking"])
    summary={
        "planning_algorithm":"RRT-Connect (NumPy)",
        "collision_model":"joint limits + 3D sphere versus approximate link capsules; no self-collision/STL checks",
        "obstacle_center_m":obstacle.center.tolist(),
        "obstacle_radius_m":obstacle.radius,
        "start_joint_rad":q0.tolist(), "goal_joint_rad":q_goal.tolist(),
        "planning_time_s":plan.elapsed_s, "planning_iterations":plan.iterations,
        "collision_checks":checker.checks,
        "raw_waypoint_count":len(plan.raw_waypoints), "waypoint_count":len(plan.waypoints),
        "path_length_joint_rad":float(np.sum(np.linalg.norm(np.diff(plan.waypoints,axis=0),axis=1))),
        "trajectory_duration_s":trajectory.duration,
        "approach_rms_error_mm":float(np.sqrt(np.mean(approach**2))),
        "tracking_rms_error_mm":float(np.sqrt(np.mean(tracking**2))),
        "tracking_max_error_mm":float(np.max(tracking)),
        "tracking_duration_s":tracking_duration,
    }
    (output_dir / "planning_summary.json").write_text(json.dumps(summary,indent=2))
    np.savetxt(output_dir / "planned_joint_waypoints.csv",plan.waypoints,delimiter=",",
               header=",".join(f"q{i+1}_rad" for i in range(6)),comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    tt=np.array([r["time_s"] for r in rows])
    axes[0].plot(tt,[r["tracking_error_mm"] for r in rows])
    axes[0].axvline(trajectory.duration,linestyle="--",label="approach → tracking")
    axes[0].set(ylabel="Tool reference error (mm)",title="Planned approach + breathing tracking")
    axes[0].legend()
    for i in range(6):
        axes[1].plot(tt,[r[f"q{i+1}_rad"] for r in rows],label=f"J{i+1}")
    axes[1].axvline(trajectory.duration,linestyle="--")
    axes[1].set(xlabel="Time (s)",ylabel="Joint position (rad)")
    axes[1].legend(ncol=6)
    fig.savefig(output_dir / "planning.png",dpi=150)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(f"Planning outputs saved to {output_dir}")
    return summary
