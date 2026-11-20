"""Headless and interactive tumor-motion tracking experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src.inverse_kinematics import solve_position_ik
from src.kinematics import RobotKinematics
from src.tumor_trajectory import (
    BreathingParameters,
    tumor_displacement,
    tumor_position,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    method: str
    initial_offset: tuple[float, ...] = (0.0,) * 6
    amplitude_scale: float = 1.0
    period: float = 4.0
    disturbance_torque: float = 0.0
    gain_scale: float = 1.0


EXPERIMENTS = {
    "nominal": Experiment("nominal_dls", "dls", gain_scale=1.5),
    "challenge": Experiment(
        "challenging_dls",
        "dls",
        initial_offset=(0.30, -0.20, 0.25, 0.10, -0.15, 0.15),
        amplitude_scale=1.5,
        period=3.2,
        disturbance_torque=0.8,
        gain_scale=1.5,
    ),
    "pseudoinverse": Experiment("nominal_pseudoinverse", "pinv", gain_scale=1.5),
    "low-gain": Experiment("nominal_low_gain", "dls", gain_scale=0.5),
    "mid-gain": Experiment("nominal_mid_gain", "dls", gain_scale=1.0),
}

HOME_ANGLES = np.array([0.10, -0.15, 0.20, 0.0, 0.10, 0.0])
KP = np.array([120.0, 120.0, 100.0, 35.0, 30.0, 20.0])
KD = np.array([25.0, 25.0, 20.0, 8.0, 8.0, 6.0])
TORQUE_LIMITS = np.array([80.0, 80.0, 60.0, 25.0, 20.0, 15.0])
CONTROL_PERIOD = 0.02
DURATION = 8.0


def _named_ids(model, object_type, names: tuple[str, ...]) -> np.ndarray:
    ids = np.array(
        [mujoco.mj_name2id(model, object_type, name) for name in names], dtype=int
    )
    if np.any(ids < 0):
        raise ValueError(f"Model is missing one of {names}")
    return ids


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results file: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    xml_path: Path,
    output_dir: Path,
    experiment: Experiment,
    viewer: bool = False,
) -> dict:
    """Run one trajectory-tracking experiment and save sample-level results."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    kinematics = RobotKinematics(model)
    joint_qpos = kinematics.qpos_addresses
    joint_dofs = kinematics.dof_addresses
    actuator_ids = _named_ids(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, tuple(f"motor{i}" for i in range(1, 7))
    )
    mocap_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "tumor_target"
    )
    if mocap_body < 0 or model.body_mocapid[mocap_body] < 0:
        raise ValueError("The MuJoCo model needs a mocap body named 'tumor_target'")
    mocap_id = int(model.body_mocapid[mocap_body])

    physics_period = float(model.opt.timestep)
    substeps = round(CONTROL_PERIOD / physics_period)
    if substeps < 1 or not np.isclose(substeps * physics_period, CONTROL_PERIOD):
        raise ValueError("CONTROL_PERIOD must be an integer multiple of MuJoCo timestep")
    sample_count = round(DURATION / CONTROL_PERIOD)
    if not np.isclose(sample_count * CONTROL_PERIOD, DURATION):
        raise ValueError("DURATION must be an integer multiple of CONTROL_PERIOD")
    parameters = BreathingParameters(
        period=experiment.period, amplitude_scale=experiment.amplitude_scale
    )

    q_start = np.clip(
        HOME_ANGLES + np.asarray(experiment.initial_offset, dtype=float),
        kinematics.joint_limits[:, 0],
        kinematics.joint_limits[:, 1],
    )
    data.qpos[joint_qpos] = q_start
    mujoco.mj_forward(model, data)
    home_position = kinematics.forward(HOME_ANGLES)
    resting_position = home_position - tumor_displacement(0.0, parameters)
    data.mocap_pos[mocap_id] = tumor_position(0.0, resting_position, parameters)
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    viewer_handle = None
    if viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_handle = mujoco_viewer.launch_passive(model, data)

    rows = []
    condition_numbers = []
    ik_times = []
    command = q_start.copy()
    torque_squares = 0.0
    try:
        for sample in range(sample_count):
            time_seconds = sample * CONTROL_PERIOD
            target = tumor_position(time_seconds, resting_position, parameters)
            data.mocap_pos[mocap_id] = target
            ik_started = time.perf_counter()
            ik_result = solve_position_ik(
                kinematics,
                target,
                command,
                method=experiment.method,
                damping=0.04,
            )
            ik_times.append(time.perf_counter() - ik_started)
            if not ik_result.converged:
                raise RuntimeError(
                    f"{experiment.name}: {experiment.method} IK failed at "
                    f"t={time_seconds:.2f}s (error={ik_result.position_error:.6f}m)"
                )
            command = ik_result.joint_angles
            condition_numbers.append(float(np.linalg.cond(kinematics.jacobian(command))))

            for substep in range(substeps):
                current_time = time_seconds + (substep + 1) * physics_period
                data.mocap_pos[mocap_id] = tumor_position(
                    current_time, resting_position, parameters
                )
                q = data.qpos[joint_qpos].copy()
                qvel = data.qvel[joint_dofs].copy()
                bias = data.qfrc_bias[joint_dofs].copy()
                disturbance = np.zeros(6)
                disturbance[2] = (
                    experiment.disturbance_torque
                    * np.sin(2 * np.pi * current_time / 1.3)
                )
                data.qfrc_applied[joint_dofs] = disturbance
                torque = bias + (
                    experiment.gain_scale * KP * (command - q)
                    - experiment.gain_scale * KD * qvel
                )
                torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
                data.ctrl[actuator_ids] = torque
                mujoco.mj_step(model, data)
                torque_squares += float(torque @ torque) * physics_period
                if viewer_handle is not None:
                    viewer_handle.sync()
                    time.sleep(physics_period)

            actual = data.geom_xpos[kinematics.tool_geom_id].copy()
            actual_joints = data.qpos[joint_qpos].copy()
            sample_time = time_seconds + CONTROL_PERIOD
            target = tumor_position(sample_time, resting_position, parameters)
            position_error = float(np.linalg.norm(actual - target))
            rows.append(
                {
                    "time_s": sample_time,
                    "target_x_m": target[0],
                    "target_y_m": target[1],
                    "target_z_m": target[2],
                    "tool_x_m": actual[0],
                    "tool_y_m": actual[1],
                    "tool_z_m": actual[2],
                    "position_error_m": position_error,
                    "ik_iterations": ik_result.iterations,
                    "ik_error_m": ik_result.position_error,
                    **{
                        f"q{i + 1}_rad": actual_joints[i]
                        for i in range(6)
                    },
                    **{
                        f"q{i + 1}_desired_rad": command[i]
                        for i in range(6)
                    },
                    **{f"tau{i + 1}_Nm": data.ctrl[actuator_ids[i]] for i in range(6)},
                }
            )
    finally:
        if viewer_handle is not None:
            viewer_handle.close()

    position_errors = np.array([row["position_error_m"] for row in rows])
    settled_errors = np.array(
        [
            row["position_error_m"]
            for row in rows
            if row["time_s"] >= 1.0
        ]
    )
    summary = {
        "experiment": experiment.name,
        "ik_method": experiment.method,
        "duration_s": DURATION,
        "control_period_s": CONTROL_PERIOD,
        "breathing_period_s": experiment.period,
        "amplitude_scale": experiment.amplitude_scale,
        "initial_joint_offset_rad": list(experiment.initial_offset),
        "disturbance_torque_amplitude_Nm": experiment.disturbance_torque,
        "gain_scale": experiment.gain_scale,
        "samples": len(rows),
        "rms_tracking_error_mm": float(np.sqrt(np.mean(position_errors**2)) * 1000),
        "post_1s_rms_tracking_error_mm": float(
            np.sqrt(np.mean(settled_errors**2)) * 1000
        ),
        "p95_tracking_error_mm": float(np.percentile(position_errors, 95) * 1000),
        "max_tracking_error_mm": float(np.max(position_errors) * 1000),
        "final_tracking_error_mm": float(position_errors[-1] * 1000),
        "mean_ik_solve_ms": float(np.mean(ik_times) * 1000),
        "mean_jacobian_condition_number": float(np.mean(condition_numbers)),
        "torque_squared_integral_N2m2s": torque_squares,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / f"{experiment.name}.csv", rows)
    return summary


def _write_summary(output_dir: Path, summaries: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


def _write_plot(output_dir: Path, summaries: list[dict]) -> None:
    figure = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    paths_axis = figure.add_subplot(1, 2, 1, projection="3d")
    error_axis = figure.add_subplot(1, 2, 2)
    colors = plt.get_cmap("tab10").colors
    for summary, color in zip(summaries, colors):
        with (output_dir / f"{summary['experiment']}.csv").open(
            "r", newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        time_values = np.array([float(row["time_s"]) for row in rows])
        target = np.array(
            [
                [float(row[f"target_{axis}_m"]) for axis in "xyz"]
                for row in rows
            ]
        )
        tool = np.array(
            [[float(row[f"tool_{axis}_m"]) for axis in "xyz"] for row in rows]
        )
        if summary["experiment"] in {"nominal_dls", "challenging_dls"}:
            paths_axis.plot(
                *target.T,
                "--",
                color=color,
                alpha=0.8,
                label=f"{summary['experiment']} target",
            )
        paths_axis.plot(*tool.T, color=color, label=f"{summary['experiment']} tool")
        error_axis.plot(
            time_values,
            [float(row["position_error_m"]) * 1000 for row in rows],
            color=color,
            label=summary["experiment"],
        )

    paths_axis.set(
        title="3D tumor and tool paths",
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
    )
    paths_axis.legend(fontsize=6)
    error_axis.set(
        title="Cartesian tracking error",
        xlabel="Time (s)",
        ylabel="Position error (mm)",
    )
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()
    figure.suptitle("CyberKnife breathing-target tracking (simulation only)")
    figure.savefig(output_dir / "tracking_results.png", dpi=180)
    plt.close(figure)


def run_experiments(
    project_root: Path,
    output_dir: Path,
    experiment_names: tuple[str, ...] = (
        "nominal",
        "challenge",
        "pseudoinverse",
        "low-gain",
        "mid-gain",
    ),
) -> list[dict]:
    xml_path = project_root / "cyberknife_6dof.xml"
    summaries = [
        run_experiment(xml_path, output_dir, EXPERIMENTS[name])
        for name in experiment_names
    ]
    _write_summary(output_dir, summaries)
    _write_plot(output_dir, summaries)
    return summaries
