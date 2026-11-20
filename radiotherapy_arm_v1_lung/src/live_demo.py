"""Interactive MuJoCo demo with manual joints and selectable target paths."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from src.inverse_kinematics import solve_position_ik
from src.kinematics import RobotKinematics
from src.simulation import CONTROL_PERIOD, HOME_ANGLES, KD, KP, TORQUE_LIMITS
from src.tumor_trajectory import (
    BreathingParameters,
    circular_target_position,
    tumor_displacement,
    tumor_position,
)


def launch_live_demo(xml_path: Path) -> None:
    """Run until the user closes a window; provide Tk controls and joint sliders."""
    import tkinter as tk

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    kinematics = RobotKinematics(model)
    joint_qpos = kinematics.qpos_addresses
    joint_dofs = kinematics.dof_addresses
    actuator_ids = np.array(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{joint + 1}"
            )
            for joint in range(6)
        ],
        dtype=int,
    )
    if np.any(actuator_ids < 0):
        raise ValueError("MuJoCo model is missing one or more robot motor actuators")

    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "tumor_target"
    )
    if target_body < 0 or model.body_mocapid[target_body] < 0:
        raise ValueError("MuJoCo model needs a mocap body named 'tumor_target'")
    mocap_id = int(model.body_mocapid[target_body])

    data.qpos[joint_qpos] = HOME_ANGLES
    mujoco.mj_forward(model, data)
    home_position = kinematics.forward(HOME_ANGLES)
    parameters = BreathingParameters()
    initial_resting_position = home_position - tumor_displacement(0.0, parameters)
    data.mocap_pos[mocap_id] = home_position
    mujoco.mj_forward(model, data)

    from mujoco import viewer as mujoco_viewer

    viewer_handle = mujoco_viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=True
    )
    viewer_handle.cam.lookat[:] = home_position
    viewer_handle.cam.distance = 2.5

    root = tk.Tk()
    root.title("CyberKnife live controls")
    root.resizable(False, False)

    state = {
        "mode": "hold",
        "trajectory": "",
        "trajectory_start": 0.0,
        "trajectory_origin": initial_resting_position.copy(),
        "command": HOME_ANGLES.copy(),
        "slider_update": False,
        "last_status_update": 0.0,
        "closed": False,
        "ik_message": "",
    }

    status = tk.StringVar(value="Mode: hold at home")
    tk.Label(root, textvariable=status, anchor="w", padx=8, pady=8).pack(
        fill="x"
    )

    controls = tk.Frame(root, padx=8, pady=4)
    controls.pack(fill="x")
    sliders = []

    def set_sliders(angles: np.ndarray) -> None:
        state["slider_update"] = True
        try:
            for index, slider in enumerate(sliders):
                slider.set(np.rad2deg(angles[index]))
        finally:
            state["slider_update"] = False

    def set_manual_joint(index: int, value: str) -> None:
        if state["slider_update"]:
            return
        state["command"][index] = np.deg2rad(float(value))
        state["mode"] = "manual"
        state["ik_message"] = ""
        status.set(f"Mode: manual joint control — J{index + 1} updated")

    joint_frame = tk.LabelFrame(root, text="Joint targets (degrees)", padx=8, pady=4)
    joint_frame.pack(fill="x", padx=8, pady=4)
    for joint in range(6):
        lower, upper = np.rad2deg(kinematics.joint_limits[joint])
        slider = tk.Scale(
            joint_frame,
            from_=lower,
            to=upper,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=340,
            label=f"Joint {joint + 1}",
            command=lambda value, index=joint: set_manual_joint(index, value),
        )
        slider.pack(fill="x")
        sliders.append(slider)
    set_sliders(HOME_ANGLES)

    def current_tool_position() -> np.ndarray:
        return data.geom_xpos[kinematics.tool_geom_id].copy()

    def start_path(name: str) -> None:
        current_position = current_tool_position()
        state["mode"] = "trajectory"
        state["trajectory"] = name
        state["trajectory_start"] = float(data.time)
        if name == "breathing":
            state["trajectory_origin"] = (
                current_position - tumor_displacement(0.0, parameters)
            )
        else:
            state["trajectory_origin"] = current_position
        state["ik_message"] = ""
        status.set(f"Mode: playing {name} trajectory")

    def hold_position() -> None:
        state["command"] = data.qpos[joint_qpos].copy()
        state["mode"] = "hold"
        state["ik_message"] = ""
        set_sliders(state["command"])
        status.set("Mode: holding current joint positions")

    def go_home() -> None:
        state["command"] = HOME_ANGLES.copy()
        state["mode"] = "manual"
        state["ik_message"] = ""
        set_sliders(HOME_ANGLES)
        status.set("Mode: moving to home")

    buttons = tk.Frame(root, padx=8, pady=8)
    buttons.pack(fill="x")
    tk.Button(
        buttons,
        text="Play breathing",
        command=lambda: start_path("breathing"),
        width=16,
    ).grid(row=0, column=0, padx=3, pady=3)
    tk.Button(
        buttons,
        text="Play circle",
        command=lambda: start_path("circle"),
        width=16,
    ).grid(row=0, column=1, padx=3, pady=3)
    tk.Button(buttons, text="Hold", command=hold_position, width=16).grid(
        row=1, column=0, padx=3, pady=3
    )
    tk.Button(buttons, text="Go home", command=go_home, width=16).grid(
        row=1, column=1, padx=3, pady=3
    )

    physics_period = float(model.opt.timestep)
    substeps = round(CONTROL_PERIOD / physics_period)
    if substeps < 1 or not np.isclose(substeps * physics_period, CONTROL_PERIOD):
        viewer_handle.close()
        root.destroy()
        raise ValueError("Control period must be an integer multiple of MuJoCo timestep")

    def desired_target(simulation_time: float) -> np.ndarray:
        elapsed = simulation_time - state["trajectory_start"]
        if state["trajectory"] == "breathing":
            return tumor_position(
                elapsed, state["trajectory_origin"], parameters
            )
        return circular_target_position(
            elapsed, state["trajectory_origin"], radius=0.025, period=6.0
        )

    def simulation_tick() -> None:
        if state["closed"]:
            return
        if not viewer_handle.is_running():
            close_windows()
            return

        if state["mode"] == "trajectory":
            target = desired_target(float(data.time))
            result = solve_position_ik(
                kinematics,
                target,
                state["command"],
                method="dls",
                damping=0.04,
            )
            if result.converged:
                state["command"] = result.joint_angles
                state["ik_message"] = ""
            else:
                state["mode"] = "hold"
                state["command"] = data.qpos[joint_qpos].copy()
                state["ik_message"] = (
                    f"IK stopped: position error {result.position_error:.4f} m"
                )
                print(state["ik_message"])
                status.set(state["ik_message"])

        for _ in range(substeps):
            current_time = float(data.time)
            if state["mode"] == "trajectory":
                target = desired_target(current_time)
            else:
                target = current_tool_position()
            data.mocap_pos[mocap_id] = target

            q = data.qpos[joint_qpos].copy()
            qvel = data.qvel[joint_dofs].copy()
            bias = data.qfrc_bias[joint_dofs].copy()
            torque = bias + 1.5 * (
                KP * (state["command"] - q) - KD * qvel
            )
            data.ctrl[actuator_ids] = np.clip(
                torque, -TORQUE_LIMITS, TORQUE_LIMITS
            )
            data.qfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)

        viewer_handle.sync()
        wall_time = time.perf_counter()
        if wall_time - state["last_status_update"] >= 0.2:
            target = desired_target(float(data.time)) if state["mode"] == "trajectory" else current_tool_position()
            error = float(np.linalg.norm(current_tool_position() - target))
            if state["mode"] == "trajectory":
                status.set(
                    f"Mode: playing {state['trajectory']} | "
                    f"sim {data.time:.1f} s | error {error * 1000:.1f} mm"
                )
            elif state["mode"] == "manual":
                status.set("Mode: manual joint control")
            elif not state["ik_message"]:
                status.set("Mode: holding current joint positions")
            state["last_status_update"] = wall_time
        root.after(round(CONTROL_PERIOD * 1000), simulation_tick)

    def close_windows() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        if viewer_handle.is_running():
            viewer_handle.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_windows)
    root.after(0, simulation_tick)
    try:
        root.mainloop()
    finally:
        if viewer_handle.is_running():
            viewer_handle.close()
