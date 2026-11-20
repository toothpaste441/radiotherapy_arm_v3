"""Runs an actual MuJoCo joint-servo settling check when MuJoCo is installed.

This complements the NumPy-only unit tests. In environments without the MuJoCo
wheel, it is explicitly SKIPPED, never reported as a successful physics run.
"""
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco', reason='MuJoCo integration test requires installed MuJoCo')

from src.kinematics import RobotKinematics
from src.mujoco_compat import fill_dense_mass_matrix
from src.stable_joint_control import ComputedTorquePD


ROOT = Path(__file__).resolve().parents[1]


def test_gravity_loaded_joint_settling_with_large_xml_damping():
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'cyberknife_6dof.xml'))
    data = mujoco.MjData(model)
    kin = RobotKinematics(model)
    qadr, vadr = kin.qpos_addresses, kin.dof_addresses
    from src.simulation import HOME_ANGLES, TORQUE_LIMITS
    home = HOME_ANGLES.copy()
    goal = home + np.array([.045, -.035, .03, .02, -.02, .012])
    data.qpos[qadr] = home
    mujoco.mj_forward(model, data)
    actuators = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i}')
        for i in range(1, 7)
    ])
    controller = ComputedTorquePD(TORQUE_LIMITS, natural_frequency=3., damping_ratio=1.)
    full_mass = np.zeros((model.nv, model.nv))
    for _ in range(int(5.0 / model.opt.timestep)):
        mujoco.mj_forward(model, data)
        q = data.qpos[qadr].copy()
        dq = data.qvel[vadr].copy()
        bias = data.qfrc_bias[vadr].copy()
        passive = data.qfrc_passive[vadr].copy()
        fill_dense_mass_matrix(mujoco, model, data, full_mass)
        data.ctrl[actuators] = controller.calculate(
            q, dq, goal, np.zeros(6), bias,
            full_mass[np.ix_(vadr, vadr)], passive=passive
        )
        mujoco.mj_step(model, data)
    final_error = np.max(np.abs(data.qpos[qadr] - goal))
    final_speed = np.max(np.abs(data.qvel[vadr]))
    assert final_error < 0.015, f'Joint servo failed: {final_error:.4g} rad'
    assert final_speed < 0.08, f'Joint servo did not settle: {final_speed:.4g} rad/s'
    assert controller.diag.max_abs_passive > 0.0


def test_full_viewpoint_wrist_goal_under_gravity():
    """Regression for the previously stalled wrist configuration.

    Test a full 6R joint command over 10 simulated seconds. This is a
    direct joint-servo test, not a proof of collision-safe tool movement.
    """
    from src.simulation import HOME_ANGLES, TORQUE_LIMITS
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'cyberknife_6dof.xml'))
    data = mujoco.MjData(model)
    kin = RobotKinematics(model)
    qadr, vadr = kin.qpos_addresses, kin.dof_addresses
    data.qpos[qadr] = HOME_ANGLES.copy()
    mujoco.mj_forward(model, data)
    goal = np.array([.55, -.53, .52, .30, -.12, .18])
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i}') for i in range(1, 7)])
    ctl = ComputedTorquePD(TORQUE_LIMITS, natural_frequency=3., damping_ratio=1.)
    M = np.zeros((model.nv, model.nv))
    for _ in range(round(10.0 / model.opt.timestep)):
        mujoco.mj_forward(model, data)
        q = data.qpos[qadr].copy()
        dq = data.qvel[vadr].copy()
        fill_dense_mass_matrix(mujoco, model, data, M)
        data.ctrl[ids] = ctl.calculate(q, dq, goal, np.zeros(6),
            data.qfrc_bias[vadr].copy(), M[np.ix_(vadr, vadr)],
            passive=data.qfrc_passive[vadr].copy())
        mujoco.mj_step(model, data)
    error = np.abs(data.qpos[qadr] - goal)
    speed = np.max(np.abs(data.qvel[vadr]))
    assert np.max(error) <= 0.015, f'Joint errors: {error}'
    assert speed <= 0.08, f'Final speed: {speed}'
    assert ctl.diag.saturated_samples == 0, 'Unexpected torque saturation'
