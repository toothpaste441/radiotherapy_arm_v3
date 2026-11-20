from pathlib import Path

import numpy as np
import pytest
mujoco = pytest.importorskip("mujoco", reason="MuJoCo integration tests require installed MuJoCo")

from src.inverse_kinematics import solve_position_ik
from src.kinematics import RobotKinematics
from src.tumor_trajectory import (
    BreathingParameters,
    circular_target_position,
    tumor_displacement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _robot():
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "cyberknife_6dof.xml"))
    return model, RobotKinematics(model)


def test_student_forward_kinematics_matches_mujoco_tool_point():
    model, kinematics = _robot()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)
    for angles in (np.zeros(6), rng.uniform(-0.4, 0.4, size=6)):
        data.qpos[kinematics.qpos_addresses] = angles
        mujoco.mj_forward(model, data)
        expected = data.geom_xpos[kinematics.tool_geom_id]
        np.testing.assert_allclose(kinematics.forward(angles), expected, atol=1e-10)


def test_geometric_jacobian_matches_finite_difference():
    _, kinematics = _robot()
    angles = np.array([0.1, -0.2, 0.25, 0.05, 0.15, -0.1])
    numerical = np.empty((3, 6))
    step = 1e-6
    for joint in range(6):
        offset = np.zeros(6)
        offset[joint] = step
        numerical[:, joint] = (
            kinematics.forward(angles + offset)
            - kinematics.forward(angles - offset)
        ) / (2 * step)
    np.testing.assert_allclose(kinematics.jacobian(angles), numerical, atol=1e-7)


def test_damped_least_squares_reaches_a_nearby_reachable_position():
    _, kinematics = _robot()
    initial = np.array([0.1, -0.15, 0.2, 0.0, 0.1, 0.0])
    target_angles = initial + np.array([0.025, -0.02, 0.02, 0.01, -0.015, 0.02])
    result = solve_position_ik(
        kinematics, kinematics.forward(target_angles), initial, method="dls"
    )
    assert result.converged
    assert result.position_error < 1e-5


def test_breathing_displacement_repeats_after_one_period():
    parameters = BreathingParameters(period=3.7, exponent=2)
    np.testing.assert_allclose(
        tumor_displacement(0.35, parameters),
        tumor_displacement(0.35 + parameters.period, parameters),
        atol=1e-12,
    )


def test_circular_demo_path_starts_at_resting_point_and_repeats():
    resting = np.array([0.8, -2.0, 1.0])
    period = 5.5
    np.testing.assert_allclose(
        circular_target_position(0.0, resting, period=period), resting, atol=1e-12
    )
    np.testing.assert_allclose(
        circular_target_position(0.7, resting, period=period),
        circular_target_position(0.7 + period, resting, period=period),
        atol=1e-12,
    )
