"""Regression tests for the joint-damping failure seen on MuJoCo 4.0.

These test the controller mathematics with a simple, independently coded
second-order plant; they do NOT replace MuJoCo integration tests.
"""

import numpy as np
import pytest

from src.stable_joint_control import ComputedTorquePD


def test_computed_torque_subtracts_passive_and_preserves_saturation():
    M = np.diag([0.06, 0.1, 0.2, 0.5, 0.01, 0.02])
    q = np.zeros(6)
    vel = np.array([0.2, -0.1, 0, 0, 0, 0])
    q_goal = np.full(6, 0.3)
    v_goal = np.zeros(6)
    bias = np.array([3.0, 1.0, 0, 0, 0, 0])
    passive = np.array([-1.0, 0.5, 0, 0, 0, 0])
    controller = ComputedTorquePD([100] * 6, natural_frequency=3.0, damping_ratio=1.0)
    result = controller.calculate(q, vel, q_goal, v_goal, bias, M, passive=passive)
    desired_acc = 9.0 * (q_goal - q) + 6.0 * (v_goal - vel)
    np.testing.assert_allclose(result, bias - passive + M @ desired_acc)
    assert controller.diag.max_abs_passive == 1.0
    saturated = ComputedTorquePD([0.1] * 6)
    clipped = saturated.calculate(q, vel, q_goal, v_goal, bias, M, passive=passive)
    assert np.all(np.abs(clipped) <= 0.1 + 1e-12)
    assert saturated.diag.saturated_samples == 1


def test_passive_force_validation():
    controller = ComputedTorquePD([100] * 6)
    with pytest.raises(ValueError):
        controller.calculate(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6),
                             np.zeros(6), np.eye(6), passive=np.zeros(5))
    with pytest.raises(ValueError):
        controller.calculate(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6),
                             np.zeros(6), np.eye(6), passive=np.full(6, np.nan))


def test_high_damping_stall_is_removed_in_simplified_plant():
    """Reproduce a persistent error when damping overwhelms inertia feedback."""
    # Illustrative single-joint damping/mass values, not identified robot data.
    mass, damping, dt, target = 0.06, 5.0, 0.002, 0.45
    def run(compensate):
        q = vel = 0.0
        ctl = ComputedTorquePD([100.0] * 6)
        M = np.diag([mass] + [1.0] * 5)
        reference = np.r_[target, np.zeros(5)]
        for _ in range(int(7.0 / dt)):
            passive = -damping * vel
            torque = ctl.calculate(
                np.r_[q, np.zeros(5)], np.r_[vel, np.zeros(5)],
                reference, np.zeros(6), np.zeros(6), M,
                passive=np.r_[passive, np.zeros(5)] if compensate else None,
            )[0]
            accel = (torque + passive) / mass
            vel += dt * accel
            q += dt * vel
        return abs(target - q)
    assert run(False) > 0.10
    assert run(True) < 1e-5
