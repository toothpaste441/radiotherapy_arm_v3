"""Student-implemented position-only inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.kinematics import RobotKinematics


@dataclass(frozen=True)
class IKResult:
    joint_angles: np.ndarray
    position_error: float
    iterations: int
    converged: bool


def solve_position_ik(
    kinematics: RobotKinematics,
    target: np.ndarray,
    initial_angles: np.ndarray,
    method: str = "dls",
    damping: float = 0.04,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    max_step: float = 0.15,
) -> IKResult:
    """Solve a 3D tool-position task with DLS or a Moore-Penrose inverse."""
    if method not in {"dls", "pinv"}:
        raise ValueError("method must be 'dls' or 'pinv'")
    if damping <= 0 or tolerance <= 0 or max_iterations < 1 or max_step <= 0:
        raise ValueError("IK tolerances, damping, iterations, and step must be positive")

    desired = np.asarray(target, dtype=float)
    angles = np.asarray(initial_angles, dtype=float).copy()
    if desired.shape != (3,) or not np.all(np.isfinite(desired)):
        raise ValueError("target must be a finite vector with shape (3,)")
    if angles.shape != (6,) or not np.all(np.isfinite(angles)):
        raise ValueError("initial_angles must be a finite vector with shape (6,)")

    limits = kinematics.joint_limits
    angles = np.clip(angles, limits[:, 0], limits[:, 1])
    for iteration in range(max_iterations + 1):
        error = desired - kinematics.forward(angles)
        error_norm = float(np.linalg.norm(error))
        if error_norm <= tolerance:
            return IKResult(angles, error_norm, iteration, True)
        if iteration == max_iterations:
            break

        jacobian = kinematics.jacobian(angles)
        if method == "dls":
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
        else:
            update = np.linalg.pinv(jacobian, rcond=1e-5) @ error

        update_norm = float(np.linalg.norm(update))
        if update_norm > max_step:
            update *= max_step / update_norm

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = np.clip(angles + scale * update, limits[:, 0], limits[:, 1])
            candidate_error = np.linalg.norm(desired - kinematics.forward(candidate))
            if candidate_error < error_norm:
                angles = candidate
                accepted = True
                break
        if not accepted:
            break

    final_error = float(np.linalg.norm(desired - kinematics.forward(angles)))
    return IKResult(angles, final_error, iteration, False)
