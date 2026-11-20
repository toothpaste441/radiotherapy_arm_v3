"""NumPy position + beam-axis geometric inverse kinematics.

A position-only IK solver cannot point a linac's beam at a fixed target. Here
we constrain three tool-position coordinates and two independent beam-direction
coordinates (represented as a three-vector residual). Tool roll is free.
This is a classroom geometry demo, NOT an implementation of clinical targeting.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BeamIKResult:
    joint_angles: np.ndarray
    position_error_m: float
    pointing_error_deg: float
    iterations: int
    converged: bool


def solve_beam_ik(kin, target_point, target_direction, initial_angles,
                  *, damping=0.035, orientation_weight_m=0.30,
                  position_tolerance_m=0.001, direction_tolerance_deg=0.30,
                  max_iterations=220, max_step_rad=0.12):
    """Damped least squares with a line search and joint limits.

    This uses the analytic derivative of the unit tool +X axis, not numerical
    differentiation or MuJoCo's Jacobian routine. Target direction is a unit
    ray direction from tool toward the synthetic target.
    """
    goal = np.asarray(target_point, dtype=float)
    desired_axis = np.asarray(target_direction, dtype=float)
    q = np.asarray(initial_angles, dtype=float).copy()
    if goal.shape != (3,) or desired_axis.shape != (3,) or q.shape != (6,):
        raise ValueError('Expected tool point and direction (3,) and joint angles (6,)')
    if not all(np.all(np.isfinite(v)) for v in (goal, desired_axis, q)):
        raise ValueError('Nonfinite input')
    axis_norm = np.linalg.norm(desired_axis)
    if axis_norm < 1e-9:
        raise ValueError('Zero target direction')
    desired_axis /= axis_norm
    if damping <= 0 or orientation_weight_m <= 0 or max_iterations < 1:
        raise ValueError('Invalid IK parameters')
    lower, upper = kin.joint_limits.T
    q = np.clip(q, lower, upper)

    def evaluate(q_trial):
        position, direction, Jv, Jd = kin.beam_state(q_trial)
        epos = goal-position
        edir = desired_axis-direction
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(desired_axis, direction), -1, 1))))
        residual = np.r_[epos, orientation_weight_m*edir]
        return np.linalg.norm(residual), np.linalg.norm(epos), angle, residual, Jv, Jd

    for i in range(max_iterations+1):
        cost, poserr, angerr, error, jv, jd = evaluate(q)
        if poserr <= position_tolerance_m and angerr <= direction_tolerance_deg:
            return BeamIKResult(q, float(poserr), angerr, i, True)
        if i == max_iterations:
            break
        j = np.vstack((jv, orientation_weight_m*jd))
        update = j.T @ np.linalg.solve(j@j.T + damping*damping*np.eye(6), error)
        n = np.linalg.norm(update)
        if n > max_step_rad:
            update *= max_step_rad/n
        accepted = False
        for scale in (1., .5, .25, .125, .0625):
            candidate = np.clip(q + scale*update, lower, upper)
            c = evaluate(candidate)[0]
            if c < cost-1e-12:
                q = candidate
                accepted = True
                break
        if not accepted:
            break
    _, poserr, angerr, _, _, _ = evaluate(q)
    return BeamIKResult(q, float(poserr), angerr, i, False)


def candidate_beam_directions(reference_direction):
    """Moderate non-coplanar viewpoint offsets (degrees), relative to baseline.

    These are arbitrary classroom demonstration viewpoints, not treatment nodes.
    """
    axis = np.asarray(reference_direction, float)
    axis /= np.linalg.norm(axis)
    # Build a right-handed transverse frame to perturb the source direction.
    up = np.array([0.,0.,1.])
    if abs(up @ axis) > 0.95:
        up = np.array([0.,1.,0.])
    u = np.cross(up, axis); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    offsets = [(0., 0.), (7., 0.), (0., 7.), (-7., 0.), (0., -7.),
               (10., 6.), (-10., 6.), (9., -6.), (-9., -6.),
               (5., 4.), (-5., 4.), (5., -4.), (-5., -4.)]
    result=[]
    for alpha,beta in offsets:
        a=np.deg2rad(alpha); b=np.deg2rad(beta)
        d=axis+np.tan(a)*u+np.tan(b)*v
        result.append((alpha,beta,d/np.linalg.norm(d)))
    return result
