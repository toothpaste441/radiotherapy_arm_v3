"""Time-parameterized joint-space quintic trajectories using NumPy only."""
from dataclasses import dataclass
import numpy as np


@dataclass
class QuinticTrajectory:
    waypoints: np.ndarray
    knots: np.ndarray
    max_joint_speed: float

    @property
    def duration(self):
        return float(self.knots[-1])

    def evaluate(self, t):
        """Return joint position, velocity, and acceleration at t seconds.

        Each segment stops at its waypoints (zero velocity/acceleration). This
        produces globally C2-continuous motion but can be slower than blended
        interpolation. Speeds are bounded by max_joint_speed.
        """
        t = float(np.clip(t, 0.0, self.duration))
        i = min(int(np.searchsorted(self.knots, t, side="right") - 1), len(self.knots)-2)
        dt = self.knots[i+1] - self.knots[i]
        s = np.clip((t-self.knots[i]) / dt, 0.0, 1.0)
        h = 10*s**3 - 15*s**4 + 6*s**5
        hd = (30*s**2 - 60*s**3 + 30*s**4)/dt
        hdd = (60*s - 180*s**2 + 120*s**3)/dt**2
        d = self.waypoints[i+1] - self.waypoints[i]
        return self.waypoints[i] + h*d, hd*d, hdd*d


def make_quintic_trajectory(waypoints, max_joint_speed=0.5, min_segment_time=0.6):
    """Set segment times so max |dq/dt| <= max_joint_speed in rad/s.

    Peak derivative of the normalized quintic step is 1.875.
    """
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 6 or len(pts)<2 or not np.all(np.isfinite(pts)):
        raise ValueError("Waypoints must be a finite (N,6) array, N>=2")
    if max_joint_speed <= 0 or min_segment_time <= 0:
        raise ValueError("Speed and minimum segment time must be positive")
    d = np.max(np.abs(np.diff(pts, axis=0)), axis=1)
    durations = np.maximum(min_segment_time, 1.875*d/max_joint_speed)
    knots = np.r_[0.0, np.cumsum(durations)]
    return QuinticTrajectory(pts, knots, max_joint_speed)
