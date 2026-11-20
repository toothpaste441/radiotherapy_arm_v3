"""Synthetic chest-marker respiratory observation and correlated tumour estimation.

No camera or medical image is processed. A noisy, simulated external chest-marker
height is the ONLY measurement seen by this estimator. The tumour position used
for truth evaluation is never passed to it. The frequency and internal motion
relationship are assumed calibrated for this educational demonstration.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass(frozen=True)
class LungMotionCalibration:
    chest_amplitude: tuple[float, float, float] = (0.006, 0.0, 0.010)
    lung_amplitude: tuple[float, float, float] = (0.004, 0.002, 0.007)
    lung_phase: float = 0.5
    chest_rotation_amplitude: float = 0.012
    period_s: float = 4.0
    marker_amplitude_m: float = 0.012


def phase_displacement(cos_phase, sin_phase, calibration: LungMotionCalibration):
    """Compute world displacement from estimated sinusoidal breathing phase."""
    c, s = float(cos_phase), float(sin_phase)
    u = (1.0 - c) * 0.5
    v = (1.0 - (c*np.cos(2*calibration.lung_phase)
                 + s*np.sin(2*calibration.lung_phase))) * 0.5
    chest = np.asarray(calibration.chest_amplitude, float) * u
    lung = np.asarray(calibration.lung_amplitude, float) * v
    angle = calibration.chest_rotation_amplitude * s
    ca, sa = np.cos(angle), np.sin(angle)
    rotate = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    return chest + rotate @ lung


def synthetic_tumour(t, anchor, calibration: LungMotionCalibration):
    """Private synthetic ground truth used ONLY by the simulation/evaluation."""
    angle = 2.0*np.pi*float(t)/calibration.period_s
    baseline = phase_displacement(1.0, 0.0, calibration)
    return np.asarray(anchor, float) + phase_displacement(
        np.cos(angle), np.sin(angle), calibration) - baseline


def external_marker_height(t, calibration: LungMotionCalibration):
    """Measured external chest excursion, not internal tumour position."""
    return float(calibration.marker_amplitude_m *
                 (1.0 - np.cos(2*np.pi*float(t)/calibration.period_s))/2.0)


class ChestMarkerObserver:
    """Least-squares harmonic breathing-phase estimator.

    Fit z(t) = offset + b*cos(omega*t) + c*sin(omega*t), testing
    candidate breathing periods. Fit uses only past measured chest positions.
    The phase is recovered from the fitted coefficients; the calibrated
    external-to-internal motion model then predicts tumour displacement.
    """
    def __init__(self, anchor, calibration: LungMotionCalibration,
                 *, period_candidates=None, min_observation_s=3.0,
                 history_s=7.0, fit_every_s=0.20):
        self.anchor = np.asarray(anchor, float).copy()
        if self.anchor.shape != (3,):
            raise ValueError('anchor must be a 3-vector')
        self.calibration = calibration
        self.periods = np.asarray(period_candidates if period_candidates is not None
                                  else np.linspace(3.2, 4.8, 17), float)
        if np.any(self.periods <= 0) or min_observation_s <= 0:
            raise ValueError('invalid period or observation duration')
        self.min_observation_s = min_observation_s
        self.history_s = history_s
        self.fit_every_s = fit_every_s
        self.samples = deque()
        self.last_fit = -np.inf
        self.period_estimate = None
        self.coefficients = None
        self.residual_m = None

    @property
    def ready(self):
        return self.coefficients is not None

    def update(self, t, observed_marker_height):
        t, y = float(t), float(observed_marker_height)
        if not np.isfinite(t+y) or (self.samples and t <= self.samples[-1][0]):
            raise ValueError('finite, strictly increasing observation timestamps required')
        self.samples.append((t,y))
        while self.samples and t - self.samples[0][0] > self.history_s:
            self.samples.popleft()
        if (len(self.samples) < 30 or
            self.samples[-1][0] - self.samples[0][0] < self.min_observation_s or
            t - self.last_fit < self.fit_every_s):
            return self.ready
        times = np.array([p[0] for p in self.samples])
        heights = np.array([p[1] for p in self.samples])
        best = None
        for T in self.periods:
            w = 2*np.pi/T
            design = np.column_stack([np.ones(len(times)),
                                      np.cos(w*times), np.sin(w*times)])
            coef, _, _, _ = np.linalg.lstsq(design,heights,rcond=None)
            mse = np.mean((design@coef-heights)**2)
            if best is None or mse < best[0]:
                best = (mse, T, coef)
        assert best is not None
        mse,T,coef = best
        self.period_estimate = float(T)
        self.coefficients = coef
        self.residual_m = float(np.sqrt(mse))
        self.last_fit = t
        return True

    def estimate(self, t, *, prediction_horizon_s=0.0):
        if not self.ready:
            return self.anchor.copy()
        _,b,c = self.coefficients
        amplitude = np.hypot(b,c)
        if amplitude < 1e-8:
            return self.anchor.copy()
        theta = 2*np.pi*(float(t)+prediction_horizon_s)/self.period_estimate
        cos_theta = -(b*np.cos(theta)+c*np.sin(theta))/amplitude
        sin_theta = (c*np.cos(theta)-b*np.sin(theta))/amplitude
        norm = np.hypot(cos_theta,sin_theta)
        cos_theta /= norm
        sin_theta /= norm
        baseline = phase_displacement(1.0,0.0,self.calibration)
        return self.anchor + phase_displacement(cos_theta,sin_theta,self.calibration)-baseline
