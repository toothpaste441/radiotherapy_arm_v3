"""Breathing-driven target motion based on the supplied trajectory notes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BreathingParameters:
    period: float = 4.0
    exponent: int = 1
    chest_amplitude: tuple[float, float, float] = (0.012, 0.0, 0.018)
    lung_amplitude: tuple[float, float, float] = (0.008, 0.004, 0.010)
    lung_phase: float = 0.5
    chest_rotation_amplitude: float = 0.02
    amplitude_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.period <= 0 or self.exponent < 1 or self.amplitude_scale <= 0:
            raise ValueError("period, exponent, and amplitude_scale must be positive")
        if len(self.chest_amplitude) != 3 or len(self.lung_amplitude) != 3:
            raise ValueError("chest_amplitude and lung_amplitude must be 3-vectors")


def _breathing_wave(time: float, period: float, exponent: int, phase: float = 0.0) -> float:
    return float(np.sin(np.pi * time / period - phase) ** (2 * exponent))


def tumor_displacement(time: float, parameters: BreathingParameters) -> np.ndarray:
    """Return tumor displacement from its resting chest-frame origin in metres."""
    chest_wave = _breathing_wave(time, parameters.period, parameters.exponent)
    lung_wave = _breathing_wave(
        time, parameters.period, parameters.exponent, parameters.lung_phase
    )
    chest = parameters.amplitude_scale * np.asarray(
        parameters.chest_amplitude, dtype=float
    ) * chest_wave
    lung = parameters.amplitude_scale * np.asarray(
        parameters.lung_amplitude, dtype=float
    ) * lung_wave

    theta_x = (
        parameters.chest_rotation_amplitude
        * parameters.amplitude_scale
        * np.sin(2 * np.pi * time / parameters.period)
    )
    cosine, sine = np.cos(theta_x), np.sin(theta_x)
    chest_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    return chest + chest_rotation @ lung


def tumor_position(
    time: float, resting_position: np.ndarray, parameters: BreathingParameters
) -> np.ndarray:
    """Return the target point in world coordinates."""
    origin = np.asarray(resting_position, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("resting_position must be a finite vector with shape (3,)")
    return origin + tumor_displacement(time, parameters)


def circular_target_position(
    time: float,
    resting_position: np.ndarray,
    radius: float = 0.025,
    period: float = 6.0,
) -> np.ndarray:
    """Return a small planar circular path that starts at resting_position."""
    origin = np.asarray(resting_position, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("resting_position must be a finite vector with shape (3,)")
    if not np.isfinite(time) or radius <= 0 or period <= 0:
        raise ValueError("time must be finite and radius and period must be positive")
    phase = 2 * np.pi * time / period
    displacement = radius * np.array([np.cos(phase) - 1.0, np.sin(phase), 0.0])
    return origin + displacement
