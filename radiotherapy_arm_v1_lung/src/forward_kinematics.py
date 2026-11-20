"""Forward kinematics using the standard DH convention for the CyberKnife arm."""

from collections.abc import Mapping

import numpy as np


_LENGTH_NAMES = ("l1", "l2", "w2", "w3", "l3", "l4", "l5", "l6")


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Build A_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    cosine_theta = np.cos(theta)
    sine_theta = np.sin(theta)
    cosine_alpha = np.cos(alpha)
    sine_alpha = np.sin(alpha)

    rotation_z = np.array(
        [
            [cosine_theta, -sine_theta, 0.0, 0.0],
            [sine_theta, cosine_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    translation_z = np.eye(4)
    translation_z[2, 3] = d

    translation_x = np.eye(4)
    translation_x[0, 3] = a

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine_alpha, -sine_alpha, 0.0],
            [0.0, sine_alpha, cosine_alpha, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return rotation_z @ translation_z @ translation_x @ rotation_x


def forward_kinematics(
    joint_angles: np.ndarray,
    link_dimensions: Mapping[str, float],
) -> np.ndarray:
    """Return the base-to-end-effector transform from the supplied DH table.

    Args:
        joint_angles: Joint values [theta_1, ..., theta_6] in radians.
        link_dimensions: Finite link lengths keyed by l1, l2, w2, w3, l3,
            l4, l5, and l6. Use consistent length units (for example, metres).

    Returns:
        The 4x4 homogeneous transform T_0_6. Its translation uses the same
        length units as link_dimensions.
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (6,):
        raise ValueError(f"joint_angles must have shape (6,), got {joint_angles.shape}")
    if not np.all(np.isfinite(joint_angles)):
        raise ValueError("joint_angles must contain only finite values")

    try:
        dimensions = np.array(
            [link_dimensions[name] for name in _LENGTH_NAMES],
            dtype=float,
        )
    except KeyError as error:
        raise ValueError(
            f"link_dimensions is missing required value {error.args[0]!r}"
        ) from error
    except (TypeError, ValueError) as error:
        raise ValueError("link_dimensions values must be numeric") from error

    if not np.all(np.isfinite(dimensions)):
        raise ValueError("link_dimensions values must all be finite")

    l1, l2, w2, w3, l3, l4, l5, l6 = dimensions
    dh_table = (
        (0.0, np.pi / 2.0, l1, joint_angles[0]),
        (l2, -np.pi, w2, joint_angles[1]),
        (0.0, -np.pi / 2.0, w3, joint_angles[2]),
        (0.0, -np.pi / 2.0, l3 + l4, joint_angles[3]),
        (0.0, np.pi / 2.0, 0.0, joint_angles[4]),
        (0.0, 0.0, l5 + l6, joint_angles[5]),
    )

    transform = np.eye(4)
    for a, alpha, d, theta in dh_table:
        transform = transform @ _dh_transform(a, alpha, d, theta)

    return transform
