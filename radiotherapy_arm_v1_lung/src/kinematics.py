"""Forward position kinematics and Jacobian for the MuJoCo robot chain."""

from __future__ import annotations

import numpy as np


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a rotation matrix using Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine, sine = np.cos(angle), np.sin(angle)
    one_minus_cosine = 1 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


class RobotKinematics:
    """Evaluate the six-revolute-joint chain encoded in a MuJoCo model.

    MuJoCo supplies the robot's fixed link transforms, joint axes, and tool
    offset; all joint-dependent transforms and the geometric Jacobian are
    computed here with NumPy.
    """

    def __init__(
        self,
        model,
        tool_geom_name: str = "tool_tip",
        joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7)),
    ) -> None:
        import mujoco

        self.model = model
        self.joint_names = joint_names
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise ValueError(f"MuJoCo model is missing one of {joint_names}")
        if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in self.joint_ids):
            raise ValueError("This implementation requires revolute hinge joints")

        self.qpos_addresses = model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_addresses = model.jnt_dofadr[self.joint_ids].astype(int)
        self.joint_limits = model.jnt_range[self.joint_ids].copy()
        if not np.all(model.jnt_limited[self.joint_ids]):
            raise ValueError("All robot joints must have limits in the MuJoCo model")

        self.tool_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, tool_geom_name
        )
        if self.tool_geom_id < 0:
            raise ValueError(f"MuJoCo model is missing tool geom {tool_geom_name!r}")
        self.tool_body_id = int(model.geom_bodyid[self.tool_geom_id])

        chain = []
        body_id = self.tool_body_id
        while body_id > 0:
            chain.append(body_id)
            body_id = int(model.body_parentid[body_id])
        self.body_chain = tuple(reversed(chain))
        chain_joint_ids = [
            int(model.body_jntadr[body_id])
            for body_id in self.body_chain
            if model.body_jntnum[body_id]
        ]
        if chain_joint_ids != self.joint_ids.tolist():
            raise ValueError("The named joints must appear in order on the tool body chain")

        self.tool_offset = model.geom_pos[self.tool_geom_id].copy()
        # MuJoCo can assign a non-identity geom-local quaternion for imported
        # mesh geometry (including compiled mesh-frame corrections). The beam
        # convention in the viewer uses geom_xmat[:, 0], NOT the body X axis.
        # Use that same local geom frame for independent NumPy FK and IK.
        geom_quat = np.asarray(model.geom_quat[self.tool_geom_id], dtype=float).copy()
        geom_quat /= np.linalg.norm(geom_quat)
        self.tool_rotation_local = _quat_to_matrix(geom_quat)

    @staticmethod
    def _validate_angles(joint_angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(joint_angles, dtype=float)
        if angles.shape != (6,) or not np.all(np.isfinite(angles)):
            raise ValueError("joint_angles must be a finite vector with shape (6,)")
        return angles

    def _chain_state(
        self, joint_angles: np.ndarray, include_orientation: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = self._validate_angles(joint_angles)
        transform = np.eye(4)
        axes = np.zeros((3, 6), dtype=float)
        pivots = np.zeros((3, 6), dtype=float)

        for body_id in self.body_chain:
            transform = transform @ np.block(
                [
                    [
                        _quat_to_matrix(self.model.body_quat[body_id]),
                        self.model.body_pos[body_id, :, None],
                    ],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            )
            if self.model.body_jntnum[body_id]:
                joint_id = int(self.model.body_jntadr[body_id])
                joint_index = int(np.flatnonzero(self.joint_ids == joint_id)[0])
                joint_position = self.model.jnt_pos[joint_id]
                pivot = transform[:3, :3] @ joint_position + transform[:3, 3]
                axis = transform[:3, :3] @ self.model.jnt_axis[joint_id]
                axes[:, joint_index] = axis / np.linalg.norm(axis)
                pivots[:, joint_index] = pivot

                rotate_about_joint = np.eye(4)
                rotate_about_joint[:3, 3] = joint_position
                rotate_about_joint[3, 3] = 1.0
                rotation = np.eye(4)
                rotation[:3, :3] = _axis_angle(
                    self.model.jnt_axis[joint_id], angles[joint_index]
                )
                translate_back = np.eye(4)
                translate_back[:3, 3] = -joint_position
                transform = (
                    transform @ rotate_about_joint @ rotation @ translate_back
                )

        tool_position = transform[:3, :3] @ self.tool_offset + transform[:3, 3]
        return (tool_position, axes, pivots, transform[:3, :3].copy()) if include_orientation else (tool_position, axes, pivots)

    def forward(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return the tool point in world coordinates, in metres."""
        return self._chain_state(joint_angles)[0]

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return the 3-by-6 translational geometric Jacobian."""
        point, axes, pivots = self._chain_state(joint_angles)
        return np.cross(axes.T, (point[:, None] - pivots).T).T

    def beam_state(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World tool-geom point, geom-local +X beam axis, and Jacobians.

        Use the *compiled geom* rotation, not just the parent body rotation.
        MuJoCo geom_xmat may include a non-identity mesh-local orientation.
        The beam-direction derivative is d(axis)/dq_i = omega_i cross axis.
        This is geometric aiming only; no radiation is delivered.
        """
        point, axes, pivots, body_rotation = self._chain_state(joint_angles, include_orientation=True)
        direction = (body_rotation @ self.tool_rotation_local)[:, 0].copy()
        jv = np.cross(axes.T, (point[:, None] - pivots).T).T
        jd = np.cross(axes.T, np.tile(direction, (6, 1))).T
        return point, direction, jv, jd
