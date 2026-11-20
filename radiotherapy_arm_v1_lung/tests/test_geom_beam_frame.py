"""Beam axis must follow tool geom orientation, not just its parent body."""
import numpy as np
from src.kinematics import RobotKinematics, _quat_to_matrix


def test_beam_state_uses_mesh_geom_local_rotation():
    kin = RobotKinematics.__new__(RobotKinematics)
    # A non-identity geom quaternion rotates geom +X onto body +Y.
    angle = np.pi / 2
    kin.tool_rotation_local = _quat_to_matrix(
        np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)])
    )
    kin._chain_state = lambda q, include_orientation=False: (
        np.array([1., 2., 3.]),
        np.tile(np.array([[0.], [0.], [1.]]), (1, 6)),
        np.zeros((3, 6)),
        np.eye(3),
    )
    point, axis, jv, jd = kin.beam_state(np.zeros(6))
    assert np.allclose(point, [1, 2, 3])
    assert np.allclose(axis, [0, 1, 0], atol=1e-12)
    assert np.allclose(jd[:, 0], [-1, 0, 0], atol=1e-12)
    assert jv.shape == (3, 6)
