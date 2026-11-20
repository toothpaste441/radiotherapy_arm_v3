"""Regression tests for the MuJoCo 3.x -> 4.x inertia API change.

These use a tiny fake API, so no MuJoCo installation is needed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.mujoco_compat import fill_dense_mass_matrix


def test_new_mujoco_4_api_uses_data_object():
    model = SimpleNamespace(nv=6)
    data = SimpleNamespace(M=np.ones(21))  # new API has no qM
    full = np.zeros((6, 6))

    class FakeMj:
        def mj_fullM(self, actual_model, actual_data, destination):
            assert actual_model is model
            assert actual_data is data
            destination[:] = 2 * np.eye(6)

    fill_dense_mass_matrix(FakeMj(), model, data, full)
    np.testing.assert_allclose(full, 2 * np.eye(6))


def test_legacy_mujoco_3_api_uses_qM_sparse_array():
    model = SimpleNamespace(nv=6)
    sparse = np.arange(21, dtype=float)
    data = SimpleNamespace(qM=sparse)
    full = np.zeros((6, 6))

    class FakeMj:
        def mj_fullM(self, actual_model, destination, actual_sparse):
            assert actual_model is model
            assert actual_sparse is sparse
            destination[:] = np.eye(6)

    fill_dense_mass_matrix(FakeMj(), model, data, full)
    np.testing.assert_allclose(full, np.eye(6))


def test_destination_must_match_model_dimensions():
    with pytest.raises(ValueError, match="Mass-matrix"):
        fill_dense_mass_matrix(None, SimpleNamespace(nv=6), object(), np.zeros((3, 3)))


def test_rejects_nonfinite_mass_matrix():
    model = SimpleNamespace(nv=6)
    class FakeMj:
        def mj_fullM(self, actual_model, actual_data, destination):
            destination[:] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        fill_dense_mass_matrix(FakeMj(), model, object(), np.zeros((6, 6)))
