"""Compatibility helpers for MuJoCo 3.x and 4.x Python bindings.

MuJoCo 4.0 changed `mj_fullM(model, dst, data.qM)` to
`mj_fullM(model, data, dst)` and removed `MjData.qM`.
This small adapter does *not* implement dynamics: MuJoCo still supplies M(q).
"""
from __future__ import annotations

import numpy as np


def fill_dense_mass_matrix(mujoco_module, model, data, destination: np.ndarray) -> None:
    """Fill preallocated (nv, nv) array with generalized inertia matrix.

    Accepts the MuJoCo module explicitly so compatibility can be unit-tested
    without requiring a physics installation.
    """
    if destination.shape != (model.nv, model.nv):
        raise ValueError("Mass-matrix output must have shape (model.nv, model.nv)")
    if hasattr(data, "qM"):
        # MuJoCo 3.x signature; qM uses MuJoCo's legacy sparse representation.
        mujoco_module.mj_fullM(model, destination, data.qM)
    else:
        # MuJoCo 4.x signature; the data object contains the CSR matrix M.
        mujoco_module.mj_fullM(model, data, destination)
    if not np.all(np.isfinite(destination)):
        raise RuntimeError("MuJoCo returned a non-finite joint inertia matrix")
