# Beam orientation hotfix (2026-11-20)

## Failure addressed
`Independent FK / MuJoCo mismatch: point=5.978734e-16 m, direction=0.70895024`

The independent FK matched the MuJoCo tool position, but the orientation code returned the tool **body** +X axis while the dynamic validation used the tool **mesh geometry** +X axis (`data.geom_xmat[tool_geom_id][:,0]`). For meshes, the compiled `model.geom_quat` can be nonidentity because of MuJoCo's mesh coordinate-frame corrections. These two axes are not generally identical.

## Correction
`src/kinematics.py` now stores the normalized `model.geom_quat[tool_geom_id]` and converts it to a rotation matrix. `RobotKinematics.beam_state()` returns the compiled tool-geometry +X direction:

`d_world = R_world_body @ R_body_geom @ [1, 0, 0]`

The Jacobian uses `omega_i x d_world`. This makes the IK beam convention consistent with the beam ray displayed and validated using `geom_xmat`. The beam is illustrative and does not constitute a calibrated physical radiation axis.

## Test
`python -m pytest -q tests/test_geom_beam_frame.py tests/test_safety_regressions.py tests/test_beam_ik.py tests/test_brain_scene.py tests/test_planning.py`

The standalone tests passed in the preparation environment. MuJoCo was not installed in that environment, so **the actual FK comparison and full dynamics still require a local run**.

Run: `python robot_simulation.py --cyberknife` (or `--cyberknife-view`). If the geometry validation still fails, capture `model.geom_quat[kin.tool_geom_id]`, `data.geom_xmat[kin.tool_geom_id]`, and `kin.beam_state(q)[1]` for diagnosis. If the validation passes but a node fails to settle, that is a separate controller/feasibility issue; do not remove the fail-closed checks.
