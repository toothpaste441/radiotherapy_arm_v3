# MuJoCo 3.x and 4.x mass matrix compatibility

The `--cyberknife` mode uses a NumPy computed-torque controller with MuJoCo's generalized inertia matrix `M(q)`.

In older MuJoCo, `mj_fullM(model, dense, data.qM)` converted the mass matrix to dense form. As of MuJoCo 4.0, the API is `mj_fullM(model, data, dense)`, and the `qM` field is no longer available. The compatibility adapter `src/mujoco_compat.py` supports both variants.

Run:

```powershell
python -c "import mujoco; print(mujoco.__version__)"
python -m pytest -q
python robot_simulation.py --cyberknife
python robot_simulation.py --cyberknife-view
```

The local regression tests mock the MuJoCo API and verify argument ordering. They are not a substitute for running dynamics and validating controller settling, physical clearance, and beam alignment in the installed MuJoCo version.
