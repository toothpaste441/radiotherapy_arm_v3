# Wrist joint response: diagnostic model update

This is an **educational modelling hypothesis**, not a validated physical model.

The real MuJoCo run showed a first-node joint error of 0.152 rad despite near-zero base-joint errors, zero torque saturation and low wrist velocities. The mesh-derived wrist inertias are very small compared with the original large joint damping.

The XML now explicitly includes **reflected joint armature inertia** at J4 = 0.01 kg m², J5 = 0.02 kg m² and J6 = 0.02 kg m². These values are illustrative and must not be claimed as measured CyberKnife motor parameters. They improve numerical conditioning and change the simulated robot dynamics.

The controller and fail-closed alignment threshold are unchanged. No collision thresholds were weakened. Rerun **all** physics experiments after altering inertial properties. Do not compare old results as though they came from the same model.

Run these in the extracted project directory with MuJoCo 4.0:

```powershell
python -m pytest -q tests/test_mujoco_controller_integration.py -vv
python robot_simulation.py --cyberknife --cyberknife-nodes 1
python robot_simulation.py --cyberknife
```

If the wrist servo regression fails, inspect the six printed joint errors; do not silently relax the limits or alignment criteria. A successful servo regression does not prove that the multi-node trajectory will succeed.

IMPORTANT: This change has passed static tests only in the build environment, because the MuJoCo wheel is not available there. All physics execution requires the user's local installation.
