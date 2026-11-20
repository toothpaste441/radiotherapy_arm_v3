# Lung simulation: collision-aware robot-base relocation

## Why it was added

The previous lung scene moved the mannequin into the treatment workspace, but kept the old brain-demo robot base. Collision checking correctly rejected the initial home pose because a modeled robot-link capsule intersected `left_arm_box`, even with arms spread.

## New algorithm (implemented in `src/lung_tracking_demo.py`)

1. Derive the original *world-coordinate* tool source and beam axis from the original CAD geometry.
2. Place the synthetic lung lesion, mannequin, table and camera marker **once in this original world frame**. The tumour remains fixed when relocating the robot.
3. Try a deterministic sequence of planar `(dx, dy, 0)` translations of the robot's fixed `robot_base` body in the generated XML, initially using zero translation and then progressively larger offsets to either side of the table. Keep the base's original quaternion, all six joints, the floor, and patient geometry intact.
4. Compile each candidate scene as a new MuJoCo model and instantiate a fresh `RobotKinematics` from it. This is essential: all six pivots/axes in world coordinates change when the base moves.
5. Reject any translated model whose home configuration fails approximate capsule-vs-patient collision checks.
6. Recompute a **new numerical position-and-beam-axis IK** configuration that aims at the *unchanged* world target from the *unchanged* world tool reference. Reject unreachable or colliding IK goals.
7. Run RRT-Connect from the unchanged joint-angle home configuration to that new goal, then verify the quintic trajectory at intermediate configurations. Select the first model that passes all of these checks.
8. Execute the existing computed-torque control and AI4000 eye-to-hand IBVS in that selected model; during movement continue checking the proxy collision model.

The scene is intentionally schematic, **not** a physically validated patient/robot safety model. Exact mesh/self-collision and continuous swept-volume contact are not proven.

## Commands

```powershell
python -m pytest -q
python robot_simulation.py --lung-vision
python robot_simulation.py --lung-vision-view
python robot_simulation.py --lung-vision-record
```

The program prints:

```
Lung scene: robot base offset = (...) m; arm offsets = (...) m; RRT waypoints = ...; rejected placement attempts = ...
```

The output summary also records `robot_base_offset_world_m` and `patient_arm_spread_m` for each experiment.

## Debugging

- If `home near left_arm_box` remains for all candidates, inspect the *world* base placement and `left_arm_box` extent rather than dropping the obstacle.
- If `goal IK infeasible/colliding`, the base may be too far from the target or the wrist orientation may be outside its feasible range. Select another world base offset or adjust the synthetic mannequin layout, then rerun the IK and all collision tests.
- If RRT fails, the chosen start/goal may be in disconnected regions under the simplified collision model.
- If the physics run fails after selection, inspect actual joint tracking, controller torque saturation and online IK fallbacks. Collision-valid offline geometry does not guarantee successful dynamics.
- This implementation relies on `mujoco` 4.0 on your Windows machine. The full physics run has **not** been verified in the creation environment, where MuJoCo is unavailable.

The original seven-viewpoint brain demonstration remains a separate command and its original robot XML is unchanged.
