# Lung patient-arm collision setup fix

The original lung scene translated the brain demonstration's mannequin to
place the synthetic target within the torso. This simultaneously moved a
patient arm into the formerly collision-free treatment pose.

The program now searches deterministic left/right outward arm offsets of
0, 4, 8, 12, 16, 20 and 25 cm. For every candidate it rebuilds the visual
capsules *and* their collision boxes together, checks home and treatment
poses, runs RRT-Connect, and collision-checks samples of the resulting
quintic approach. The first accepted scene is used for simulation.

The resulting visual mannequin is schematic. An arm spread selected to
avoid the robot is not patient anatomy or a safe radiotherapy pose.
Do not disable the patient-arm collision proxies to suppress failure.

Start with:
    python -m pytest -q
    python robot_simulation.py --lung-view
or:
    python robot_simulation.py --lung-vision-view

Observe `Lung scene: selected arm offsets ...` in the terminal.

If all layouts fail, the error message lists rejected configurations.
Then the treatment pose and overall scene geometry must be redesigned;
do not assume the planner can fix an invalid goal.

The complete MuJoCo run has not been executed in the packaging environment.

## Robot-base relocation update

The lung example now also searches robot-base **translations** in `src/lung_tracking_demo.py` and recompiles the XML for each candidate, recalculating IK in the new world geometry. See `docs/LUNG_ROBOT_BASE_RELOCATION.md`. Arm spread is secondary; collision checks remain active.
