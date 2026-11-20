# Planning and trajectory integration (AI4000)

## What was added

- `src/path_planning.py`: independent NumPy RRT-Connect with joint limits, sampled edges, spherical obstacle/capsule-link collision approximation, and path shortcutting.
- `src/trajectory_generation.py`: C2-continuous quintic stop-at-waypoints trajectories, including positions, velocities, and accelerations. Each segment's duration enforces a chosen maximum joint speed.
- `src/planning_experiments.py`: integrates the planner with the existing MuJoCo robot, PD actuators, numerical IK, and breathing target; writes metrics and plots.
- `tests/test_planning.py`: four unit tests for geometry, collision validation, planning, and trajectory interpolation.
- `robot_simulation.py`: two new options (`--plan`, `--plan-view`). The original options and existing results are unchanged.

## Installation and commands (Windows PowerShell, from project root)

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_planning.py
python -m pytest -q
python robot_simulation.py --plan
python robot_simulation.py --plan-view
```

`--plan` runs a headless **approach phase** followed by 8 s of respiratory tracking. `--plan-view` shows the MuJoCo viewer. You should test `--plan` first. The original `python robot_simulation.py` still runs all five baseline experiments; `python robot_simulation.py --view` still launches the original interactive target-tracking view.

**Output files** after running `--plan`:

- `planning_scene.xml`: generated XML with a **visual-only orange sphere** representing an illustrative obstacle. The original `cyberknife_6dof.xml` is not changed.
- `results/planning/planned_joint_waypoints.csv`: planned RRT route after shortcutting.
- `results/planning/approach_and_tracking.csv`: control samples and tracking error.
- `results/planning/planning_summary.json`: planning time, iterations, collision checks, path length, RMS errors.
- `results/planning/planning.png`: tool tracking error and joint position plots.

## Algorithm and mathematics

1. Set the start to `HOME_ANGLES` (from `src/simulation.py`). Set an **illustrative** treatment joint configuration by adding `(0.45,-0.38,0.32,0.30,-0.22,0.18)` radians and clipping into joint limits.
2. Generate an artificial spherical obstacle near the midpoint of the tool's starting and finishing positions. The start and end configurations are checked for collision.
3. Search in six-dimensional joint configuration space using RRT-Connect. A tree repeatedly samples a joint vector, finds the nearest node, extends by bounded joint-space distance, and checks all interpolated configurations. Two trees are grown from the start and goal and connected. Shortcutting reduces unnecessary waypoints.
4. Each link is **approximated** by a capsule whose axis connects two successive revolute joint pivots (the last connects joint six to the tool). Each capsule is checked against a sphere. Joint limits are checked explicitly. These are simplified planning proxies: **not exact STL collision, not self-collision checking, and not patient geometry**. The orange obstacle in MuJoCo has collisions disabled; the NumPy proxy checks enforce avoidance at the path planning level. This is educational only.
5. Time-parameterize the route with quintic polynomial `h(s)=10s^3-15s^4+6s^5`, `s in [0,1]`. For a waypoint difference `Delta q` and segment duration `T`, desired `q=q0+h(s) Delta q`. `qdot=h'(s) Delta q/T` and `qddot=h''(s) Delta q/T^2`. The peak velocity is `1.875 |Delta q|/T`.
6. Apply the existing saturated PD plus MuJoCo bias compensation, with `qd_des` feedforward through derivative feedback during the approach. Continue with position-only numerical DLS IK and the existing respiratory motion model when the approach trajectory finishes.

## Important limitations / what to verify on your laptop

- MuJoCo was not available in the packaging environment; the new integrated dynamics run was **not** executed. The four independent NumPy planning tests and Python syntax checks passed.
- Because the planned obstacle is not a physical contact geom, the runtime does **not** generate collision/contact forces for it. Avoidance was checked at discrete planning and trajectory states. Dynamic tracking errors could cause a real arm to intersect the obstacle. For a stronger project, implement collision geometry for the body and links and check the **actual** tracked arm, not just its planned joint targets.
- The approach uses zero velocity and acceleration at every waypoint (smooth but conservative). The final pose is a demonstrator configuration, not a verified radiotherapy treatment configuration.
- The controller's wrist torque saturation found in the existing experiment logs remains unresolved. Check the new run's torque logs too.
- At the approach-to-tracking switch, the controller begins online breathing IK. A transition period might be needed if the final approach error is large.
- Complete `docs/TECHNICAL_REPORT.md` with actual output plots, timing, limitations, and the individual contribution section after you run the code locally.

## If planning fails

- Verify that the STL mesh filenames and `assets/` folder are intact.
- If an endpoint collision error occurs, adjust the illustrative goal in `_make_goal` or obstacle radius/placement in `_choose_obstacle` in `src/planning_experiments.py`.
- If RRT fails, increase `max_iterations` in `rrt_connect` or adjust the obstacle and seed.
- If physics or IK fails during the transition, inspect joint torque saturation, consider more conservative approach speeds, or start with a home-to-goal approach only. Do not silently bypass failed safety/collision checks.
