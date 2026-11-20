# CyberKnife-Style Tumor Motion Tracking (MuJoCo)

A six-revolute-joint robot arm tracks a synthetic breathing-driven target in
MuJoCo. The project includes student-implemented forward position kinematics,
the translational Jacobian, two numerical inverse-kinematics methods, a
joint-space feedback controller, repeatable experiments, CSV metrics, and
plots. It is an educational simulation only; it is not a clinical or treatment
planning system.

## Run it

Python 3.10 or later is recommended. From the repository directory:

```powershell
python -m pip install -r requirements.txt
python robot_simulation.py
```

The default command runs five experiments and writes sample logs,
`results/metrics.csv`, `results/metrics.json`, and
`results/tracking_results.png`.

To run one case:

```powershell
python robot_simulation.py --experiment nominal
python robot_simulation.py --experiment challenge
python robot_simulation.py --experiment pseudoinverse
python robot_simulation.py --experiment low-gain
python robot_simulation.py --experiment mid-gain
```

For the live MuJoCo demonstration:

```powershell
python robot_simulation.py --view
```

This opens a continuous simulation plus a control panel. Use **Play breathing**
or **Play circle** to run a repeating target trajectory, **Hold** to stop
tracking at the current joint positions, and **Go home** to return to the
initial pose. The six joint sliders set manual joint targets in degrees; moving
any slider switches to manual control. Close either window to end the live
session. The finite headless experiments remain available without `--view`.

Run the focused checks with:

```powershell
python -m pytest
```

## Model, methods, and assumptions

The six-joint serial chain and its fixed transforms, joint axes, limits, and
tool offset come from [`cyberknife_6dof.xml`](cyberknife_6dof.xml). STL meshes
are display geometry, scaled from millimetres to metres. The code uses MuJoCo
for model loading, rigid-body dynamics, and reference/validation positions;
it does not call MuJoCo's inverse kinematics or Jacobian routines.
The existing [`src/forward_kinematics.py`](src/forward_kinematics.py) remains a
separate DH reference implementation; the active simulation uses
[`src/kinematics.py`](src/kinematics.py) so its FK matches the CAD-derived
MuJoCo chain without guessing DH link dimensions.

For a joint on a body, the implemented transform rotates about its specified
local pivot:

`T_body(q) = T_fixed T(pivot) R(axis, q) T(-pivot)`.

Chaining these transforms gives the tool position `p(q)`. For revolute joint
`i`, the implemented translational Jacobian column is
`J_i = axis_i x (p - pivot_i)`. The tool point is position-controlled; tool
orientation is not constrained, leaving a redundant six-joint solution.

The inverse kinematics minimizes Cartesian position error. Damped least
squares uses `dq = J^T (J J^T + lambda^2 I)^-1 e`; the comparison method uses
the Moore-Penrose pseudoinverse. Both methods limit update size, enforce joint
limits, and use a short error-reducing line search. The damping is fixed at
0.04; a small singular value therefore does not cause unbounded steps.

The target follows the supplied chest/lung motion notes. Chest and lung
translations use `sin^(2n)(pi t/T)` and
`sin^(2n)(pi t/T - phi)`, respectively, and the lung displacement is rotated
by a small chest rotation about X. Default amplitudes, period, and phase are
explicitly documented in `src/tumor_trajectory.py`; they are illustrative,
not patient data.

At each 20 ms control update, position IK produces a desired joint vector.
The implemented saturated joint PD law is
`tau = sat(qfrc_bias + Kp(q_des - q) - Kd qdot)`. `qfrc_bias` is MuJoCo's
model-based bias-force term; PD feedback, torque saturation, IK, and the
Jacobian are implemented in Python. Gains and torque limits are defined near
the top of `src/simulation.py`; the fixed gains are a demonstrator setting,
not clinically validated tuning.

## Experiments and measures

1. **Nominal DLS:** nominal initial condition, 4 s breathing period, and
   default motion amplitudes.
2. **Challenging DLS:** perturbed initial joints, 1.5x motion amplitude,
   3.2 s breathing period, and a 0.8 Nm sinusoidal joint-three disturbance.
3. **Nominal pseudoinverse:** same initial condition and trajectory as the
   nominal DLS run, changing only the IK algorithm.
4. **Low-gain tuning:** nominal trajectory with both PD gain matrices scaled
   to 0.5.
5. **Mid-gain tuning:** nominal trajectory with both PD gain matrices scaled
   to 1.0; the nominal case uses the selected 1.5 scale.

`metrics.csv` reports RMS, 95th-percentile, peak, and final tool-to-target
position error; mean IK solve time; mean Jacobian condition number; and
post-1-second RMS tracking error and integrated squared actuator torque.
Per-sample CSVs retain target/tool positions, joint states, IK convergence
effort, and applied torque. Nominal DLS versus pseudoinverse is the IK
comparison. The gain sweep scales `Kp` and `Kd` together to preserve their
ratio. The selected 1.5 scale has lower measured tracking error with nearly
unchanged torque effort than the 1.0 setting; it is checked again in the
challenging case.

## Limitations and report

The robot inertial properties are those inferred by MuJoCo from the supplied
mesh/model; the model has no calibrated actuator identification or validated
patient coordinate registration. Robot-link collision is disabled in the XML,
and no collision avoidance, radiation dose, image guidance, breathing
estimation, or safety interlocks are simulated. Results must not be interpreted
as evidence of treatment accuracy or safety.
In the original **breathing-only** experiments, no separate obstacle-avoidance
planner is used: the task prescribes the Cartesian target path and online IK
generates the desired joint positions. The optional multi-node and initial
approach demonstrations use the separate NumPy RRT-Connect implementation.

See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for the concise
submission report, including generated quantitative results and a group
contribution section to complete before submission.

## Added joint-space path planning and trajectory generation

The optional two-phase demonstration (initial positioning, then respiratory
tracking) is documented in [`docs/PLANNING_INTEGRATION.md`](docs/PLANNING_INTEGRATION.md).
From the repository root:

```powershell
python robot_simulation.py --plan          # headless run and quantitative outputs
python robot_simulation.py --plan-view     # MuJoCo viewer
python -m pytest -q tests/test_planning.py
```

This uses a NumPy RRT-Connect planner and quintic joint trajectory; original
simulation commands remain unchanged. Obstacle collisions use simplified
link-capsule proxies, not full mesh collision or verified medical geometry.


## Synthetic brain patient + tumour scene (added)

Run `python robot_simulation.py --brain-view` for the table, mannequin and
synthetic brain target, including an RRT/trajectory-controlled approach and
a virtual beam axis with measured miss distance.
Use `python robot_simulation.py --brain` for headless result logs.
This is a geometric educational demo, not a real radiotherapy simulation.
See [`docs/BRAIN_TARGETING.md`](docs/BRAIN_TARGETING.md) for details and limitations.

## New: CyberKnife-style multi-node brain movement demo

Run `python robot_simulation.py --cyberknife-view` to demonstrate seven
non-coplanar synthetic brain-target viewpoints with NumPy geometric
position-and-direction IK, RRT-Connect repositioning, quintic interpolation,
PD-driven MuJoCo movement, and per-node beam-axis error measurements.
Use `--cyberknife` for headless mode. See `docs/CYBERKNIFE_STYLE_MOTION.md`.
No radiation is simulated; this is not a clinical treatment model.

## Revised multi-viewpoint controller + collision diagnostics

The `--cyberknife` / `--cyberknife-view` workflow was revised after a previous run showed 15 rad/s joint speeds and approximately 200 mm virtual-beam miss distances. It now uses NumPy inertia-scaled computed-torque PD, independent FK/beam-axis verification against MuJoCo, box/sphere collision proxies, sampled path checking, physics-step proxy monitoring, and a **fail-closed** settle/aim gate before advancing between nodes.

```powershell
python -m pytest -q tests/test_safety_regressions.py tests/test_beam_ik.py tests/test_brain_scene.py tests/test_planning.py
python robot_simulation.py --cyberknife-view
# Optional simplified gravity-free diagnostic (NOT real-world dynamics):
python robot_simulation.py --cyberknife-view --cyberknife-zero-g
```

See `docs/STABILIZATION_AND_VALIDATION.md`. Outputs under `results/cyberknife_motion` must be regenerated. The original CSVs in this package are historical runs, **not evidence that the revised controller works**. Note that the collision model remains approximate and neither robot self-collision nor mesh-level safety is guaranteed. A failed stage stops the run and records its reason.


## V3 damping-compensated multi-node simulation (MuJoCo 4.0 compatible)

The latest multi-node controller includes **passive joint damping compensation**:
`tau = clip(qfrc_bias - qfrc_passive + M @ u, -limits, limits)` with
`u = qdd_des + wn²*(q_des-q) + 2*zeta*wn*(qd_des-qd)`. This corrects
a first-node failure where the robot nearly stopped while still 0.294 rad
from the commanded pose. No actuator torque limits were increased.

The patient/table collision proxy set was also extended, and infeasible
viewpoint candidates are now skipped rather than accepted. This remains an
approximate collision model, **not actual mesh contact or patient safety**.

**Run the real MuJoCo check locally before claiming success:**

```powershell
python -m pytest -q
python robot_simulation.py --cyberknife --cyberknife-nodes 1
python robot_simulation.py --cyberknife
python robot_simulation.py --cyberknife-view
```

The headless process now returns a nonzero exit code if any node fails.
See `docs/CONTROLLER_V3_DIAGNOSIS.md` for the failure analysis, new logs,
exact alignment thresholds and verification caveats.

## New: breathing-observed synthetic lung-target compensation

This is a **separate simulation**, leaving the successful seven-viewpoint
brain demonstration untouched. A moving external green chest marker and a
transparent rising chest panel provide synthetic respiratory observations.
A harmonic observer estimates respiratory phase/period from noisy chest-marker
height (not from the true tumour coordinates). A stated calibrated model
predicts the internal red target's position. Beam-axis DLS IK and the existing
computed-torque controller adjust the robot's **virtual beam aim** during breathing.

```powershell
python robot_simulation.py --lung              # compared against a static beam, saves plots
python robot_simulation.py --lung-view         # visualizes compensated motion
python robot_simulation.py --lung-record       # records MP4 and runs comparison
python robot_simulation.py --lung --lung-duration 16
```

Outputs: `results/lung_tracking/` (`lung_summary.json`, compensated and static
baseline CSV logs, `lung_tracking_results.png`, optional MP4). See
`docs/LUNG_RESPIRATORY_COMPENSATION.md` for mathematics, caveats and debugging.
**No camera, real image processing, radiotherapy or clinical validation is present.**


## AI4000 Lecture 33–37 simulated eye-to-hand visual servoing

`python robot_simulation.py --lung-vision` compares simulated fixed-camera IBVS respiratory compensation against a static beam; `--lung-vision-view` displays the compensated scene and `--lung-vision-record` records its MP4. See `docs/AI4000_VISION_BASED_LUNG_TRACKING.md` for the pinhole camera, image interaction matrix, eye-to-hand joint Jacobian, DLS IBVS law, limitations and debugging guide. This mode uses synthetic projected pixels, **not** actual video image processing or validated clinical tumour motion data.
