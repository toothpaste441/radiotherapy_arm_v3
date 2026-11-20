# CyberKnife-Style Tumor Motion Tracking

## Problem and assumptions

This project simulates a six-revolute-joint serial arm tracking a moving point
representing a tumor. The arm is modelled in MuJoCo from the supplied STL
assembly and `cyberknife_6dof.xml`. Link inertias are inferred from the mesh
model, links are rigid, and joint limits are taken from the XML. The target is
a synthetic chest/lung displacement, not patient data. Position tracking is
evaluated; the tool orientation is unconstrained.

The work is simulation-only and is not a clinical CyberKnife model, a
radiotherapy treatment plan, or a safety/accuracy claim.

## Mathematical method

For each revolute joint, the student implementation composes the XML fixed
body transform with a rotation around the specified local joint pivot:

`T_body(q) = T_fixed T(pivot) R(axis, q) T(-pivot)`.

The tool position `p(q)` is obtained by chaining these transforms. The
translational Jacobian is computed geometrically:

`J_i = axis_i x (p - pivot_i)`.

For Cartesian error `e = p_target - p(q)`, the DLS update is
`dq = J^T (J J^T + lambda^2 I)^-1 e`, with `lambda = 0.04`. The comparison
method uses `dq = J^+ e`. Both methods apply a step limit, joint-limit clipping,
and an error-reducing line search. The target is position-only, so the 6-DoF
arm has three redundant degrees of freedom.

The target uses the supplied breathing form:
`u(t) = sin^(2n)(pi t/T)` for chest motion and
`v(t) = sin^(2n)(pi t/T - phi)` for lung motion. Lung displacement is rotated
by a small chest rotation about X. The illustrative nominal settings are
`T = 4 s`, `n = 1`, `phi = 0.5 rad`, chest amplitudes `(12, 0, 18) mm`,
lung amplitudes `(8, 4, 10) mm`, and chest rotation amplitude `0.02 rad`.
For the original breathing-only experiment, the prescribed Cartesian tumour
path is the reference trajectory and no separate free-space planner is used.
The optional initial-positioning and multi-viewpoint extensions instead use
student-implemented NumPy RRT-Connect and quintic joint-space interpolation,
with limited capsule/primitive collision proxies.

At 20 ms intervals, IK supplies the desired joint positions. The student
controller computes torque as
`tau = sat(qfrc_bias + Kp(q_des - q) - Kd qdot)`, using the MuJoCo bias-force
term for model-based compensation and the XML motor actuators for torque
application. The gains and torque limits are fixed in `src/simulation.py`.
They were selected as simulation demonstrator values, not tuned for a real
device. Tuning is evaluated reproducibly by multiplying both gain matrices by
0.5, 1.0, and 1.5 on the same nominal target, while keeping the damping ratio
and all other settings unchanged. Compare tracking RMS with integrated
squared torque. Select the 1.5 setting because the measured RMS error falls
while torque effort changes negligibly relative to 1.0, then check it again
in the challenging case. The gains used for nominal and challenging runs are
`Kp = (180, 180, 150, 52.5, 45, 30) Nm/rad` and
`Kd = (37.5, 37.5, 30, 12, 12, 9) Nms/rad`, with torque limits
`(80, 80, 60, 25, 20, 15) Nm`. This simple gain sweep is not a clinical
tuning procedure.

## Simulation experiments

| Case | IK | Initial condition / operating change |
|---|---|---|
| Nominal | DLS | Nominal joints; nominal breathing trajectory |
| Challenging | DLS | Joint offset `(0.30, -0.20, 0.25, 0.10, -0.15, 0.15) rad`; 1.5x motion, 3.2 s period, 0.8 Nm joint-three disturbance |
| Comparison | Pseudoinverse | Same initial state and trajectory as nominal DLS |
| Low-gain tuning | DLS | Nominal case with `Kp` and `Kd` both scaled by 0.5 |
| Mid-gain tuning | DLS | Nominal case with `Kp` and `Kd` both scaled by 1.0 |

The DLS-versus-pseudoinverse comparison changes the IK algorithm only. CSV
metrics include RMS, 95th-percentile, peak, and final Cartesian tracking error,
mean IK computation time, mean Jacobian condition number, and integrated
squared torque. `results/tracking_results.png` visualizes the 3D paths and
tracking errors. Re-run `python robot_simulation.py` to regenerate the data.

### Results

The numerical table below is populated from the reproducible simulation output
in `results/metrics.csv`. Position errors are in millimetres unless otherwise
marked; post-1s RMS excludes the startup interval to separate initial-state
transient from continued tracking.

| Case | RMS | Post-1s RMS | 95th percentile | Peak | Final | Mean IK time (ms) | Torque-squared integral (N²m²s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal DLS (1.5x) | 4.82 | 5.00 | 8.61 | 8.90 | 0.91 | 1.510 | 15,336.5 |
| Challenging DLS (1.5x) | 29.87 | 8.38 | 36.47 | 222.66 | 11.30 | 1.671 | 15,445.2 |
| Nominal pseudoinverse (1.5x) | 4.81 | 5.00 | 8.60 | 8.89 | 0.89 | 1.471 | 15,336.5 |
| Low-gain DLS (0.5x) | 6.50 | 6.76 | 10.58 | 10.92 | 3.01 | 1.453 | 15,314.5 |
| Mid-gain DLS (1.0x) | 5.25 | 5.45 | 9.15 | 9.46 | 1.49 | 1.418 | 15,333.0 |

Compare nominal DLS and nominal pseudoinverse using RMS and peak error,
computation time, and effort; no method should be declared superior without
referencing the measured values. Compare the three gain settings using the
same nominal trajectory. The challenging run's full-horizon peak includes the
deliberate initial joint offset; report the post-1s RMS as well when discussing
continued tracking.

On this nominal trajectory, the 1.5x gain setting reduces RMS error by about
8.2% relative to 1.0x, while the integrated squared torque increases by about
0.02%. The DLS and pseudoinverse cases differ by about 0.008 mm in RMS error;
their average Jacobian condition numbers are approximately 4.77, so this case
does not show a meaningful tracking advantage for either IK method. IK timing
is a single local run and should be treated as indicative only. In the
challenging case, the 223 mm peak is the deliberate initial-position
transient at the start; post-1s RMS is 8.38 mm. That transient and the
remaining error are important limitations, not evidence of clinical accuracy.

## Limitations and conclusion

This simulation does not model image registration, target-estimation delay,
respiratory sensor noise, collision avoidance, dose delivery, or clinical
constraints. Mesh-derived inertias and the simplified target motion are
unvalidated. The output demonstrates a student-implemented kinematics/IK and
feedback-control pipeline under explicit synthetic cases only. The measured
nominal tracking errors are a few millimetres in this model; the challenging
initial-condition transient is much larger and highlights the need for
acquisition/settling logic before tracking. These results do not imply
treatment accuracy or safety.

## Reproducibility

Install `requirements.txt`, run `python robot_simulation.py`, run
`python -m pytest`, then use `python robot_simulation.py --view` for the live
demonstration. The live control panel provides manual joint sliders, breathing
and circular path buttons, hold, and home controls; it runs until a window is
closed. Generated CSV and PNG outputs are stored in `results/`.


## Additional synthetic brain-target motion experiment (V3; not yet dynamically validated)

The extended scene adds a mannequin and treatment table with a stationary
synthetic brain point. Seven illustrative tool viewpoints are computed from
position-plus-beam-axis inverse kinematics (analytic beam-axis Jacobian,
NumPy damped least squares); RRT-Connect plans joint-space motion between
configurations, and quintic interpolation produces desired positions,
velocities and accelerations. MuJoCo steps the resulting simulated dynamics.

A subsequent controller revision uses inertia-scaled computed-torque PD with
MuJoCo passive damping compensation, and an arrival gate that demands
consistent joint settling and visual-beam alignment before advancing.
The independent FK and beam axis were previously verified against actual
MuJoCo 4.0 poses to numerical precision; however, the revised gravity-loaded
seven-viewpoint simulation has **not** been verified as successful in this
report. For limitations and the exact run commands see
`docs/CONTROLLER_V3_DIAGNOSIS.md`.

Do not conflate the historical numeric breathing-tracking results above with
new brain-targeting results. No clinical safety, radiation, or dose claims are
made. Collision geometry is approximate; no STL-mesh collision validation.
Group members must fill their individual contribution details before submission.
