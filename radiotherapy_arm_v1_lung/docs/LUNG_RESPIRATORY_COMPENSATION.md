# Synthetic lung target — external chest-marker observation and compensation

## Important distinction

This is an **educational, simulation-only** demonstration. It does **not** detect a lung tumour from an image, and it has no real camera, radiation, dose calculation, treatment planning, clinical registration, or validated patient collision avoidance. An external chest marker is **synthetically moved** and read with simulated measurement noise. A *known, assumed calibration* maps its breathing phase to internal synthetic tumour displacement. This correlation is not generally valid in patients without measured calibration.

The robot **does not stop breathing**. Instead, it changes its **virtual beam direction** as the internal target moves, attempting to keep the beam axis aligned with the tumour. Tool source position remains approximately fixed, which avoids commanding the arm to enter the patient's torso.

## Commands

In the extracted `radiotherapy_arm_v1` folder (MuJoCo 4.0 supported through `src/mujoco_compat.py`):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_lung_observer.py
python robot_simulation.py --lung
python robot_simulation.py --lung-view
python robot_simulation.py --lung-record
```

- `--lung`: headless **two experiments** (compensated vs same static-beam baseline), plus CSV, JSON and PNG.
- `--lung-view`: viewer for compensated run alone, useful for a live presentation.
- `--lung-record`: record compensated run to `results/lung_tracking/lung_compensation.mp4` and also run static baseline headlessly.
- `--lung-duration 16`: change tracking duration in simulated seconds.

The original `--cyberknife-view`, `--cyberknife-record`, `--view`, and five respiratory experiment commands remain unchanged.

## Complete signal → robot workflow

1. `src/lung_scene.py` generates a simplified table, patient and lung lesion near a reachable reference beam pose. It also adds a **green external chest marker** (mocap), a **red internal target**, and **yellow beam-axis markers**.
2. `src/lung_observer.py` generates the external chest marker waveform `u(t) = [1-cos(2πt/T)]/2` for the *simulation only*. A noisy scalar height reading is taken from the green mocap marker. **The estimator has no access to the red tumour's true position**.
3. `ChestMarkerObserver` fits a sinusoid to a window of *past readings* using `np.linalg.lstsq` and searches candidate breathing periods 3.2–4.8 s. Once enough measurements accumulate, it estimates the chest phase and transforms it into a predicted internal displacement using the assumed fixed, phase-lagged chest–lung calibration.
4. The internal tumour is moved according to an independent synthetic ground-truth function, with chest and lung displacement, a phase lag, and a small chest rotation. This ground truth is used for **visualization and error metrics only**.
5. `src/beam_ik.py` uses the same **position + beam-axis damped least squares IK** as the brain demo. It keeps the source point fixed and changes the ray direction from source to **estimated** tumour position. It never uses ground-truth target position for IK.
6. An initial `RRT-Connect` + quintic trajectory brings the six-joint robot to the reference treatment pose. Then online IK updates the joint targets at the 20 ms command period.
7. `src/stable_joint_control.py` computes torque from the actual MuJoCo `M`, bias and passive forces. This occurs every 2 ms physics step; torque commands are clipped to the XML actuator limits.
8. Each timestep checks joint-state validity, overspeed and simplified collision proxy intersections. If the online IK or collision check fails, the last valid command is retained and the event is counted in `ik_fallbacks`.
9. A separate static-beam run starts from the **same reference pose**, with exactly the same simulated target and sensor noise seed, but without updating its beam orientation. This is the no-respiratory-compensation baseline.

## Key equations

External surface motion:

\[
 y(t)=A_m\frac{1-\cos(\omega t)}{2}+\epsilon(t),\quad \omega=2\pi/T.
\]

Harmonic model fitted *only to past measurements*:

\[
 y(t)\approx a+b\cos(\hat\omega t)+c\sin(\hat\omega t).
\]

Calibrated internal motion:

\[
 p_t(t)=p_0 + A_c u(t)+ R_x(\theta_c(t))A_l v(t)-d(0),
 \qquad v(t)=\frac{1-\cos(\hat\theta(t)-2\phi)}{2}.
\]

Beam direction from the fixed tool source to the **estimated** target:

\[
 \hat d_d(t)=\frac{\hat p_t(t)-p_{source}}{\|\hat p_t(t)-p_{source}\|}.
\]

Online beam-axis IK:

\[
\Delta q=J_w^T(J_wJ_w^T+\lambda^2I)^{-1}
\begin{bmatrix}p_{source}-p(q)\\w(\hat d_d-\hat d(q))\end{bmatrix}.
\]

Computed-torque control:

\[
\tau=\operatorname{clip}\{M(q)[\ddot q_d+\omega_n^2(q_d-q)
 +2\zeta\omega_n(\dot q_d-\dot q)]+b-\tau_{passive}\}.
\]

The controlled quantity is the **virtual forward beam ray** (not beam dose, not patient movement). Geometric ray miss and internal-estimate error are separately evaluated against the known simulation truth.

## Saved files

`results/lung_tracking/lung_summary.json`: compensated & static-baseline RMS/P95/max ray miss, internal estimate RMS, observer period, saturation count, IK fallback count, and comparison improvement percentage.

`results/lung_tracking/compensated.csv`, `static_baseline.csv`: timestamped measured chest displacement, estimated period, estimated/actual synthetic internal target positions, beam miss, joint commands, joint measurements, joint error and speed.

`results/lung_tracking/lung_tracking_results.png`: time-series comparison and observer error.

`results/lung_tracking/lung_compensation.mp4`: only if `--lung-record` is used.

## Debugging

- **Scene/approach proxy collision:** the generated anatomy occupies the robot's workspace. Adjust schematic geometry in `lung_scene.py` (not the actual robot collision limits) or start from another validated source pose; never disable checks just to pass.
- **Observer not ready:** allow at least three seconds of external readings. The approach and settling phases provide calibration samples.
- **Estimator error high:** inspect `estimated_period_s`, noise, surrogate calibration phase/amplitude, and assumed period grid. An external chest marker does not uniquely determine internal tumour motion.
- **Large beam miss with small estimated-target error:** check beam-frame convention, beam IK convergence, joint following, and actuator saturation.
- **Many `ik_fallbacks`:** online IK targets may be infeasible or near joint/collision limits. Keep last safe desired joint configuration and report the limitation.
- **Simulation doesn't outperform static beam:** report the result accurately. This is a comparison, not a guaranteed improvement; tune the observer and controller using logged data, not just hand-picked displayed errors.
- **Original brain demo:** `--cyberknife-view` and its files are deliberately unchanged.

## Testing and claims

The new observer has standalone tests for periodicity, noisy marker estimation, early unready behaviour and XML scene generation. The complete MuJoCo dynamics must be run on your machine. Tests alone do not demonstrate successful respiratory compensation.
