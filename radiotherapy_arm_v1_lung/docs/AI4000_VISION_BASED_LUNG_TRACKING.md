# Class-aligned simulated vision-based lung tracking (AI4000 Lectures 33–37)

This is a **synthetic pinhole-camera / eye-to-hand image-based visual servoing (IBVS)** demonstration, not image recognition on real frames, internal tumour imaging, or clinical motion compensation. A fixed virtual camera beside the patient projects a simulated **external chest marker** into 2-D features. Only these synthetic features, with added noise, are used to estimate breathing. The hidden internal tumour coordinates are used by the simulator to score the experiment, never by the observer or visual controller.

## Run

From the folder containing `robot_simulation.py`, install `requirements.txt`, then:

```powershell
python -m pytest -q
python robot_simulation.py --lung-vision
python robot_simulation.py --lung-vision-view
python robot_simulation.py --lung-vision-record
```

`--lung-vision` compares compensation with a **static-beam baseline** and saves CSV / summary / plots in `results/lung_tracking/`. `--lung-vision-view` shows only the compensated run; run the headless mode for both metrics.

## Exactly which classroom concepts are used

**Lecture 33 – Pinhole camera**: For world point `P`, fixed camera center `c`, world-to-camera rotation `R`, and camera coordinates `(X,Y,Z)=R(P-c)`:

`u = cx + f X/Z`, `v = cy + f Y/Z`.

The camera is fixed beside the chest (eye-to-hand). `camera.project` gives a synthetic image coordinate and pixel noise is added in `observe_external_chest`. The vertical chest displacement is recovered from *observed pixels* under a known-depth, vertical-motion calibration. A camera cannot infer arbitrary 3-D target location from one image point.

**Lecture 34 and Lecture 37 – Image interaction matrix**: Differentiate projection for a moving point and a fixed camera:

`J_img_camera = [[f/Z, 0, -(u-cx)/Z], [0, f/Z, -(v-cy)/Z]]`.

`J_img_world = J_img_camera @ R_world_to_camera`.

**Lecture 13 – Robot Jacobian / tool offset**: The virtual beam endpoint is `B(q)=p_tool(q)+L d(q)`, where `L=0.26 m`. Its Jacobian is

`J_B(q) = J_v(q) + L J_d(q)`.

**Lecture 37 – Eye-to-hand IBVS**: The image-to-joint Jacobian is

`L_q = J_img_world(B(q)) @ J_B(q)` (2×6).

Define `e=s_beam-s_tumour_predicted` and command the damped visual correction

`qdot_IBVS = -gamma * L_q.T @ inv(L_q @ L_q.T + lambda**2 * I) @ e`.

This gives a *bounded outer-loop image-feedback correction* added to the joint target returned by the existing 3-D beam-axis DLS IK. The lower-level computed-torque PD controller (AI4000 Lecture 31) is unchanged. The `--lung-vision` mode uses a gain of 0.7/s and damping 10 px/rad, with maximum norm 0.18 rad/s. The image-feedback joint correction is limited to one 20 ms update. The geometric IK still constrains the beam-source position, depth, and safe configuration; the one-image-feature IBVS problem alone is rank deficient.

**Respiratory estimation**: The past noisy external image coordinates are converted to a height signal, and an LS sinusoidal phase/frequency observer (`lung_observer.py`) estimates respiratory phase. A *pre-assumed chest-to-tumour surrogate calibration* maps that phase to a predicted internal tumour point. This is not taught as the IBVS interaction-matrix derivation; it is an explicit synthetic estimation assumption layered above it. No internal tumour ground truth is leaked to the estimator.

## Files

- `src/vision_servo.py`: fixed camera, projection, image Jacobian, beam-point Jacobian, damped IBVS joint-rate correction.
- `src/lung_observer.py`: sinusoidal external marker observer / assumed 3-D tumour correlation.
- `src/lung_tracking_demo.py`: integrates synthetic image measurement, observer, IBVS, beam IK and computed-torque PD.
- `src/lung_scene.py`: creates patient, chest marker and decorative fixed camera housing.
- `tests/test_vision_servo.py`: projection, observation calibration, finite-difference Jacobian and error-convergence tests.

## How to read the results

The compensated CSV logs `camera_marker_u_px`, `camera_marker_v_px`, `image_error_px`, `ibvs_joint_speed_cmd_rad_s`, `estimation_error_mm`, `true_beam_miss_mm` and the commanded/actual joint angles. `image_error_px` is defined against the **estimated** target image location; `true_beam_miss_mm` is a separate evaluation using the hidden synthetic tumour. The static-beam baseline is an ablation of compensation, not an independently validated real-world benchmark.

## Limitations

The camera is a mathematical pinhole projection, not a rendered or real vision processing pipeline. The geometric virtual beam endpoint is a projected feature, not a visible radiotherapy beam. A single camera feature cannot recover tumour depth; camera calibration and the chest-to-internal tumour correlation are assumed. No radiation, no clinical imaging, no dosimetry, no clinical accuracy claim, and only approximate obstacle collision checks. Full MuJoCo physics runs must be verified locally (including settling and image error) before claiming visual compensation improves tracking.
