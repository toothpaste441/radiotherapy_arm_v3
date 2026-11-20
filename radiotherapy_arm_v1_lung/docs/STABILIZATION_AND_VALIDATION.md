# CyberKnife-inspired simulation: controller and geometry audit

This project is an **educational MuJoCo simulation only**, with synthetic patient geometry and a virtual aiming ray. It does not model a calibrated robot, radiation delivery, patient registration, collision-safety certification, or treatment planning.

## What changed

1. **Controller diagnostics / controlled stop:** `src/stable_joint_control.py` contains a NumPy computed-torque PD controller using the MuJoCo joint inertia matrix and bias-force terms, plus explicit saturation/speed diagnostics. In `src/cyberknife_motion_demo.py`, excessive joint speed, proxy intrusion, nonfinite state, or failure to settle **halts the multi-node run**. We deliberately do **not** silently increase motor torque limits. The natural frequency is 3 rad/s and damping ratio is 1 for a simple nominal second-order error response (before motor saturation); actual stabilization **must** be established on the user's computer; the code does not claim validated stability.
2. **FK and beam direction:** The demonstration independently computes FK using NumPy, then cross-checks tool position and +X beam orientation against MuJoCo `geom_xpos/geom_xmat` at home, reference, and every proposed node. A disagreement terminates execution. The actual beam markers use the physical MuJoCo orientation, not a line drawn to the tumour.
3. **Improved collision proxies:** `src/robust_collision.py` implements exact finite-segment–axis-aligned-box distance and segment–sphere distance. The scene provides boxes for tabletop, pedestal, torso, pelvis, limbs, and a head sphere. The planner and execution monitor both use the same proxies. **This is still not exact STL collision checking**, does not cover robot self-collisions, and cannot guarantee contact-free motion between sampled physics steps. Patient/table visual geoms remain non-colliding in MuJoCo; the proxy monitor stops on detected intrusion but cannot physically prevent it during the same step.
4. **Arrival gate:** The robot must remain within the joint, velocity, aimpoint and ray thresholds for 0.4 simulated seconds before the hold is declared reached; it must then remain aligned through the hold. If not, the run stops at that node and writes a failure reason, rather than proceeding to the next node.
5. **Revalidation:** Offline tests check the IK, beam-direction Jacobian, collision math, arrival state, saturation, and planning. Every generated quintic path is sampled for proxy clearance. Runtime diagnostics report which node failed. Reported 'clearance' is against *only* these educational proxies.

## Run commands (Windows PowerShell, inside the extracted project folder)

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_safety_regressions.py tests/test_beam_ik.py tests/test_brain_scene.py tests/test_planning.py
python robot_simulation.py --cyberknife-view
```

**Debugging mode:** If the gravity-loaded model cannot settle because of insufficient actuator torques, run the intentionally **gravity-free dynamics diagnostic**:

```powershell
python robot_simulation.py --cyberknife-view --cyberknife-zero-g
```

This zero-gravity run is a simplified physics diagnostic, **not a solution to the actual gravity-loaded control problem**. Use it to distinguish gravity/actuator inability from misdefined FK or aiming geometry. Do not report its output as gravity-loaded performance.

## Interpreting the output

`results/cyberknife_motion/multinode_summary.json` now gives `successful`, `failure_reason`, `requested_nodes`, `completed_nodes`, FK checks, controller saturation frequency and maximum speed. Per-sample CSV logs carry desired and actual joint angles, motor torques, geometric ray miss, and proximity to the collision proxies. The diagnostic figure plots targeting error, joint speeds, saturation and joint errors.

- If the simulation stops due to `Overspeed`, the controller/dynamic model is unstable or actuator-limited. Inspect `actuator_saturated_steps`, `max_abs_bias_torque_Nm` and sampled torques.
- If it stops due to `did not settle/align`, the desired kinematic target was not achieved under simulated dynamics. Do not label the viewpoint successful.
- If it stops due to proxy collision, adjust start/goal/viewpoints or the obstacle representation, then replan.
- If FK/MuJoCo validation fails, check the tool geom offset, axis convention, joint transformations and model loading before tuning control.
- `planning_iterations=0` simply means the direct edge was clear; RRT is still implemented, but the scene does not force the planner to search around an obstacle.

## Important unresolved validation

MuJoCo is not available in the environment where these revised files were packaged. Therefore full physics execution, physical joint settling, gravity bias feasibility, exact CAD collision and seven-node alignment **have not been confirmed**. The new code is intended to make such failures explicit, not to conceal them. The inertia-scaled PD avoids imposing the same fixed torque gains on very different joint inertias, but the true gravity-loaded performance is unknown until a real MuJoCo run.
