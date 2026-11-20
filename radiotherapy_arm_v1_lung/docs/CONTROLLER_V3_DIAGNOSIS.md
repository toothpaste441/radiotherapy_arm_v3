# CyberKnife-style motion V3: passive-force control correction

This is an **educational, visual-only brain-pointing robot simulator**. It does not model radiotherapy, radiation dose, treatment safety or clinical beam delivery.

## Reproduced problem

On MuJoCo 4.0, the preceding build successfully validated independent FK and beam orientation, but the first planned node failed the alignment/settling gate. Reported results included:

- Final max joint error: **0.29414 rad**
- Final joint speed: **0.00340 rad/s**
- Final beam-ray miss: **56.18 mm**
- Actuator saturation: **0 steps**

This is a **slow convergence / damping mismatch**, not actuator saturation. The old controller calculated `tau = qfrc_bias + M @ u`, even though `damping` is set to 5/2/1 in the six XML joints and MuJoCo applies passive spring/damper forces separately.

## Controller correction

MuJoCo's unconstrained joint-space equation (with no external applied forces) is:

`M(q) qdd + bias(q, qd) = tau_actuator + qfrc_passive`

Therefore the NumPy feedback-linearizing computed-torque command is:

`u = qdd_des + wn**2*(q_des-q) + 2*zeta*wn*(qd_des-qd)`

`tau = clip(bias - qfrc_passive + M @ u, -limits, limits)`

where `wn = 3 rad/s`, `zeta = 1`, and the original actuator limits are preserved. This cancels the modelled passive damping **only in the simulation model**. MuJoCo supplies `M`, `bias`, and `qfrc_passive`; NumPy implements the control law. `mj_forward` refreshes these values before use at each substep.

The change was tested against a standalone damped second-order plant; that is **not the same as running the MuJoCo 6R model**.

## Collision / selection corrections

- Additional simplified collision proxies now cover the mattress, pillow, neck, nose and table feet, in addition to the earlier head, table, torso, pelvis, arms and legs.
- The beam-viewpoint generator now **rejects IK-reachable configurations that violate these proxies** and tests additional candidate directions until the requested number of feasible nodes is found.
- The selected candidate number, rejected candidates and collision reason are written to the summary. The planned trajectories are still sampled and dynamically monitored; simulation stops on a detected proxy violation.
- Collision checks are **approximations** using centreline capsules and simplified primitives. They do not guarantee absence of visible-mesh intersections or self-collision. The patient visuals have MuJoCo physical contact disabled. Do not call this clinically safe or mesh-collision-free.

## Fast Windows verification (project root)

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python robot_simulation.py --cyberknife --cyberknife-nodes 1
python robot_simulation.py --cyberknife
python robot_simulation.py --cyberknife-view
```

Run the commands in order. **Only continue to seven nodes once the first node reports `successful: true`.** If it fails, open `results/cyberknife_motion/multinode_summary.json` and the new `multinode_samples.csv`. The CLI now exits with a nonzero code on an unsuccessful run.

Success criteria per node: a position error at most 10 mm, virtual-beam miss at most 5 mm, max joint speed at most 0.08 rad/s, max joint error at most 0.015 rad, and no approximate proxy collision, sustained for the arrival gate. The 2-second hold must also remain aligned and settled.

Useful diagnostics: `max_abs_passive_force_Nm`, `max_abs_bias_torque_Nm`, `actuator_saturated_steps`, and per-node `final_actual_joints_rad` versus `final_command_joints_rad`. The `--cyberknife-zero-g` option is for comparative debugging only, not the submitted gravity-loaded result.

## Verification completed during authoring

- Python syntax and import validation: completed.
- NumPy-only geometry/planning/controller unit tests: completed; `24 passed`.
- Offline XML-based geometry/planning for seven candidate directions: completed, using the proxy model rather than compiled MuJoCo contacts.
- Actual MuJoCo 4.0 simulation: **not run in the authoring environment** (MuJoCo package cannot be installed due unavailable package network). The new `tests/test_mujoco_controller_integration.py` executes a real gravity-loaded single-position settling test on your Windows environment where MuJoCo 4.0 is installed.

**Do not present the new seven-node movement as successful unless your local MuJoCo run confirms it.** Keep historical original breathing-tracking experiments as previously measured, independently reproducible data.
