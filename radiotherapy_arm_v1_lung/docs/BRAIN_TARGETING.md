# Synthetic brain-tumour targeting scene (education only)

This extension adds a **schematic supine patient, table, pillow, head, and a small
red synthetic tumour** to the existing CyberKnife-style six-joint MuJoCo robot.
It does **not** read CT/MRI, model a real patient or anatomy, compute delivered
radiation dose, estimate clinical accuracy, or make a treatment recommendation.

## Start

From the extracted project root (the folder containing `robot_simulation.py`):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_brain_scene.py tests/test_planning.py
python robot_simulation.py --brain-view
```

Run without a viewer with `python robot_simulation.py --brain`.
The original `--plan`, `--plan-view`, `--view`, and five experiments are unchanged.

The program writes `brain_patient_scene.xml` **in the project root**, alongside
`cyberknife_6dof.xml`, so all relative STL mesh paths resolve. It does not
modify the original XML.

## What you should see

- Grey-and-blue patient table with a simplified human figure lying supine.
- Translucent head, with a small red dot *inside* marking a synthetic tumour.
- Yellow dots showing a **virtual ray along the actual tool-local +X axis**.
- An RRT-Connect planned approach from the original home angles to a chosen
  example treatment joint configuration; a quintic trajectory is sent to the
  PD-controlled MuJoCo motors.
- A stationary hold at the final configuration. This is **brain targeting**, not
  the old breathing/lung tumour motion scenario.

The demonstration places a fake brain tumour 26 cm along the *goal configuration*
end-effector's local +X axis. It is intentionally constructed to be reachable
in a geometric sense. It is **not a target registered from MRI** and not an
independently identified anatomical brain lesion.

## Geometric metrics

- `tool_aimpoint_error_mm`: how far the tool is from its intended standoff pose.
- `beam_ray_miss_mm`: shortest perpendicular separation of target from the ray
  starting at the tool with direction equal to the current tool-local +X axis.
- `tumour_ahead_of_beam`: whether the target lies forward along the ray.
- `tool_reference_error_mm`: actual tool versus the current commanded tool
  position, measured throughout the approach and hold.

The beam is a **visual guide**, not an irradiation model. The existing IK solver
controls **tool position only**, not tool orientation. As a result the virtual
ray can miss the target even if the endpoint is nearby. **Report those errors**
accurately rather than describing the demonstration as successful radiotherapy.

## Generated outputs

```
brain_patient_scene.xml                  # generated MuJoCo scene
results/brain_targeting/
  brain_targeting_samples.csv           # time-series motion and targeting metrics
  brain_targeting_summary.json          # quantitative summary and caveats
  brain_planned_waypoints.csv           # NumPy RRT joint-space plan
  brain_targeting.png                   # tracking / aiming error and joint plots
```

## Limits and next improvements

- The visual patient/table are **collision-disabled in MuJoCo** to avoid pretending
  that these simplistic primitives create a valid patient safety model.
- The planner instead checks conservative sphere proxies around the head,
  upper body, and table against approximate link capsules. It does not verify
  full STL mesh collisions, self-collision, swept collisions, a head-fixation
  device, or dynamic clearance during actual motor execution.
- Its direct home-to-goal configuration path may be valid. RRT-Connect then
  returns the direct path without expanding any trees; this is expected.
- Position IK is not equivalent to radiotherapy aiming. A future *full pose
  controller* should constrain beam axis, source-to-target distance, and
  virtual collision-free approach while verifying actuator torques.
- Wrist motors in prior experiments were often saturated. Inspect the new
  torque columns and the warning if the measured aimpoint is not reached.
- This does not simulate image guidance, treatment planning, dosimetry,
  motion prediction, registration, or medical-device safety constraints.
