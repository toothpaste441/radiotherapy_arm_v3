# helper Python file to prepare STL meshes for joint-relative inspection and analysis
import trimesh
from pathlib import Path

STL_DIR = Path("assets")
OUTPUT_DIR = STL_DIR / "joint1_local"
OUTPUT_DIR.mkdir(exist_ok=True)

# Estimated Joint 1 centre in original CAD coordinates (mm)
pivot = [828.437, 842.635, 1432.980]

moving_parts = [
    "robot_assembly - robot_base_v1-1.STL",
    "robot_assembly - link1-1.STL",
    "robot_assembly - link2-1.STL",
    "robot_assembly - wrist_yaw-1.STL",
    "robot_assembly - wrist_pitch-1.STL",
    "robot_assembly - wrist_roll-1.STL",
    "robot_assembly - end_effector-1.STL",
]

for filename in moving_parts:
    mesh = trimesh.load(STL_DIR / filename, force="mesh")

    # Express vertices relative to Joint 1
    mesh.vertices -= pivot

    output_name = filename.replace(".STL", "_J1.STL")
    mesh.export(OUTPUT_DIR / output_name)

    print("Created:", output_name)

print("Joint 1 mesh preparation complete.")
