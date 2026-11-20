import trimesh
from pathlib import Path
import numpy as np

ASSETS = Path("assets")
OUT = ASSETS / "robot_6dof"
OUT.mkdir(exist_ok=True)

# Original CAD assembly coordinates, in mm
pivots = {
    "J1": np.array([828.437, 842.635, 1432.980]),
    "J2": np.array([670.958, 1062.635, 1431.879]),
    "J3": np.array([785.635, 1525.444, 1629.792]),
    "J4": np.array([827.256, 1672.655, 2190.482]),
    "J5": np.array([839.985, 1699.319, 2227.439]),
    "J6": np.array([849.881, 1688.966, 2206.179]),
}

parts = {
    "base_joint": ("robot_assembly - robot_base_v1-1.STL", "J1"),
    "link1": ("robot_assembly - link1-1.STL", "J2"),
    "link2": ("robot_assembly - link2-1.STL", "J3"),
    "wrist_yaw": ("robot_assembly - wrist_yaw-1.STL", "J4"),
    "wrist_pitch": ("robot_assembly - wrist_pitch-1.STL", "J5"),
    "wrist_roll": ("robot_assembly - wrist_roll-1.STL", "J6"),
    "end_effector": ("robot_assembly - end_effector-1.STL", "J6"),
}

for name, (filename, joint) in parts.items():
    mesh = trimesh.load(ASSETS / filename, force="mesh")
    mesh.vertices -= pivots[joint]
    output = OUT / f"{name}.STL"
    mesh.export(output)
    print(f"Created {output}")

print("\nAll pivot-relative meshes generated.")