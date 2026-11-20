
from pathlib import Path
import trimesh

PROJECT_DIR = Path("./")
STL_DIR = PROJECT_DIR / "assets" / "robot_6dof"

scene = trimesh.Scene()

for file_path in sorted(STL_DIR.glob("*.STL")):

    mesh = trimesh.load(
        file_path,
        force="mesh",
        process=False
    )

    print(
        file_path.name,
        "Bounds:", mesh.bounds,
        "Centroid:", mesh.centroid
    )

    scene.add_geometry(
        mesh,
        geom_name=file_path.stem,
        node_name=file_path.stem
    )

# Add world coordinate axes
axes = trimesh.creation.axis(
    origin_size=0.05,
    axis_length=0.5
)

scene.add_geometry(axes, geom_name="world_axes")

# Save an interactive 3D model
output = PROJECT_DIR / "geometry_reports" / "all_parts.glb"
scene.export(output)

print("\nExported:", output)

# Open interactive viewer
scene.show()
