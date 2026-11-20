
from pathlib import Path

import trimesh
import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path("./")
STL_DIR = PROJECT_DIR / "assets" /"robot_6dof"
OUTPUT_DIR = PROJECT_DIR / "geometry_reports"

OUTPUT_DIR.mkdir(exist_ok=True)

# Find all STL files
stl_files = sorted(STL_DIR.glob("*.stl"))

if not stl_files:
    raise FileNotFoundError(f"No STL files found in {STL_DIR}")

print(f"Found {len(stl_files)} STL files\n")

results = []

for file_path in stl_files:

    print("=" * 60)
    print(f"FILE: {file_path.name}")

    # Load mesh without automatically repairing it
    mesh = trimesh.load(
        file_path,
        force="mesh",
        process=False
    )

    # Basic geometry
    bounds = mesh.bounds
    dimensions = mesh.extents

    # Surface-based centroid
    centroid = mesh.centroid

    # Connected mesh components
    components = mesh.split(only_watertight=False)

    # Volume and volume-based centre of mass are meaningful
    # only for a valid, watertight, consistently oriented mesh.
    valid_volume = mesh.is_watertight and mesh.is_winding_consistent

    volume = mesh.volume if valid_volume else np.nan

    center_of_mass = (
        mesh.center_mass if valid_volume else [np.nan] * 3
    )

    # Print inspection results
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Dimensions:", dimensions)
    print("Minimum bounds:", bounds[0])
    print("Maximum bounds:", bounds[1])
    print("Surface area:", mesh.area)
    print("Watertight:", mesh.is_watertight)
    print("Consistent winding:", mesh.is_winding_consistent)
    print("Connected components:", len(components))
    print("Volume:", volume)
    print("Surface centroid:", centroid)
    print("Volume centre of mass:", center_of_mass)

    results.append({
        "filename": file_path.name,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "x_extent": dimensions[0],
        "y_extent": dimensions[1],
        "z_extent": dimensions[2],
        "min_x": bounds[0][0],
        "min_y": bounds[0][1],
        "min_z": bounds[0][2],
        "max_x": bounds[1][0],
        "max_y": bounds[1][1],
        "max_z": bounds[1][2],
        "surface_area": mesh.area,
        "watertight": mesh.is_watertight,
        "winding_consistent": mesh.is_winding_consistent,
        "components": len(components),
        "volume": volume,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2]
    })

# Save results
report = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "geometry_report.csv"
report.to_csv(csv_path, index=False)

print("\n" + "=" * 60)
print("GEOMETRY INSPECTION COMPLETED")
print("Files inspected:", len(stl_files))
print("Report saved:", csv_path)

print("\nDimensions summary:")
print(report[["filename", "x_extent", "y_extent", "z_extent"]])
# import trimesh
# from pathlib import Path

# for file in Path("D:/AI_Robotics/Cyberknife").glob("*.STL"):
#     mesh = trimesh.load(file, force="mesh", process=False)

#     print(f"\n{file.name}")
#     print("Bounds:\n", mesh.bounds)
#     print("Centre:", mesh.bounding_box.centroid)
#     print("Extents:", mesh.extents)