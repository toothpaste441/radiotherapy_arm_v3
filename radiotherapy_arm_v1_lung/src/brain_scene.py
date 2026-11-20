"""Build an illustrative patient/table/head scene around a *reachable* robot tool pose.

Visualization only: primitive geoms, not patient anatomy or collision-safe CAD.
Units are metres; the target is an artificial point inside the head volume.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from src.path_planning import SphereObstacle
from src.robust_collision import BoxObstacle


def _xyz(x):
    return ' '.join(f'{float(a):.8f}' for a in x)


@dataclass(frozen=True)
class BrainScene:
    xml_path: Path
    brain_target: np.ndarray
    head_center: np.ndarray
    tool_aim_point: np.ndarray
    obstacles: tuple[SphereObstacle, ...]
    beam_names: tuple[str, ...]
    beam_axis_world_at_goal: np.ndarray


def create_brain_scene(source_xml: Path, output_xml: Path, tool_aim_point,
                       beam_axis_world=(1.,0.,0.), beam_dots: int = 15) -> BrainScene:
    """Create a deterministic scene with table, supine patient and synthetic tumour.

    Target is 26 cm along the chosen tool-local +X beam axis at the goal pose.
    It is a geometrical aim demonstration, NOT dosimetry, radiotherapy
    planning, beam collimation or a model of a clinical CyberKnife beam.
    """
    tool = np.asarray(tool_aim_point, dtype=float)
    if tool.shape != (3,) or not np.all(np.isfinite(tool)):
        raise ValueError('tool_aim_point must be a finite 3-vector')
    if beam_dots < 2:
        raise ValueError('beam_dots must be >= 2')
    axis=np.asarray(beam_axis_world,dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)<1e-9:
        raise ValueError('beam_axis_world must be a finite nonzero 3-vector')
    axis=axis/np.linalg.norm(axis)
    tumor = tool + 0.26*axis
    head = tumor - 0.010*axis
    # Feet extend in +Y; the patient is laid out along the table's long axis.
    table = np.array([head[0], head[1] + 0.93, 0.695])
    tree = ET.parse(source_xml)
    world = tree.getroot().find('worldbody')
    if world is None:
        raise ValueError('Source model has no worldbody')

    def geom(name, shape, pos, size, rgba, **kw):
        kwargs = dict(name=name, type=shape, pos=_xyz(pos), size=_xyz(size),
                      rgba=rgba, contype='0', conaffinity='0')
        kwargs.update(kw)
        return ET.SubElement(world, 'geom', **kwargs)

    # Table: top mattress and pedestal with sturdy feet.
    geom('patient_table_top','box', table, [0.38,1.36,0.06], '0.72 0.78 0.83 1')
    geom('patient_table_pad','box',table+[0,0,0.082],[0.34,1.31,0.025],'0.18 0.42 0.61 1')
    geom('patient_table_pedestal','box',table+[0,0,-0.32],[0.16,0.37,0.265], '0.43 0.48 0.53 1')
    for i, yy in enumerate((-0.46,0.46)):
        geom(f'table_foot_{i}','box',table+[0,yy,-0.60],[0.29,0.15,0.035], '0.35 0.39 0.44 1')

    skin='0.82 0.63 0.51 1'
    gown='0.32 0.68 0.71 1'
    # Supine mannequin (deliberately schematic, not an anatomical model).
    neck = head + [0,0.174,-0.055]
    geom('patient_neck','capsule',neck,[0.055,0.072],skin,quat='0.70710678 0.70710678 0 0')
    geom('patient_head','ellipsoid',head,[0.118,0.142,0.115], '0.85 0.68 0.55 0.73')
    geom('patient_hair','ellipsoid',head+[0,0,0.033],[0.120,0.144,0.086],'0.23 0.20 0.18 0.48')
    # Small facial features help clarify supine orientation in the viewer.
    for i, xoff in enumerate((-0.038,0.038)):
        geom(f'patient_eye_{i}','sphere',head+[xoff,-0.055,0.105],[0.010],'0.18 0.16 0.15 0.9')
    geom('patient_nose','sphere',head+[0,-0.095,0.109],[0.016],skin)
    geom('patient_pillow','box',head+[0,0,-0.13],[0.17,0.17,0.03],'0.86 0.87 0.91 1')
    torso = head + np.array([0.,0.53,-0.085])
    geom('patient_torso','box',torso,[0.235,0.35,0.115],gown)
    geom('patient_pelvis','ellipsoid',head+[0,1.01,-0.105],[0.21,0.19,0.115],gown)
    for i, xoff in enumerate((-0.135,0.135)):
        geom(f'patient_arm_{i}','capsule',head+[xoff*2,0.60,-0.115], [0.062,0.30],skin,
             quat='0.70710678 0.70710678 0 0')
        geom(f'patient_leg_{i}','capsule',head+[xoff,1.48,-0.105], [0.083,0.39],gown,
             quat='0.70710678 0.70710678 0 0')

    # A 7-mm red virtual lesion inside the head, tracked using the original
    # tumour_target mocap body in cyberknife_6dof.xml.
    # Its position is set by data.mocap_pos in the simulation.
    target_geom = tree.getroot().find('.//geom[@name="tumor_marker"]')
    if target_geom is not None:
        target_geom.set('size','0.008')
        target_geom.set('rgba','0.95 0.08 0.06 1')

    dots=[]
    for i in range(beam_dots):
        name=f'virtual_beam_dot_{i:02d}'
        body=ET.SubElement(world,'body',name=name,mocap='true',pos=_xyz(tool))
        ET.SubElement(body,'geom',name=f'{name}_geom',type='sphere',size='0.004',
                      rgba='1 0.84 0.12 0.85',contype='0',conaffinity='0')
        dots.append(name)

    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True,exist_ok=True)
    tree.write(output_xml,encoding='unicode',xml_declaration=True)
    # Conservative *approximations* for the joint-space planner, not full-mesh
    # or accurate patient-table collision detection.
    obstacles=(
        SphereObstacle(head.copy(),0.142,'head_proxy'),
        SphereObstacle(neck.copy(),0.125,'neck_proxy'),
        SphereObstacle(head+np.array([0,-0.095,0.109]),0.016,'nose_proxy'),
        BoxObstacle(head+np.array([0,0,-0.13]),np.array([0.17,0.17,0.03]),'pillow_box'),
        BoxObstacle(torso.copy(),np.array([0.235,0.35,0.115]),'torso_box'),
        BoxObstacle(table.copy(),np.array([0.38,1.36,0.06]),'table_top_box'),
        BoxObstacle(table+np.array([0,0,0.082]),np.array([0.34,1.31,0.025]),'mattress_box'),
        BoxObstacle(table+[0,0,-0.32],np.array([0.16,0.37,0.265]),'table_pedestal_box'),
        BoxObstacle(table+np.array([0,-0.46,-0.60]),np.array([0.29,0.15,0.035]),'table_foot_a'),
        BoxObstacle(table+np.array([0,0.46,-0.60]),np.array([0.29,0.15,0.035]),'table_foot_b'),
        BoxObstacle(head+np.array([0,1.01,-0.105]),np.array([0.21,0.19,0.115]),'pelvis_box'),
        BoxObstacle(head+np.array([-0.27,0.60,-0.115]),np.array([0.062,0.35,0.062]),'left_arm_box'),
        BoxObstacle(head+np.array([0.27,0.60,-0.115]),np.array([0.062,0.35,0.062]),'right_arm_box'),
        BoxObstacle(head+np.array([-0.135,1.48,-0.105]),np.array([0.083,0.47,0.083]),'left_leg_box'),
        BoxObstacle(head+np.array([0.135,1.48,-0.105]),np.array([0.083,0.47,0.083]),'right_leg_box'),
    )
    return BrainScene(output_xml,tumor,head,tool,obstacles,tuple(dots),axis)


def set_beam_markers(model, data, names, tool_position, beam_direction, length=0.26):
    """Draw a ray along the ACTUAL tool +X axis (not snapped to the tumour)."""
    import mujoco
    start=np.asarray(tool_position,float)
    axis=np.asarray(beam_direction,float)
    axis=axis/np.linalg.norm(axis)
    end=start+float(length)*axis
    for i,name in enumerate(names):
        body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        mocap_id=int(model.body_mocapid[body_id])
        data.mocap_pos[mocap_id]=start+(i+1)/(len(names)+1)*(end-start)
