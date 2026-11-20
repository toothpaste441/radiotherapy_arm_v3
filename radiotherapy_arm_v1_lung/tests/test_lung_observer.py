import numpy as np
import pytest
from src.lung_observer import (ChestMarkerObserver,LungMotionCalibration,
                               external_marker_height,synthetic_tumour,phase_displacement)
from src.lung_scene import create_lung_scene


def test_periodic_synthetic_motion():
    cal=LungMotionCalibration()
    origin=np.array([1.,-2.,1.])
    assert np.allclose(synthetic_tumour(0,origin,cal),origin)
    assert np.allclose(synthetic_tumour(4.,origin,cal),origin,atol=1e-12)
    assert not np.allclose(synthetic_tumour(1.,origin,cal),origin)


def test_observer_uses_noisy_external_measurements_only():
    cal=LungMotionCalibration()
    anchor=np.array([1.,-2.,1.])
    ob=ChestMarkerObserver(anchor,cal)
    rng=np.random.default_rng(7)
    for t in np.arange(0,8,.02):
        y=external_marker_height(t,cal)+rng.normal(0,.0003)
        ob.update(t,y)
    assert ob.ready
    assert abs(ob.period_estimate-4.)<=.1
    errs=[np.linalg.norm(ob.estimate(t)-synthetic_tumour(t,anchor,cal))
          for t in np.arange(8,12,.05)]
    assert np.sqrt(np.mean(np.square(errs)))<.0025


def test_missing_observation_holds_anchor():
    anchor=np.array([0.5,0.5,0.5]); ob=ChestMarkerObserver(anchor,LungMotionCalibration())
    assert np.array_equal(ob.estimate(20.),anchor)
    ob.update(0,0)
    with pytest.raises(ValueError):ob.update(0,0)


def test_lung_scene_moves_lesion_into_torso(tmp_path):
    source=__import__('pathlib').Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    tool=np.array([1.,-2.,1.]);direction=np.array([1.,0.,0.])
    scene=create_lung_scene(source,tmp_path/'lung_patient_scene.xml',tool,direction)
    assert np.allclose(scene.tumour_anchor,tool+.26*direction)
    assert np.linalg.norm(scene.torso_center-scene.tumour_anchor)<.03
    assert any('torso' in x.name for x in scene.obstacles)
    import xml.etree.ElementTree as ET
    root=ET.parse(scene.xml_path)
    assert root.find('.//body[@name="chest_observation_marker"]') is not None

def test_arm_spread_moves_visual_geometry_and_collision_proxy_consistently(tmp_path):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from src.robust_collision import BoxObstacle
    xml=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    tool=np.array([1.,-2.,1.])
    axis=np.array([1.,0.,0.])
    a=create_lung_scene(xml,tmp_path/'a.xml',tool,axis,arm_spread_m=(0.,0.))
    b=create_lung_scene(xml,tmp_path/'b.xml',tool,axis,arm_spread_m=(0.12,0.04))
    def geompos(path,name):
        el=ET.parse(path).getroot().find(f'.//geom[@name="{name}"]')
        return np.fromstring(el.get('pos'),sep=' ')
    assert np.isclose(geompos(b.xml_path,'patient_arm_0')[0]-
                      geompos(a.xml_path,'patient_arm_0')[0],-0.12)
    assert np.isclose(geompos(b.xml_path,'patient_arm_1')[0]-
                      geompos(a.xml_path,'patient_arm_1')[0],0.04)
    for side,shift in [('left',-0.12),('right',0.04)]:
        o1=next(x for x in a.obstacles if x.name==f'{side}_arm_box')
        o2=next(x for x in b.obstacles if x.name==f'{side}_arm_box')
        assert isinstance(o1,BoxObstacle) and isinstance(o2,BoxObstacle)
        assert np.isclose(o2.center[0]-o1.center[0],shift)
    assert np.allclose(a.tumour_anchor,b.tumour_anchor)


def test_invalid_arm_offsets_rejected(tmp_path):
    from pathlib import Path
    xml=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    with pytest.raises(ValueError):
        create_lung_scene(xml,tmp_path/'bad.xml',[1,2,3],[1,0,0],
                          arm_spread_m=(-0.01,0))


def test_relocating_robot_base_does_not_move_patient_or_target(tmp_path):
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from src.lung_scene import relocate_robot_base
    source=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    scene=create_lung_scene(source,tmp_path/'base_moved.xml',
                            np.array([1.,-2.,1.]),np.array([1.,0.,0.]))
    def xyz(name,tag='body'):
        root=ET.parse(scene.xml_path).getroot()
        found=root.find(f'.//{tag}[@name="{name}"]')
        return np.fromstring(found.get('pos'),sep=' ')
    base_before=xyz('robot_base')
    table_before=xyz('patient_table_top','geom')
    marker_before=xyz('chest_observation_marker')
    tumour_before=scene.tumour_anchor.copy()
    moved=relocate_robot_base(scene,(0.35,-0.25,0.0))
    assert np.allclose(xyz('robot_base'),base_before+[.35,-.25,0.])
    assert np.allclose(xyz('patient_table_top','geom'),table_before)
    assert np.allclose(xyz('chest_observation_marker'),marker_before)
    assert np.allclose(moved.tumour_anchor,tumour_before)
    assert moved.base_offset_world_m == (0.35,-0.25,0.0)
    assert np.allclose(scene.obstacles[0].center,moved.obstacles[0].center)


def test_robot_base_relocation_rejects_vertical_and_invalid_offsets(tmp_path):
    from pathlib import Path
    from src.lung_scene import relocate_robot_base
    source=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    scene=create_lung_scene(source,tmp_path/'scene.xml',[1,-2,1],[1,0,0])
    for delta in ((0,0,.1),(float('nan'),0,0),(0,.2),(.1,.2,float('inf'))):
        with pytest.raises(ValueError):
            relocate_robot_base(scene,delta)
