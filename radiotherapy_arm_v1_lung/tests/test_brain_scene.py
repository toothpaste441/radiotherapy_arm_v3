"""Patient/table scene tests that do NOT require MuJoCo."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from src.brain_scene import create_brain_scene


def test_brain_target_is_on_reference_beam_axis(tmp_path):
    original=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    axis=np.array([0.8,0.4,0.4])
    tip=np.array([1.1,-1.8,0.95])
    scene=create_brain_scene(original,tmp_path/'brain_patient_scene.xml',tip,axis)
    beam=(scene.brain_target-tip)
    assert np.isclose(np.linalg.norm(beam),0.26,rtol=0,atol=1e-9)
    assert np.allclose(beam/np.linalg.norm(beam),axis/np.linalg.norm(axis))
    assert np.linalg.norm(scene.brain_target-scene.head_center)<0.12
    assert len(scene.beam_names)==15


def test_scene_xml_contains_patient_and_table_without_editing_source(tmp_path):
    original=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    before=original.read_bytes()
    scene=create_brain_scene(original,tmp_path/'brain_patient_scene.xml',[1.,-1.,1.])
    xml=ET.parse(scene.xml_path).getroot()
    for name in ['patient_table_top','patient_table_pad','patient_head',
                 'patient_torso','patient_pillow','virtual_beam_dot_00_geom']:
        assert xml.find(f'.//geom[@name="{name}"]') is not None
    for name in ['patient_head','patient_table_top']:
        geom=xml.find(f'.//geom[@name="{name}"]')
        assert geom.attrib['contype']=='0'
        assert geom.attrib['conaffinity']=='0'
    assert original.read_bytes()==before


def test_bad_target_axis_rejected(tmp_path):
    original=Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml'
    try:
        create_brain_scene(original,tmp_path/'scene.xml',[0,0,0],beam_axis_world=[0,0,0])
    except ValueError:
        return
    raise AssertionError('Zero direction was not rejected')
