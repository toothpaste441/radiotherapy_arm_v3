"""Offline verification of beam pointing against the exact transforms in project XML."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from src.kinematics import _axis_angle, _quat_to_matrix
from src.beam_ik import candidate_beam_directions, solve_beam_ik

XML = ET.parse(Path(__file__).resolve().parents[1]/'cyberknife_6dof.xml').getroot()

class OfflineChain:
    """Tiny independent XML geometry evaluator; no MuJoCo runtime required."""
    joint_limits = np.array([[-1.5708,1.5708],[-1.5708,1.5708],[-2.0944,2.0944],
                             [-3.14159,3.14159],[-1.5708,1.5708],[-3.14159,3.14159]])

    def beam_state(self,q):
        rotation = np.eye(3); origin = np.zeros(3)
        axes=[]; pivots=[]
        body=XML.find('worldbody/body[@name="robot_base"]')
        joint_index=0
        while body is not None:
            rb=_quat_to_matrix(np.fromstring(body.get('quat','1 0 0 0'),sep=' '))
            pb=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            origin=origin+rotation@pb
            rotation=rotation@rb
            joint=body.find('joint')
            if joint is not None:
                axis=np.fromstring(joint.get('axis'),sep=' ')
                pivot=np.fromstring(joint.get('pos','0 0 0'),sep=' ')
                pivots.append(origin+rotation@pivot)
                axes.append(rotation@(axis/np.linalg.norm(axis)))
                origin=origin+rotation@pivot
                rotation=rotation@_axis_angle(axis,q[joint_index])
                origin=origin-rotation@pivot
                joint_index+=1
            body=body.find('body')
        geom=XML.find('.//geom[@name="tool_tip"]')
        p=origin+rotation@np.fromstring(geom.get('pos'),sep=' ')
        d=rotation[:,0]
        a=np.array(axes)
        pv=np.array(pivots)
        return p,d,np.cross(a,p[None,:]-pv).T,np.cross(a,np.tile(d,(6,1))).T


def test_beam_direction_jacobian_finite_difference():
    kin=OfflineChain()
    q=np.array([.55,-.53,.52,.30,-.12,.18])
    _,_,_,jdir=kin.beam_state(q)
    delta=1e-6
    numerical=np.column_stack([(kin.beam_state(q+np.eye(6)[i]*delta)[1]-
                                kin.beam_state(q-np.eye(6)[i]*delta)[1])/(2*delta)
                               for i in range(6)])
    assert np.allclose(jdir,numerical,atol=1e-6)


def test_multiple_synthetic_aim_nodes_reachable():
    kin=OfflineChain()
    q0=np.array([.55,-.53,.52,.30,-.12,.18])
    p,d,_,_=kin.beam_state(q0)
    tumour=p+.26*d
    for _,_,axis in candidate_beam_directions(d)[:7]:
        result=solve_beam_ik(kin,tumour-.26*axis,axis,q0)
        assert result.converged, (result.position_error_m,result.pointing_error_deg)
        assert result.position_error_m<.001
        assert result.pointing_error_deg<.3


def test_invalid_direction_rejected():
    kin=OfflineChain()
    try:
        solve_beam_ik(kin,np.zeros(3),np.zeros(3),np.zeros(6))
        assert False
    except ValueError:
        pass
