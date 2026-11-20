"""Standalone geometric / controller regression checks (no MuJoCo required)."""
import numpy as np
from src.robust_collision import BoxObstacle, segment_box_distance, DetailedCollisionChecker
from src.stable_joint_control import SaturatedPD, ArrivalGate
from src.cyberknife_motion_demo import beam_metrics


def test_segment_box_collision_and_clearance():
    box=BoxObstacle(np.array([0.,0.,0.]),np.array([1.,2.,1.]),'table')
    assert segment_box_distance(np.array([-3.,0.,0.]),np.array([3.,0.,0.]),box)==0.
    assert np.isclose(segment_box_distance(np.array([-3.,4.,0.]),np.array([3.,4.,0.]),box),2.)
    assert np.isclose(segment_box_distance(np.array([3.,3.,0.]),np.array([3.,3.,0.]),box),np.sqrt(5))


def test_collision_box_catches_link_that_sphere_might_miss():
    class Kin:
        joint_limits=np.tile([-1.,1.],(6,1))
        def _chain_state(self,q):
            pivots=np.array([[0,1,2,3,4,5],[0.,0.,0.,0.,0.,0.],[2.,2.,2.,2.,2.,2.]])
            return np.array([6.,0.,2.]),np.zeros((3,6)),pivots
    box=BoxObstacle(np.array([3.,0.,2.]),np.array([.2,.2,.2]),'patient_table')
    checker=DetailedCollisionChecker(Kin(),(box,),link_radius=.02,clearance=.01)
    assert not checker.valid(np.zeros(6))
    assert checker.check(np.zeros(6))[1]=='patient_table'


def test_arrival_requires_consecutive_samples():
    g=ArrivalGate(required_samples=3,joint_tol=.02,speed_tol=.08)
    q=np.zeros(6)
    assert not g.update(q,np.zeros(6),q)
    assert not g.update(q,np.zeros(6),q)
    assert not g.update(q,np.ones(6),q)
    assert not g.update(q,np.zeros(6),q)
    assert not g.update(q,np.zeros(6),q)
    assert g.update(q,np.zeros(6),q)


def test_controller_clips_and_reports_saturation():
    c=SaturatedPD([10]*6,[2]*6,[3]*6)
    torque=c.calculate(np.zeros(6),np.zeros(6),np.ones(6),np.zeros(6),np.zeros(6))
    assert np.allclose(torque,3.)
    assert c.diag.saturated_samples==1


def test_beam_metrics_distinguishes_axis_miss_from_position():
    miss,angle,front=beam_metrics(np.array([0.,0.,0.]),np.array([1.,0.,0.]),np.array([1.,.2,0.]))
    assert np.isclose(miss,.2)
    assert 0<angle<90 and front


def test_inertia_scaled_computed_torque_controller():
    from src.stable_joint_control import ComputedTorquePD
    c=ComputedTorquePD([100]*6,natural_frequency=3.,damping_ratio=1.)
    M=np.diag([2.,3.,4.,.001,.002,.003])
    out=c.calculate(np.zeros(6),np.zeros(6),np.ones(6),np.zeros(6),np.zeros(6),M)
    assert np.allclose(out,9*np.diag(M))
    out2=c.calculate(np.zeros(6),np.ones(6),np.zeros(6),np.zeros(6),np.zeros(6),M)
    assert np.allclose(out2,-6*np.diag(M))
