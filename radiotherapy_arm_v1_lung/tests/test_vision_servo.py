"""Pure NumPy tests for class-based L33-L37 pinhole / eye-to-hand IBVS."""
import numpy as np
from src.vision_servo import (FixedEyeToHandCamera,
    image_based_joint_velocity,beam_endpoint_and_jacobian)


def test_pinhole_marker_breathing_observation():
    origin=np.array([1.35,-1.8,1.1])
    camera=FixedEyeToHandCamera.beside_patient(origin)
    for rise in (0.,.003,.012):
        s=camera.project(origin+np.array([0.,0.,rise]))
        assert abs(camera.recover_vertical_excursion(s,origin)-rise)<1e-12


def test_projected_jacobian_matches_finite_differences():
    cam=FixedEyeToHandCamera.beside_patient(np.array([1.3,-1.8,1.]))
    p=np.array([1.4,-1.65,1.1]);v=np.array([.03,-.02,.04]);eps=1e-6
    fd=(cam.project(p+eps*v)-cam.project(p-eps*v))/(2*eps)
    assert np.allclose(fd,cam.image_jacobian_world(p)@v,atol=1e-6)


class MockBeamKinematics:
    def beam_state(self,q):
        # 6-joint proxy with image-controllable x/z motion; direction fixed.
        q=np.asarray(q)
        pos=np.array([1.3+q[0],-1.8,1.+q[1]])
        direction=np.array([0.,1.,0.])
        Jv=np.zeros((3,6));Jv[0,0]=1.;Jv[2,1]=1.
        Jd=np.zeros((3,6))
        return pos,direction,Jv,Jd


def test_eye_to_hand_damped_visual_servo_reduces_pixel_error():
    camera=FixedEyeToHandCamera.beside_patient(np.array([1.3,-1.8,1.]))
    kin=MockBeamKinematics()
    q=np.zeros(6)
    target=np.array([1.35,-1.54,1.035])
    errors=[]
    for _ in range(40):
        dq,e,Lq,s,des=image_based_joint_velocity(
            camera,kin,q,target,gain_s_inv=1.,max_joint_speed=.20)
        errors.append(np.linalg.norm(e))
        assert Lq.shape==(2,6)
        q=q+.05*dq
    assert errors[-1]<errors[0]*.65


def test_2d_image_jacobian_cannot_constrain_six_dof():
    camera=FixedEyeToHandCamera.beside_patient(np.array([1.3,-1.8,1.]))
    _,_,Lq,_,_=image_based_joint_velocity(camera,MockBeamKinematics(),
                      np.zeros(6),np.array([1.3,-1.54,1.]))
    assert np.linalg.matrix_rank(Lq)==2
