"""Educational *simulated* eye-to-hand image-based visual servoing (AI4000 L33-37).

This module does not analyse camera pixels. A fixed calibrated pinhole camera
projects virtual 3-D markers into synthetic image coordinates, where pixel noise
can be added. No internal tumour ground truth is used by the controller.

L37: s_dot=J_img R_cw J_world q_dot = L_q q_dot;
     q_dot=-gain * L_q^T (L_q L_q^T+lambda^2 I)^-1 (s-s_star).
For this 6R robot the controlled point is the geometric *virtual beam endpoint*
p(q)+standoff*d(q), so J_world=J_v+standoff*J_d. One camera's 2-D feature
cannot resolve depth or 6-DOF pose; retain geometric beam IK and collision
checks as a secondary 3-D constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FixedEyeToHandCamera:
    center_world: np.ndarray
    world_to_camera: np.ndarray
    focal_px: float = 850.0
    principal_px: tuple[float, float] = (480.0, 270.0)

    def __post_init__(self):
        p=np.asarray(self.center_world,float)
        R=np.asarray(self.world_to_camera,float)
        if p.shape!=(3,) or R.shape!=(3,3) or self.focal_px<=0:
            raise ValueError('invalid fixed camera shape or focal length')
        if not np.allclose(R@R.T,np.eye(3),atol=1e-8) or np.linalg.det(R)<.99:
            raise ValueError('camera rotation must be right-handed orthonormal')

    @classmethod
    def beside_patient(cls, marker_origin, *, focal_px=850., distance=1.2):
        """Camera looking along +world Y; its image vertical axis is -world Z."""
        origin=np.asarray(marker_origin,float)
        R=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        center=origin+np.array([0.,-distance,0.2])
        return cls(center,R,focal_px)

    def camera_xyz(self,point_world):
        return self.world_to_camera@(np.asarray(point_world,float)-self.center_world)

    def project(self,point_world):
        x,y,z=self.camera_xyz(point_world)
        if z<=.05:
            raise ValueError('projected feature behind/too close to camera')
        cx,cy=self.principal_px
        return np.array([cx+self.focal_px*x/z,cy+self.focal_px*y/z])

    def image_jacobian_world(self,point_world):
        """Lecture 37 fixed-camera image Jacobian, 2x3, in world coordinates."""
        x,y,z=self.camera_xyz(point_world)
        if z<=.05:
            raise ValueError('invalid feature depth')
        f=self.focal_px
        J=np.array([[f/z,0.,-f*x/z**2],
                    [0.,f/z,-f*y/z**2]])
        return J@self.world_to_camera

    def recover_vertical_excursion(self,measured_pixel,marker_rest):
        """Convert observed image feature displacement to a height using known depth.

        Only valid for the calibrated marker constrained to travel vertically in
        the world frame (depth unchanged in this camera geometry).
        """
        ref=self.project(marker_rest)
        measured=np.asarray(measured_pixel,float)
        z=self.camera_xyz(marker_rest)[2]
        slope=(self.image_jacobian_world(marker_rest)@np.array([0.,0.,1.]))[1]
        if abs(slope)<1e-8:raise ValueError('marker height unobservable')
        return float((measured[1]-ref[1])/slope)


def beam_endpoint_and_jacobian(kinematics,q,standoff_m=.26):
    """Position and translational Jacobian of fixed-length virtual beam point."""
    p,d,Jv,Jd=kinematics.beam_state(np.asarray(q,float))
    return p+standoff_m*d,Jv+standoff_m*Jd


def image_based_joint_velocity(camera,kinematics,q,target_world,
                               *,gain_s_inv=.7,damping_px_per_rad=10.,
                               max_joint_speed=.18,standoff_m=.26):
    """IBVS image error and damped pseudoinverse joint-rate command.

    The image target must be a *predicted* location from the calibrated chest
    surrogate, not the synthetic hidden tumour's ground truth.
    """
    beam_point,Jpoint=beam_endpoint_and_jacobian(kinematics,q,standoff_m)
    s=camera.project(beam_point)
    desired=camera.project(target_world)
    e=s-desired
    Lq=camera.image_jacobian_world(beam_point)@Jpoint  # 2x6
    damp=float(damping_px_per_rad)
    dq=-gain_s_inv*Lq.T@np.linalg.solve(Lq@Lq.T+damp**2*np.eye(2),e)
    speed=np.linalg.norm(dq)
    if speed>max_joint_speed:dq*=max_joint_speed/speed
    return dq,e,Lq,s,desired
