"""Explicit approximate collision proxies for an educational 6R robot.

AABB boxes represent furniture and the torso; spheres represent the head.
Robot links remain straight joint-to-joint capsules. This is not mesh-level
collision detection or a clinical safety model.
"""
from dataclasses import dataclass
import numpy as np
from src.path_planning import CollisionChecker, SphereObstacle, point_segment_distance

@dataclass(frozen=True)
class BoxObstacle:
    center: np.ndarray
    half_extents: np.ndarray
    name: str = 'box'

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        extent = np.asarray(self.half_extents, dtype=float)
        if center.shape != (3,) or extent.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)) or np.any(extent <= 0):
            raise ValueError('Invalid axis-aligned collision box')
        object.__setattr__(self,'center',center)
        object.__setattr__(self,'half_extents',extent)


def point_box_distance(point, box):
    return float(np.linalg.norm(np.maximum(np.abs(point-box.center)-box.half_extents,0)))


def segment_box_distance(a,b,box):
    """Exact distance from finite segment to 3D axis-aligned box.

    Squared distance along a segment is piecewise quadratic. Its breakpoints
    occur where segment coordinates enter/leave the box slabs.
    """
    a=np.asarray(a,float)-box.center
    b=np.asarray(b,float)-box.center
    d=b-a
    h=box.half_extents
    breaks=[0.,1.]
    for i in range(3):
        if abs(d[i])>1e-14:
            for side in (-h[i],h[i]):
                t=(side-a[i])/d[i]
                if 0<t<1: breaks.append(float(t))
    breaks=sorted(set(breaks))
    best=float('inf')
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid=(lo+hi)/2
        xmid=a+mid*d
        active=np.abs(xmid)>h
        if not np.any(active):
            return 0.
        bound=np.where(xmid>=0,h,-h)
        aa=a[active]-bound[active]
        dd=d[active]
        denom=float(dd@dd)
        t=float(np.clip(-float(aa@dd)/denom,lo,hi)) if denom>1e-20 else lo
        x=a+t*d
        dist=float(np.linalg.norm(np.maximum(np.abs(x)-h,0)))
        best=min(best,dist)
    return best if np.isfinite(best) else point_box_distance(a+box.center,box)


class DetailedCollisionChecker(CollisionChecker):
    """Checks joint-limits, sphere/box proxies and straight robot link capsules."""
    def __init__(self,kinematics,obstacles=(),link_radius=0.012,clearance=0.012,interpolation_step=0.035):
        super().__init__(kinematics,obstacles,link_radius,clearance,interpolation_step)

    def check(self,q):
        self.checks+=1
        q=np.asarray(q,dtype=float)
        bounds=self.kinematics.joint_limits
        if q.shape!=(6,) or not np.all(np.isfinite(q)) or np.any(q<bounds[:,0]-1e-10) or np.any(q>bounds[:,1]+1e-10):
            return False,'joint_limits',float('-inf')
        tip,_,pivots=self.kinematics._chain_state(q)
        points=np.vstack([pivots.T,tip])
        required=self.link_radius+self.clearance
        minimum=float('inf')
        for obstacle in self.obstacles:
            for a,b in zip(points[:-1],points[1:]):
                if isinstance(obstacle,SphereObstacle):
                    dist=point_segment_distance(obstacle.center,a,b)-obstacle.radius
                elif isinstance(obstacle,BoxObstacle):
                    dist=segment_box_distance(a,b,obstacle)
                else:
                    raise TypeError('Unsupported obstacle type')
                minimum=min(minimum,dist-required)
                if dist<=required:
                    return False,obstacle.name,minimum
        return True,'',minimum

    def valid(self,q):
        return self.check(q)[0]
