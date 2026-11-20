"""Simulation controller with saturation diagnostics and fail-closed arrival gates.

Not a safety-certified controller. Do not use for physical robotics/therapy.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class JointDiagnostics:
    max_speed: float=0.
    saturated_samples: int=0
    max_abs_bias: float=0.
    samples: int=0
    max_abs_error: float=0.
    max_abs_passive: float=0.
    max_uncompensated_passive: float=0.

class SaturatedPD:
    def __init__(self,kp,kd,limits, *, bias_compensation=True):
        self.kp=np.asarray(kp,float)
        self.kd=np.asarray(kd,float)
        self.limits=np.asarray(limits,float)
        self.bias_compensation=bias_compensation
        if any(x.shape!=(6,) for x in (self.kp,self.kd,self.limits)) or np.any(self.limits<=0):
            raise ValueError('Expected six positive torque limits and six gains')
        self.diag=JointDiagnostics()

    def calculate(self,q,vel,q_des,vel_des,bias):
        q=np.asarray(q,float); vel=np.asarray(vel,float)
        bias=np.asarray(bias,float)
        raw=self.kp*(q_des-q)+self.kd*(vel_des-vel)
        if self.bias_compensation: raw=raw+bias
        tau=np.clip(raw,-self.limits,self.limits)
        self.diag.samples+=1
        self.diag.saturated_samples+=int(np.any(np.abs(raw)>self.limits+1e-9))
        self.diag.max_abs_bias=max(self.diag.max_abs_bias,float(np.max(np.abs(bias))))
        self.diag.max_speed=max(self.diag.max_speed,float(np.max(np.abs(vel))))
        self.diag.max_abs_error=max(self.diag.max_abs_error,float(np.max(np.abs(q_des-q))))
        return tau


def settled(q,dq,q_target, *, joint_tol=0.015,speed_tol=0.08):
    return (float(np.max(np.abs(np.asarray(q)-np.asarray(q_target))))<=joint_tol and
            float(np.max(np.abs(dq)))<=speed_tol)


class ArrivalGate:
    def __init__(self,required_samples=25, **thresholds):
        self.required_samples=int(required_samples)
        self.thresholds=thresholds
        self.count=0

    def update(self,q,dq,q_target,additional_ok=True):
        ok=bool(additional_ok) and settled(q,dq,q_target,**self.thresholds)
        self.count=self.count+1 if ok else 0
        return self.count>=self.required_samples


class ComputedTorquePD(SaturatedPD):
    """Inertia-aware feedback implemented in NumPy.

    tau = clip(bias - passive + M @ (omega^2 (q_des-q)
              + 2*zeta*omega (dq_des-dq) + ddq_des), motor_limits).

    MuJoCo separates gravity/Coriolis bias from passive spring/damper forces.
    Omitting the minus-passive term produces substantial steady-state lag when
    fixed joint damping is large compared with the mesh-inferred inertia.

    M, bias, and passive are read from MuJoCo's physics model. The actual
    feedback computation and saturation are independent Python/NumPy.
    """
    def __init__(self,limits, *, natural_frequency=3.0, damping_ratio=1.0):
        super().__init__(np.ones(6),np.ones(6),limits)
        self.natural_frequency=float(natural_frequency)
        self.damping_ratio=float(damping_ratio)
        if self.natural_frequency<=0 or self.damping_ratio<=0:
            raise ValueError('Frequency and damping ratio must be positive')

    def calculate(self,q,vel,q_des,vel_des,bias,mass_matrix,acc_des=None, *, passive=None):
        q=np.asarray(q,float); vel=np.asarray(vel,float)
        bias=np.asarray(bias,float); M=np.asarray(mass_matrix,float)
        if M.shape!=(6,6) or not np.all(np.isfinite(M)):
            raise ValueError('Require finite six-by-six joint inertia matrix')
        acc=np.zeros(6) if acc_des is None else np.asarray(acc_des,float)
        passive=np.zeros(6) if passive is None else np.asarray(passive,float)
        if passive.shape != (6,) or not np.all(np.isfinite(passive)):
            raise ValueError("Require finite 6-vector passive force")
        w=self.natural_frequency; z=self.damping_ratio
        target_acc=(w*w*(np.asarray(q_des)-q)
                    +2*z*w*(np.asarray(vel_des)-vel)+acc)
        raw=bias-passive+M@target_acc
        tau=np.clip(raw,-self.limits,self.limits)
        self.diag.samples+=1
        self.diag.saturated_samples+=int(np.any(np.abs(raw)>self.limits+1e-9))
        self.diag.max_abs_bias=max(self.diag.max_abs_bias,float(np.max(np.abs(bias))))
        self.diag.max_abs_passive=max(self.diag.max_abs_passive,float(np.max(np.abs(passive))))
        self.diag.max_speed=max(self.diag.max_speed,float(np.max(np.abs(vel))))
        self.diag.max_abs_error=max(self.diag.max_abs_error,float(np.max(np.abs(np.asarray(q_des)-q))))
        return tau
