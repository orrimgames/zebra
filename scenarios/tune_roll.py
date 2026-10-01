"""Worst-case roll stability: square-wave yaw commands while going over a
1:12 ramp (up, flat, down).  Reports max lean and heading wobble."""
import mujoco, numpy as np, sys, itertools
sys.path.insert(0,'.')
from orb import model
from orb.params import P
from orb.control import Robot, LowLevel
a=np.arctan(1/12.); L=6/np.cos(a); h=0.5
extra=f'''<geom type="box" pos="{3+3} 0 {h/2 - 0.1/np.cos(a)}" size="{L/2} 3 0.1" euler="0 {-a} 0"/>
<geom type="box" pos="{9+2} 0 {h-0.1}" size="2.02 3 0.1"/>
<geom type="box" pos="{13+3} 0 {h/2 - 0.1/np.cos(a)}" size="{L/2} 3 0.1" euler="0 {a} 0"/>'''
m=mujoco.MjModel.from_xml_string(model.standalone_xml(floor_extra=extra)); d=mujoco.MjData(m)
rb=Robot(m,P); dt=m.opt.timestep

def run(Kp=2.0, Kd=1.2, slew=3.0, lam_max=0.12, period=1.6, amp=0.22, T=20, law=None):
    mujoco.mj_resetData(m,d); ll=LowLevel(P); stf=0.0
    lmax=0; lean_hist=[]
    for i in range(int(T/dt)):
        s=rb.state(d)
        c,_=ll.speed(s,1.4,dt); d.ctrl[rb.act['drive']]=c; d.ctrl[rb.act['pod_level']]=ll.level(s)
        yd=amp*np.sign(np.sin(2*np.pi*d.time/period)) if d.time>2 else 0
        v=s['v']; vv=max(abs(v),0.3)
        lam_d=np.clip(np.arctan(yd*P.geo.R/vv),-lam_max,lam_max)
        lam=-s['lean']
        tgt=(2.2+3.3*v*v)*lam_d + Kp*(lam_d-lam) + Kd*s['lean_rate']
        tgt=np.clip(tgt,-P.geo.steer_max,P.geo.steer_max)
        stf+=np.clip(tgt-stf,-slew*dt,slew*dt)
        d.ctrl[rb.act['steer']]=stf
        mujoco.mj_step(m,d)
        if d.time>2: lmax=max(lmax,abs(s['lean']))
    return np.degrees(lmax)
if __name__=='__main__':
    res=[]
    for Kp,Kd,slew in itertools.product((0,2,4),(1.2,2.0,3.0),(1.5,3.0,5.0)):
        res.append((run(Kp=Kp,Kd=Kd,slew=slew),Kp,Kd,slew))
    res.sort()
    for r in res: print("max lean %.1f deg  Kp=%g Kd=%g slew=%g"%r)
