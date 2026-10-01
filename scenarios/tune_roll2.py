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
def run(Kp=0.0, Kd=1.2, slew=5.0, Ky=0.0, yslew=0.4, T=20, wave='sq', v_ref=1.4):
    mujoco.mj_resetData(m,d); ll=LowLevel(P); stf=0.0; yc=0.0
    lmax=0; err=[]
    for i in range(int(T/dt)):
        s=rb.state(d)
        c,_=ll.speed(s,v_ref,dt); d.ctrl[rb.act['drive']]=c; d.ctrl[rb.act['pod_level']]=ll.level(s)
        if wave=='sq': yd=0.22*np.sign(np.sin(2*np.pi*d.time/2.5)) if d.time>2 else 0
        else: yd=0.2*np.sin(2*np.pi*d.time/3.0) if d.time>2 else 0
        yc+=np.clip(yd-yc,-yslew*dt,yslew*dt)
        v=s['v']; vv=max(abs(v),0.3)
        lam_d=np.clip(np.arctan(yc*P.geo.R/vv),-0.12,0.12)
        lam=-s['lean']
        tgt=(2.2+3.3*v*v)*lam_d + Kp*(lam_d-lam) + Kd*s['lean_rate'] + Ky*(yc - s['yaw_rate'])
        tgt=np.clip(tgt,-P.geo.steer_max,P.geo.steer_max)
        stf+=np.clip(tgt-stf,-slew*dt,slew*dt)
        d.ctrl[rb.act['steer']]=stf
        mujoco.mj_step(m,d)
        if d.time>3: lmax=max(lmax,abs(s['lean'])); err.append(yc-s['yaw_rate'])
    return np.degrees(lmax), float(np.sqrt(np.mean(np.square(err))))
if __name__=='__main__':
    res=[]
    for Kp,Kd,slew,Ky,ys in itertools.product((0,1),(0.8,1.2,1.6),(5,),(0,0.5,1.0),(0.4,0.8)):
        l1,e1=run(Kp=Kp,Kd=Kd,slew=slew,Ky=Ky,yslew=ys)
        res.append((l1+40*e1,l1,e1,Kp,Kd,slew,Ky,ys))
    res.sort()
    for r in res[:10]: print("score %.1f maxlean %.1f yaw-rms %.3f  Kp=%g Kd=%g slew=%g Ky=%g yslew=%g"%r)
