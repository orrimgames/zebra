import mujoco, numpy as np, sys, itertools
sys.path.insert(0,'.')
from orb import model
from orb.params import P
from orb.control import Robot, LowLevel
m=mujoco.MjModel.from_xml_string(model.standalone_xml()); d=mujoco.MjData(m)
rb=Robot(m,P); dt=m.opt.timestep
def run(Kpsi=1.0, Kp=2.0, Kd=0.8, Ki=2.0, slew=1.0, v=1.2, verbose=False, T=20):
    mujoco.mj_resetData(m,d); ll=LowLevel(P)
    stf=0.0; errs=[]; yr=[]; il=0.0
    for i in range(int(T/dt)):
        t=d.time; s=rb.state(d)
        psi_d = 0 if t<5 else (np.radians(30) if t<12 else 0)
        e=np.arctan2(np.sin(psi_d-s['heading']),np.cos(psi_d-s['heading']))
        r_des=np.clip(Kpsi*e,-0.2,0.2)
        c,_=ll.speed(s,v,dt); d.ctrl[rb.act['drive']]=c; d.ctrl[rb.act['pod_level']]=ll.level(s)
        vv=max(abs(s['v']),0.3)*np.sign(s['v'] if abs(s['v'])>0.02 else 1)
        lam_d=np.arctan(r_des*P.geo.R/vv)
        lam=-s['lean']
        cff=2.2+3.3*s['v']**2
        il=np.clip(il+(lam_d-lam)*dt,-0.2,0.2)
        target=cff*lam_d + Kp*(lam_d-lam) + Kd*s['lean_rate'] + Ki*il
        target=np.clip(target,-P.geo.steer_max,P.geo.steer_max)
        stf+=np.clip(target-stf,-slew*dt,slew*dt)
        d.ctrl[rb.act['steer']]=stf
        mujoco.mj_step(m,d)
        if t>4: errs.append(abs(e)); yr.append(abs(s['yaw_rate']))
        if verbose and i%250==0: print(f"{t:5.1f} hdg={np.degrees(s['heading']):6.1f} yr={s['yaw_rate']:+.2f} lean={np.degrees(s['lean']):+.1f} leanrate={s['lean_rate']:+.2f} steer={np.degrees(stf):+.1f}")
    return np.degrees(np.mean(errs)), max(yr)
res=[]
for Kp,Kd,Ki,slew in itertools.product((0,2,5),(0.6,1.2,2.0),(0,3),(0.8,1.5)):
    r=run(Kp=Kp,Kd=Kd,Ki=Ki,slew=slew); res.append((r,Kp,Kd,Ki,slew)); 
res.sort()
for r in res[:8]: print(r)
