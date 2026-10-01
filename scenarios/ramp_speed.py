import mujoco, numpy as np, sys
sys.path.insert(0,'.')
from orb import model
from orb.params import P
from orb.control import Robot, LowLevel
a=np.arctan(1/12.)
# up ramp 6 m long starting x=3, flat top 4m, down ramp 6 m
L=6/np.cos(a); h=0.5
extra=f'''<geom type="box" pos="{3+3} 0 {h/2 - 0.1/np.cos(a)}" size="{L/2} 2 0.1" euler="0 {-a} 0"/>
<geom type="box" pos="{9+2} 0 {h-0.1}" size="2.02 2 0.1"/>
<geom type="box" pos="{13+3} 0 {h/2 - 0.1/np.cos(a)}" size="{L/2} 2 0.1" euler="0 {a} 0"/>'''
m=mujoco.MjModel.from_xml_string(model.standalone_xml(floor_extra=extra)); d=mujoco.MjData(m)
rb=Robot(m,P); dt=m.opt.timestep
def run(**kw):
    mujoco.mj_resetData(m,d); ll=LowLevel(P)
    for k,v in kw.items(): setattr(ll,k,v)
    vmax=0; vmin=9; rows=[]
    for i in range(int(22/dt)):
        s=rb.state(d)
        c,thd=ll.speed(s,1.4,dt); d.ctrl[rb.act['drive']]=c; d.ctrl[rb.act['pod_level']]=ll.level(s)
        mujoco.mj_step(m,d)
        if d.time>4 and s['pos'][0]<21: vmax=max(vmax,s['v']); vmin=min(vmin,s['v'])
        if i%500==0: rows.append((d.time,s['pos'][0],s['v'],np.degrees(s['th_y']),np.degrees(s['pod_pitch']),-c))
    return vmax,vmin,rows
for kw in (dict(),):
    vmax,vmin,rows=run(**kw); print(kw,'vmax %.2f vmin %.2f'%(vmax,vmin))
for r in rows: print("t=%.0f x=%.1f v=%.2f pend=%.0f pod=%.1f tau=%.1f"%r)
