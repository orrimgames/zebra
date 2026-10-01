"""Open-loop steering map: battery swing angle -> lean -> yaw rate, at several speeds."""
import mujoco, numpy as np, sys
sys.path.insert(0, '.')
from orb import model
from orb.params import P
from orb.control import Robot, LowLevel

m = mujoco.MjModel.from_xml_string(model.standalone_xml()); d = mujoco.MjData(m)
rb = Robot(m, P); dt = m.opt.timestep
rows = []
for v in (0.5, 1.0, 1.5):
    for st in (5, 10, 15, 20, 25):
        mujoco.mj_resetData(m, d); ll = LowLevel(P)
        yr, ln = [], []
        for i in range(int(9 / dt)):
            s = rb.state(d)
            c, _ = ll.speed(s, v, dt)
            d.ctrl[rb.act['drive']] = c; d.ctrl[rb.act['pod_level']] = ll.level(s)
            d.ctrl[rb.act['steer']] = np.radians(st) if d.time > 4 else 0
            mujoco.mj_step(m, d)
            if d.time > 6.5:
                yr.append(s['yaw_rate']); ln.append(s['lean'])
        r = np.mean(yr); l = np.mean(ln)
        rows.append((v, st, np.degrees(l), r, abs(s['v']) / max(abs(r), 1e-6)))
        print(f"v={v} steer={st:2d}deg lean={np.degrees(l):6.2f}deg yaw_rate={r:6.3f} rad/s  radius={rows[-1][-1]:5.2f} m  pred R/tan(lean)={P.geo.R/np.tan(abs(l)):5.2f}")
