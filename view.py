"""Interactive MuJoCo viewer.

    python view.py               # watch the full autonomous delivery live
    python view.py --teleop      # drive it yourself on the campus

Teleop keys (click the viewer window first):
    Up / Down      faster / slower (also reverse)
    Left / Right   steer (lean the battery)
    Space          stop
    O              open / close the hatch (only when stopped)
Tip: double-click the ball and press Ctrl+Right-drag to orbit; use the
"Camera" dropdown in the left panel to look through cam_left / cam_right.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import mujoco.viewer

ap = argparse.ArgumentParser()
ap.add_argument("--teleop", action="store_true")
ap.add_argument("--payload", type=float, default=5.0)
a = ap.parse_args()

from orb.sim import Sim

if not a.teleop:
    sim = Sim(video=False, payload=a.payload)
    with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.m.body("shell").id
        v.cam.distance = 3.5; v.cam.elevation = -20
        t0 = time.time()
        def hook(s):
            if not v.is_running():
                return False
            if s.d.time - (time.time() - t0) > 0:            # keep to real time
                time.sleep(max(0.0, s.d.time - (time.time() - t0)))
            if int(s.d.time * 500) % 8 == 0:
                v.sync()
            return True
        print(sim.run(T=400, verbose=True, hook=hook))
    sys.exit()

# ----------------------------------------------------------------------- teleop
from orb.control import Robot, LowLevel
sim = Sim(video=False, payload=a.payload)
m, d, rb, ll = sim.m, sim.d, sim.rb, sim.ll
cmd = dict(v=0.0, yaw=0.0, lid=False)

def key(k):
    if k == 265: cmd["v"] = min(2.0, cmd["v"] + 0.2)
    elif k == 264: cmd["v"] = max(-0.6, cmd["v"] - 0.2)
    elif k == 263: cmd["yaw"] = min(0.3, cmd["yaw"] + 0.06)
    elif k == 262: cmd["yaw"] = max(-0.3, cmd["yaw"] - 0.06)
    elif k == 32: cmd["v"], cmd["yaw"] = 0.0, 0.0
    elif k == ord("O") and abs(rb.state(d)["v"]) < 0.05: cmd["lid"] = not cmd["lid"]
    print(f"v_cmd={cmd['v']:+.1f} m/s  yaw_cmd={cmd['yaw']:+.2f} rad/s  hatch={'open' if cmd['lid'] else 'closed'}")

with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = m.body("shell").id
    v.cam.distance = 3.5; v.cam.elevation = -20
    dt = m.opt.timestep
    t0 = time.time()
    while v.is_running():
        s = rb.state(d)
        sim.update_agents(d.time, s["pos"])
        c, _ = ll.speed(s, cmd["v"], dt)
        d.ctrl[rb.act["drive"]] = c
        d.ctrl[rb.act["steer"]] = ll.steer(s, cmd["yaw"], dt)
        d.ctrl[rb.act["pod_level"]] = ll.level(s)
        d.ctrl[rb.act["lid"]] = -1.9 if cmd["lid"] else 0.0
        sim.eyes.update(d.time, (np.clip(cmd["yaw"] * 3, -0.6, 0.6), 0.0), (0.35, 0.95, 1.0), happy=cmd["lid"])
        mujoco.mj_step(m, d)
        if int(d.time / dt) % 8 == 0:
            v.sync()
            lag = d.time - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
