"""Curb / step up and down with the legs. Reports success, time and food g."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, mujoco
from orb.harness import Rig
from orb.params import Params
from orb.curb import CurbClimber


def curb_run(h, up=True, P=None, seed=0, edge_noise=0.02, mu_edge=0.9, T=40, verbose=False, video=None):
    P = P or Params()
    rng = np.random.default_rng(seed)
    xe = 2.0
    if up:
        extra = f'<geom name="curb" type="box" pos="{xe + 3} 0 {h/2}" size="3 3 {h/2}" friction="{mu_edge} 0.01 0.003" rgba="0.80 0.76 0.68 1"/>'
        z0 = 0.0
    else:
        extra = f'<geom name="curb" type="box" pos="{xe - 4} 0 {h/2}" size="4 3 {h/2}" friction="{mu_edge} 0.01 0.003" rgba="0.80 0.76 0.68 1"/>'
        z0 = h
    r = Rig(P, extra=extra, realistic=True, seed=seed, pos=(0.0, 0.0))
    if not up:
        r.d.qpos[2] += h; mujoco.mj_forward(r.m, r.d)
    xe_est = xe + rng.normal(0, edge_noise / 2)
    h_est = h + rng.normal(0, 0.006)
    cc = CurbClimber(P, r.legs, h_est, xe_est, up=up, ll=r.ll)
    from orb.curb import CliffSensor
    cliff = CliffSensor(r.m, r.rb, seed=seed)
    odo = dict(x=0.0, prev=None)
    rec = dict(peak_g=0.0, soup=0.0, t_done=None, zmax=0.0, lean=0.0)
    wq = [r.m.jnt_dofadr[r.m.joint(n).id] for n in ("wheel_3", "wheel_4")]
    def vfn(t, s):
        s = dict(s)
        s["legF"] = r.legs.forces(r.d)
        s["cliff"] = cliff.read(r.d)
        s["wheel_v"] = float(np.mean([r.d.qvel[a] for a in wq]) * P.geo.leg_wheel_r)
        return cc.step(t, s, odo["x"])
    frames = []
    ren = None
    if video:
        import mujoco as mj
        ren = mj.Renderer(r.m, 360, 640)
        cam = mj.MjvCamera(); cam.azimuth = 90; cam.elevation = -4; cam.distance = 1.9
    def cb(t, st):
        if ren is not None and int(round(t / 0.004)) % 10 == 0:
            cam.lookat[:] = [st["pos"][0], 0, 0.35]
            ren.update_scene(r.d, cam); img = ren.render()
            import cv2
            img = cv2.putText(img.copy(), f"{cc.state}  t={t:4.1f}s  legs={np.round(st['legs'],2)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frames.append(img)
        if odo["prev"] is not None:
            odo["x"] += (st["pos"][0] - odo["prev"]) * (1 + 0.01)
        odo["prev"] = st["pos"][0]
        if t > 0.5:
            a = r.d.sensor("pod_acc").data
            rec["af"] = rec.get("af", np.array([0, 0, 9.81])) + 0.4 * (a - rec.get("af", np.array([0, 0, 9.81])))   # ~10 ms low-pass (food/cup response)
            gg = float(np.linalg.norm(rec["af"] - [0, 0, 9.81]) / 9.81)
            if gg > rec["peak_g"]:
                rec["peak_g"], rec["t_peak"], rec["st_peak"] = gg, round(t, 2), cc.state
            af = rec["af"]
            if af[2] > 4.9:
                rec["soup"] = max(rec["soup"], float(np.degrees(np.arctan2(np.hypot(af[0], af[1]), af[2]))))
            rec["lean"] = max(rec["lean"], abs(np.degrees(st["lean"])))
        if verbose and int(t * 250) % 250 == 0:
            print(f"t={t:5.1f} {cc.state:8s} x={st['pos'][0]:.3f} z={st['pos'][2]:.3f} v={st['v']:+.2f} legs={np.round(st['legs'],3)} th={np.degrees(st['th_y']):.0f}")
        if cc.result is not None:
            rec["t_done"] = t
            return False
    r.run(T, vfn, cb=cb)
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=25)
    st = r.rb.state(r.d)
    zc = st["pos"][2]
    xf = st["pos"][0]
    ok = cc.result == "done" and (abs(zc - (h + P.geo.R)) < 0.02 if up else abs(zc - P.geo.R) < 0.02) and (xe < xf < xe + 0.8)
    return dict(height_mm=round(h * 1000), up=up, ok=bool(ok), final=(round(float(xf), 3), round(float(zc), 3)), result=cc.result, time_s=round(rec["t_done"] - cc.events[0][0], 1) if rec["t_done"] and cc.events else None,
                peak_pod_g=round(rec["peak_g"], 2), peak_at=(rec.get("t_peak"), rec.get("st_peak")), max_soup_deg=round(rec["soup"], 1), max_lean_deg=round(rec["lean"], 1), events=cc.events)


if __name__ == "__main__":
    import json
    for up in (True, False):
        for h in (0.05, 0.10, 0.15):
            print(json.dumps(curb_run(h, up, verbose="-v" in sys.argv)), flush=True)
