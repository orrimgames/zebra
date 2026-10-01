"""ORB v2 engineering stress suite (headless MuJoCo, realistic control rig).

    python scenarios/stress_suite.py            -> out/stress.json (+ plots via make_figs.py)
    python scenarios/stress_suite.py grade wind -> run only some sections

Every controller here runs at 250 Hz on delayed, noisy, biased sensors
(orb/control.py SensorModel), with actuator lags and current limits, the
same as on the robot. Food acceleration is reported after a 10 ms low-pass
(what a cup actually feels; sim contact spikes shorter than that are not physical).
"""
import os, sys, json, time
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
from orb.harness import Rig, perturbed
from orb.params import Params
from orb.wind import Wind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)


class FoodMeter:
    """Peak food acceleration (g, 10 ms low-pass) and liquid-surface tilt."""
    def __init__(self, d):
        self.d = d; self.af = np.array([0, 0, 9.81]); self.peak = 0.0; self.tilt = 0.0
    def __call__(self):
        a = self.d.sensor("pod_acc").data
        self.af = self.af + 0.4 * (a - self.af)
        self.peak = max(self.peak, float(np.linalg.norm(self.af - [0, 0, 9.81]) / 9.81))
        if self.af[2] > 4.9:
            self.tilt = max(self.tilt, float(np.degrees(np.arctan2(np.hypot(self.af[0], self.af[1]), self.af[2]))))


def pursuit(r, y_line=0.0, direction=None):
    """Pure pursuit along the line y = y_line, travelling +x (or -x if the robot starts facing -x)."""
    if direction is None:
        direction = 1.0 if np.cos(r.rb.state(r.d)["heading"]) >= 0 else -1.0
    def yaw(t, s):
        st = r.rb.state(r.d)
        v = max(abs(st["v"]), 0.3)
        Ld = 0.9 + 0.7 * v
        a = np.arctan2(direction * (y_line - st["pos"][1]), Ld) + (0.0 if direction > 0 else np.pi) - st["heading"]
        a = (a + np.pi) % (2 * np.pi) - np.pi
        mx = r.ll.max_yaw_rate(v)
        return float(np.clip(2 * v * np.sin(a) / Ld, -mx, mx))
    return yaw


# ----------------------------------------------------------------------------- grade
def test_grade(P=None):
    P = P or Params()
    out = []
    for pct in (0, 5, 8.33, 12, 15, 18, 20, 22, 25, 28):
        a = np.arctan(pct / 100)
        L = 14.0
        extra = f'<geom type="box" pos="{2 + L/2*np.cos(a)} 0 {L/2*np.sin(a) - 0.1/np.cos(a)}" size="{L/2} 3 0.1" euler="0 {-a} 0"/>' if pct > 0 else ""
        res = dict(grade_pct=pct)
        for direction in ("up", "down"):
            if direction == "down" and pct == 0:
                continue
            if direction == "up":
                r = Rig(P, extra=extra, seed=1)
            else:
                # start on the slope near the top, facing down it
                x0 = 2 + (L - 2.0) * np.cos(a)
                z0 = (L - 2.0) * np.sin(a)
                r = Rig(P, extra=extra, seed=1, pos=(x0, 0), yaw=np.pi)
                r.d.qpos[2] += z0; mujoco.mj_forward(r.m, r.d)
            fm = FoodMeter(r.d); log = []
            vset = 1.0
            def cb(t, s):
                fm(); log.append((s["pos"][0], s["v"], s["th_y"], abs(s["pod_pitch"])))
            r.run(16 if direction == "up" else 10, lambda t, s: vset, yaw_fn=pursuit(r), cb=cb)
            Lg = np.array(log)
            if direction == "up":
                on = (Lg[:, 0] > 4) & (Lg[:, 0] < 12)
                ok = bool(on.sum() > 50 and Lg[:, 0].max() > 9 and np.percentile(Lg[on, 1], 10) > 0.6)
                res["up_ok"] = ok
                res["up_pendulum_deg"] = round(float(np.degrees(np.median(Lg[on, 2]))), 1) if on.any() else None
            else:
                sel = np.arange(len(Lg)) > len(Lg) * 0.4
                res["down_ok"] = bool(np.abs(Lg[sel, 1]).max() < vset + 0.35)
                res["down_max_speed"] = round(float(np.abs(Lg[sel, 1]).max()), 2)
            res[f"{direction}_food_g"] = round(fm.peak, 2)
        out.append(res)
        print("grade", res, flush=True)
    return out


# ----------------------------------------------------------------------------- lips
def test_lips(P=None):
    P = P or Params()
    out = []
    for h_mm in (5, 10, 15, 20, 25, 30):
        h = h_mm / 1000
        row = dict(height_mm=h_mm)
        for v in (0.4, 0.8, 1.2, 1.6):
            extra = f'<geom type="box" pos="4.5 0 {h/2}" size="1.5 3 {h/2}"/>'
            r = Rig(P, extra=extra, seed=1)
            fm = FoodMeter(r.d)
            def cb(t, s):
                if t > 0.5:
                    fm()
                return s["pos"][0] < 4.6
            r.run(16, lambda t, s: v, cb=cb)
            row[f"v{v}"] = dict(passes=bool(r.d.qpos[0] > 4.0), food_g=round(fm.peak, 2))
        out.append(row)
        print("lips", row, flush=True)
    return out


# ----------------------------------------------------------------------------- curbs
def test_curbs(P=None):
    from curb_test import curb_run
    out = []
    for up in (True, False):
        for h in (0.03, 0.06, 0.10, 0.12, 0.15):
            r = curb_run(h, up, P=P, seed=0)
            r.pop("events", None)
            out.append(r)
            print("curb", r, flush=True)
    return out


def test_curb_robustness(n=15):
    """Wet curb (mu 0.55-0.9), +/-3 cm edge error, +/-1 cm height error, 0-7 kg food, built-robot spread."""
    from curb_test import curb_run
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        Pp = perturbed(rng, level=0.7)
        mu = float(rng.uniform(0.55, 0.9))
        for up, h in ((True, 0.12), (False, 0.12), (True, 0.15), (False, 0.15)):
            r = curb_run(h, up, P=Pp, seed=100 + i, edge_noise=0.06, mu_edge=mu)
            rows.append(dict(i=i, up=up, h=h, mu=round(mu, 2), payload=round(Pp.m.payload, 1), ok=r["ok"], result=r["result"],
                             safe=bool(r["ok"] or (r["result"] in ("abort", "too_high") and r["peak_pod_g"] < 5.0)),
                             food_g=r["peak_pod_g"], time_s=r["time_s"]))
            print("curb-rob", rows[-1], flush=True)
    ok = [r["ok"] for r in rows]
    return dict(trials=len(rows), success_rate=round(float(np.mean(ok)), 3),
                safe_outcome_rate=round(float(np.mean([r["safe"] for r in rows])), 3),
                refused_or_aborted=int(sum((not r["ok"]) and r["safe"] for r in rows)), worst_food_g=round(max(r["food_g"] for r in rows), 2),
                p90_food_g=round(float(np.percentile([r["food_g"] for r in rows], 90)), 2), rows=rows)


# ----------------------------------------------------------------------------- braking
def test_braking(P=None):
    P = P or Params()
    out = []
    for v0 in (1.4, 2.0, 2.2):
        for emerg in (False, True):
            r = Rig(P, seed=1)
            fm = FoodMeter(r.d)
            rec = {}
            def vf(t, s):
                if t >= 8:
                    r.ll.emergency = emerg
                    return 0.0
                return v0
            def cb(t, s):
                if t >= 8:
                    fm()
                    if "x0" not in rec:
                        rec["x0"] = s["pos"][0]; rec["v0"] = s["v"]
                    if abs(s["v"]) < 0.03 and "x1" not in rec and t > 8.3:
                        rec["x1"] = s["pos"][0]; rec["t1"] = t - 8
            r.run(14, vf, cb=cb)
            out.append(dict(from_mps=v0, emergency=emerg, stop_dist_m=round(rec["x1"] - rec["x0"], 2), stop_time_s=round(rec["t1"], 2),
                            food_g=round(fm.peak, 2), liquid_tilt_deg=round(fm.tilt, 1)))
            print("brake", out[-1], flush=True)
    return out


# ----------------------------------------------------------------------------- turning
def test_turning(P=None):
    P = P or Params()
    out = []
    for v in (0.4, 0.8, 1.2, 1.6, 2.0, 2.2):
        r = Rig(P, seed=2)
        mx = r.ll.max_yaw_rate(v)
        pts = []
        r.run(24, lambda t, s: v, yaw_fn=lambda t, s: (mx if t > 6 else 0.0),
              cb=lambda t, s: pts.append((t, s["pos"][0], s["pos"][1], s["lean"])))
        A = np.array(pts); sel = A[:, 0] > 13
        x, y = A[sel, 1], A[sel, 2]
        M = np.stack([x, y, np.ones_like(x)], 1)
        sol = np.linalg.lstsq(M, -(x * x + y * y), rcond=None)[0]
        rad = float(np.sqrt(sol[0] ** 2 / 4 + sol[1] ** 2 / 4 - sol[2]))
        out.append(dict(speed_mps=v, turn_radius_m=round(rad, 2), lean_deg=round(float(np.degrees(np.mean(np.abs(A[sel, 3])))), 1),
                        planner_radius_m=round(r.ll.min_turn_radius(v), 2)))
        print("turn", out[-1], flush=True)
    return out


# ----------------------------------------------------------------------------- wind
def test_wind(P=None):
    P = P or Params()
    out = dict(lane=[], headwind=[], parked=[], estimator=None)
    # lane keeping in a gusty crosswind
    for U in (6.0, 10.0, 14.0):
        for v in (0.8, 1.4, 2.0):
            w = Wind(U, np.pi / 2, turb=0.18, gusts=[(10.0, 3.0, 0.6 * U, 0.0)], seed=int(U * 10 + v * 3))
            r = Rig(P, seed=3, wind=w)
            log = []
            r.run(18, lambda t, s: v, yaw_fn=pursuit(r),
                  cb=lambda t, s: log.append((t, s["pos"][1], s["lean"], s["steer"], np.linalg.norm(w.last))))
            L = np.array(log); sel = L[:, 0] > 4
            row = dict(mean_mps=U, peak_gust_mps=round(float(L[:, 4].max()), 1), speed_mps=v,
                       max_lateral_m=round(float(np.abs(L[sel, 1]).max()), 3), max_lean_deg=round(float(np.degrees(np.abs(L[sel, 2]).max())), 1),
                       tray_use_pct=round(float(100 * np.abs(L[sel, 3]).max() / P.geo.steer_max), 0))
            out["lane"].append(row); print("wind-lane", row, flush=True)
    # headwind / tailwind speed holding
    for U in (10.0, 15.0, -15.0):
        w = Wind(abs(U), np.pi if U > 0 else 0.0, turb=0.15, seed=5)
        r = Rig(P, seed=4, wind=w)
        log = []
        r.run(14, lambda t, s: 1.4, yaw_fn=pursuit(r), cb=lambda t, s: log.append((t, s["v"], s["th_y"])))
        L = np.array(log); sel = L[:, 0] > 6
        row = dict(wind_mps=U, mean_speed=round(float(L[sel, 1].mean()), 2), min_speed=round(float(L[sel, 1].min()), 2),
                   pendulum_deg=round(float(np.degrees(np.median(L[sel, 2]))), 1))
        out["headwind"].append(row); print("wind-head", row, flush=True)
    # parked, legs down, hatch OPEN, gusts from the side and from behind the lid
    for G, direc, lid in ((15.0, np.pi / 2, True), (15.0, 0.0, True), (20.0, 0.0, False), (25.0, np.pi / 2, False), (25.0, 0.0, False), (30.0, np.pi / 2, False), (30.0, 0.0, False)):
        w = Wind(0.4 * G, direc, turb=0.2, gusts=[(6.0, 3.0, 0.6 * G, 0.0)], seed=9)
        r = Rig(P, seed=5, wind=w)
        gq = r.legs.ground_q()
        W = P.total_mass() * 9.81
        park = Parker(r, W, t0=1.0)
        def cb(t, s):
            park(t)
            if park.t_down is not None and t > park.t_down + 0.3 and lid:
                r.lid_cmd = -1.95
            if "p0" not in rec and t > 5.0:
                rec["p0"] = s["pos"].copy()
            if t > 5.0:
                rec["tilt"] = max(rec.get("tilt", 0), abs(np.degrees(s["lean"])), abs(np.degrees(s["pod_pitch"])))
        rec = {}
        r.run(16, lambda t, s: 0.0, cb=cb, parked_fn=park.parked)
        st = r.rb.state(r.d)
        row = dict(peak_gust_mps=G, from_side=bool(direc != 0.0), hatch_open=lid, drift_mm=round(1000 * float(np.linalg.norm(st["pos"][:2] - rec["p0"][:2])), 1),
                   max_tilt_deg=round(rec["tilt"], 1), stays=bool(np.linalg.norm(st["pos"][:2] - rec["p0"][:2]) < 0.05))
        out["parked"].append(row); print("wind-park", row, flush=True)
    # does the robot know how windy it is? (steady crosswind, straight line)
    est = []
    for U in (5.0, 10.0, 15.0):
        w = Wind(U, np.pi / 2, turb=0.0, seed=1)
        r = Rig(P, seed=6, wind=w)
        vals = []
        r.run(14, lambda t, s: 1.2, yaw_fn=pursuit(r), cb=lambda t, s: vals.append((t, r.ll.crosswind_estimate(), r.wind_force[1])))
        V = np.array(vals); sel = V[:, 0] > 9
        F_est = float(np.mean(V[sel, 1])); F_true = float(np.mean(V[sel, 2]))
        U_est = float(np.sqrt(abs(F_est) / (0.5 * P.a.rho * P.a.cd_sphere * np.pi * P.geo.R ** 2)))
        est.append(dict(true_mps=U, estimated_mps=round(U_est, 1), side_force_true_N=round(F_true, 1), side_force_est_N=round(F_est, 1)))
    out["estimator"] = est
    print("wind-est", est, flush=True)
    return out


# ----------------------------------------------------------------------------- parking
class Parker:
    """Park the way the robot does it: the speed loop stops the ball and holds
    it until it's still, THEN the brake goes on, THEN the legs come down."""
    def __init__(self, r, W, t0):
        self.r, self.W, self.t0 = r, W, t0
        self.t_brake = self.t_down = None
        self.still = 0.0

    def __call__(self, t):
        r = self.r
        if t <= self.t0:
            return
        if self.t_brake is None:
            st = r.rb.state(r.d)
            self.still = self.still + 0.004 if abs(st["v"]) < 0.02 else 0.0
            if self.still > 0.3:
                r.brake(True); self.t_brake = t
            return
        if r.legs.park(r.d, self.W, t - self.t_brake) and self.t_down is None:
            self.t_down = t

    def parked(self, t, s):
        return self.t_brake is not None


def test_park(P=None):
    P = P or Params()
    out = []
    for pct, axis in ((0, "cross"), (5, "cross"), (8.33, "cross"), (12, "cross"), (8.33, "along"), (15, "along")):
        a = np.arctan(pct / 100)
        r = Rig(P, seed=7, floor_euler=(a, 0, 0) if axis == "cross" else (0, a, 0))
        gq = r.legs.ground_q()
        rec = {}
        W = P.total_mass() * 9.81
        park = Parker(r, W, t0=0.6)
        def cb(t, s):
            park(t)
            if park.t_down is not None and t > park.t_down + 0.5:
                r.lid_cmd = -1.95
                if "p0" not in rec:
                    rec["p0"] = s["pos"].copy()
                rec["tilt"] = max(rec.get("tilt", 0.0), abs(np.degrees(s["pod_pitch"])))
        r.run(16, lambda t, s: 0.0, cb=cb, parked_fn=park.parked)
        st = r.rb.state(r.d)
        drift = float(np.linalg.norm(st["pos"][:2] - rec["p0"][:2])) if "p0" in rec else float("nan")
        rec.setdefault("tilt", float("nan"))
        out.append(dict(slope_pct=pct, direction=axis, legs_down_s=round(park.t_down - 0.6, 2) if park.t_down else None, drift_mm=round(drift * 1000, 1),
                        pod_tilt_deg=round(abs(np.degrees(st["pod_roll"])) + abs(np.degrees(st["pod_pitch"])), 1), max_body_pitch_deg=round(rec["tilt"], 1),
                        stable=bool(drift < 0.02)))
        print("park", out[-1], flush=True)
    return out


# ----------------------------------------------------------------------------- energy
def test_energy(P=None):
    P = P or Params()
    E = P.e
    out = []
    for v in (1.4, 2.0):
        r = Rig(P, seed=8)
        acc = dict(drive=0.0, steer=0.0, level=0.0)
        xs, path = [], []
        dt = r.m.opt.timestep * r.n_ctrl
        def cb(t, s):
            if t < 10:
                return
            d, rb = r.d, r.rb
            tau = d.actuator_force[rb.act["drive"]]; w = d.qvel[rb.va["drive"]] - d.qvel[rb.va["pod_pitch"]] * 0
            acc["drive"] += (max(tau * d.qvel[rb.va["drive"]], 0) / E.drivetrain_eff + 0.02 * tau * tau) * dt
            ts = d.actuator_force[rb.act["steer"]]; ws = d.qvel[rb.va["steer"]]
            acc["steer"] += (abs(ts * ws) / 0.6 + 0.01 * ts * ts + 0.3) * dt
            tl = d.actuator_force[rb.act["pod_level"]]
            acc["level"] += (0.02 * tl * tl) * dt
            xs.append(np.linalg.norm(s["pos"][:2]) if False else 0.0)
            path.append(s["pos"][:2].copy())
        r.run(70, lambda t, s: v, yaw_fn=lambda t, s: 0.05 * np.sin(t * 0.7), cb=cb)
        dur = 60.0
        dist = float(np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1)))
        hotel = sum(E.hotel_w.values())
        p_motion = sum(acc.values()) / dur
        p_tot = p_motion + hotel
        usable = E.battery_wh * E.usable_frac
        out.append(dict(speed_mps=v, motion_W=round(p_motion, 1), hotel_W=round(hotel, 1), total_W=round(p_tot, 1),
                        Wh_per_km=round(p_tot * dur / 3600 / (dist / 1000), 1),
                        runtime_driving_h=round(usable / p_tot, 1), range_km=round(usable / p_tot * v * 3.6, 1),
                        runtime_typical_day_h=round(usable / (0.4 * p_tot + 0.6 * (hotel - 6.0)), 1)))
        print("energy", out[-1], flush=True)
    return out


# ----------------------------------------------------------------------------- sim-to-real Monte Carlo
def mc_one(i):
    """One randomly-built robot (mass/CG/friction/motor/latency/sensor spread,
    0-7 kg food) does: 2 m/s S-curves, emergency stop from 2 m/s, gusty
    crosswind lane keeping, and holds on a 12 % ramp. Returns pass/fail per task."""
    rng = np.random.default_rng(1000 + i)
    Pt = perturbed(rng, level=1.0)
    Pn = Params()                                     # controller only knows the nominal design
    res = dict(i=i, payload=round(Pt.m.payload, 1), mu=round(Pt.c.slide, 2), delay_ms=round(1000 * Pt.sensor_delay, 1),
               torque_scale=round(Pt.d.motor_peak_torque / Pn.d.motor_peak_torque, 2))
    # S-curves at 2 m/s
    r = Rig(Pt, P_nom=Pn, seed=i, sensor_scale=2.0)
    fm = FoodMeter(r.d); log = []
    def yaw(t, s):
        return 0.6 * r.ll.max_yaw_rate(2.0) * np.sign(np.sin(2 * np.pi * t / 6.0)) if t > 4 else 0.0
    r.run(20, lambda t, s: 2.0, yaw_fn=yaw, cb=lambda t, s: (fm(), log.append((t, s["lean"], s["v"], s["th_y"]))))
    L = np.array(log)
    res["scurve_ok"] = bool(np.degrees(np.abs(L[:, 1]).max()) < 16 and np.abs(L[L[:, 0] > 6, 2] - 2.0).max() < 0.5)
    res["scurve_max_lean"] = round(float(np.degrees(np.abs(L[:, 1]).max())), 1)
    # emergency stop from 2 m/s
    r = Rig(Pt, P_nom=Pn, seed=i + 1, sensor_scale=2.0)
    rec = {}
    def vf(t, s):
        if t > 7:
            r.ll.emergency = True; return 0.0
        return 2.0
    def cb(t, s):
        if t > 7 and "x0" not in rec:
            rec["x0"] = s["pos"][0]
        if t > 7.3 and abs(s["v"]) < 0.03 and "x1" not in rec:
            rec["x1"] = s["pos"][0]
    r.run(12, vf, cb=cb)
    res["estop_m"] = round(rec.get("x1", np.nan) - rec["x0"], 2) if "x1" in rec else None
    res["estop_ok"] = bool(res["estop_m"] is not None and res["estop_m"] < 1.6)
    # gusty crosswind at the top of the driving envelope (8 m/s mean, gusts to ~14-15 m/s):
    # above this the planner pulls over and braces on its legs
    w = Wind(8.0, np.pi / 2, turb=0.2, gusts=[(8.0, 3.0, 6.0, 0.0)], seed=i)
    r = Rig(Pt, P_nom=Pn, seed=i + 2, sensor_scale=2.0, wind=w)
    log = []
    r.run(15, lambda t, s: 1.2, yaw_fn=pursuit(r), cb=lambda t, s: log.append((t, s["pos"][1])))
    L = np.array(log)
    res["wind_dev_m"] = round(float(np.abs(L[L[:, 0] > 4, 1]).max()), 2)
    res["wind_ok"] = bool(res["wind_dev_m"] < 0.45)
    # 12 % ramp climb
    a = np.arctan(0.12); Lr = 12.0
    extra = f'<geom type="box" pos="{2 + Lr/2*np.cos(a)} 0 {Lr/2*np.sin(a) - 0.1/np.cos(a)}" size="{Lr/2} 3 0.1" euler="0 {-a} 0"/>'
    r = Rig(Pt, P_nom=Pn, extra=extra, seed=i + 3, sensor_scale=2.0)
    log = []
    r.run(14, lambda t, s: 1.0, yaw_fn=pursuit(r), cb=lambda t, s: log.append((s["pos"][0], s["v"])))
    L = np.array(log); on = (L[:, 0] > 4) & (L[:, 0] < 10)
    res["ramp_ok"] = bool(on.sum() > 50 and np.percentile(L[on, 1], 10) > 0.6)
    res["all_ok"] = bool(res["scurve_ok"] and res["estop_ok"] and res["wind_ok"] and res["ramp_ok"])
    print("mc", res, flush=True)
    return res


def test_montecarlo(n=40, workers=1):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(mc_one, range(n))
    keys = ("scurve_ok", "estop_ok", "wind_ok", "ramp_ok", "all_ok")
    rates = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
    es = [r["estop_m"] for r in rows if r["estop_m"] is not None]
    return dict(n=n, pass_rates=rates, estop_p95_m=round(float(np.percentile(es, 95)), 2) if es else None,
                wind_dev_p95_m=round(float(np.percentile([r["wind_dev_m"] for r in rows], 95)), 2), rows=rows)


SECTIONS = dict(grade=test_grade, lips=test_lips, curbs=test_curbs, curb_robustness=test_curb_robustness,
                braking=test_braking, turning=test_turning, wind=test_wind, park=test_park, energy=test_energy,
                montecarlo=test_montecarlo)

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    path = os.path.join(OUT, "stress.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for name in want:
        t0 = time.time()
        print("=== running", name, flush=True)
        results[name] = SECTIONS[name]()
        results[name + "_wall_s"] = round(time.time() - t0)
        P = Params()
        results["mass"] = dict(total_loaded_kg=round(P.total_mass(), 2), empty_kg=round(P.total_mass() - P.m.payload, 2),
                               pendulum_moment_kgm=round(P.pendulum_moment(), 3), steer_moment_kgm=round(P.steer_moment(), 3),
                               cg_below_centre_m=round(P.cg_depth(), 3), static_grade_limit_deg_at_60deg_swing=round(P.max_static_grade(np.radians(60)), 1))
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    print("saved", path)
