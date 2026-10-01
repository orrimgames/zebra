"""Engineering test suite for the ORB delivery sphere (headless MuJoCo).

    python scenarios/test_suite.py          -> out/tests.json + out/tests.png

Every number in the design spec comes from here.
"""
import os, sys, json
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")   # headless Linux; desktops use the default GL
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco
from orb import model
from orb.params import Params
from orb.control import Robot, LowLevel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)


def make(P, extra=""):
    m = mujoco.MjModel.from_xml_string(model.standalone_xml(P, floor_extra=extra))
    return m, mujoco.MjData(m), Robot(m, P)


def drive(m, d, rb, ll, v_fn, T, yaw_fn=None, cb=None):
    dt = m.opt.timestep
    for i in range(int(T / dt)):
        s = rb.state(d)
        c, _ = ll.speed(s, v_fn(d.time, s), dt)
        d.ctrl[rb.act["drive"]] = c
        d.ctrl[rb.act["pod_level"]] = ll.level(s)
        d.ctrl[rb.act["steer"]] = ll.steer(s, yaw_fn(d.time, s) if yaw_fn else 0.0, dt)
        mujoco.mj_step(m, d)
        if cb and cb(d.time, s) is False:
            break


def soup_tilt(d):
    a = d.sensor("pod_acc").data
    return float(np.degrees(np.arctan2(np.hypot(a[0], a[1]), a[2])))


# ----------------------------------------------------------------------------- 1. grade
def test_grade(P):
    """Climb a long constant slope at 0.6 m/s; pass if it holds >=0.4 m/s for 3 m."""
    res = []
    for pct in (0, 4, 8.33, 10, 12, 14, 16, 18, 20, 22):
        a = np.arctan(pct / 100)
        L = 14.0
        extra = f'<geom type="box" pos="{2 + L/2*np.cos(a)} 0 {L/2*np.sin(a) - 0.1/np.cos(a)}" size="{L/2} 3 0.1" euler="0 {-a} 0"/>' if pct > 0 else ""
        m, d, rb = make(P, extra); ll = LowLevel(P)
        log = []
        drive(m, d, rb, ll, lambda t, s: 0.6, 26, cb=lambda t, s: log.append((s["pos"][0], s["v"], s["th_y"], abs(s["pod_pitch"]))))
        L_ = np.array(log)
        on = (L_[:, 0] > 4) & (L_[:, 0] < 12)
        ok = bool(on.sum() > 0 and L_[:, 0].max() > 9 and np.percentile(L_[on, 1], 10) > 0.35)
        res.append(dict(grade_pct=pct, climbs=ok,
                        pendulum_deg=round(float(np.degrees(np.median(L_[on, 2]))) if on.any() else float("nan"), 1),
                        max_pod_tilt_deg=round(float(np.degrees(L_[on, 3].max())) if on.any() else float("nan"), 1)))
    return res


# ----------------------------------------------------------------------------- 2. lips / steps
def test_steps(P):
    res = []
    for h_mm in (5, 8, 10, 15, 20, 25, 30, 40, 50):
        h = h_mm / 1000
        row = dict(height_mm=h_mm)
        for v in (0.3, 0.7, 1.0, 1.4):
            extra = f'<geom type="box" pos="4.5 0 {h/2}" size="1.5 3 {h/2}"/>'
            m, d, rb = make(P, extra); ll = LowLevel(P)
            pk = [0.0]
            def cb(t, s):
                if t > 0.5:   # skip the initial settle
                    pk[0] = max(pk[0], float(np.linalg.norm(d.sensor("pod_acc").data - [0, 0, 9.81])))
                return s["pos"][0] < 4.0
            drive(m, d, rb, ll, lambda t, s: v, 16, cb=cb)
            row[f"v{v}"] = dict(passes=bool(d.qpos[0] > 3.4), peak_g=round(pk[0] / 9.81, 2))
        res.append(row)
    return res


# ----------------------------------------------------------------------------- 3. braking
def test_braking(P):
    m, d, rb = make(P); ll = LowLevel(P)
    rec = {}
    def vfn(t, s):
        return 1.4 if t < 8 else 0.0
    def cb(t, s):
        if t >= 8 and "x0" not in rec:
            rec["x0"] = s["pos"][0]; rec["v0"] = s["v"]
        if t >= 8:
            rec["soup"] = max(rec.get("soup", 0), soup_tilt(d))
            rec["pod"] = max(rec.get("pod", 0), abs(np.degrees(s["pod_pitch"])))
            if abs(s["v"]) < 0.02 and "x1" not in rec and t > 8.5:
                rec["x1"] = s["pos"][0]; rec["t1"] = t
    ll.dec_lim = 0.9
    drive(m, d, rb, ll, vfn, 16, cb=cb)
    return dict(from_mps=round(rec["v0"], 2), stop_dist_m=round(rec["x1"] - rec["x0"], 2), stop_time_s=round(rec["t1"] - 8, 2),
                max_soup_tilt_deg=round(rec["soup"], 1), max_pod_tilt_deg=round(rec["pod"], 1))


# ----------------------------------------------------------------------------- 4. turning
def test_turning(P):
    out = []
    for v in (0.3, 0.5, 0.8, 1.1, 1.4):
        m, d, rb = make(P); ll = LowLevel(P)
        pts = []
        drive(m, d, rb, ll, lambda t, s: v, 22, yaw_fn=lambda t, s: (1.0 if t > 6 else 0.0),
              cb=lambda t, s: pts.append((t, s["pos"][0], s["pos"][1], s["yaw_rate"], s["lean"])) or None)
        A = np.array(pts); sel = A[:, 0] > 12
        # circle fit
        x, y = A[sel, 1], A[sel, 2]
        M = np.stack([x, y, np.ones_like(x)], 1)
        sol = np.linalg.lstsq(M, -(x * x + y * y), rcond=None)[0]
        r = float(np.sqrt(sol[0] ** 2 / 4 + sol[1] ** 2 / 4 - sol[2]))
        out.append(dict(speed_mps=v, turn_radius_m=round(r, 2), yaw_rate_deg_s=round(float(np.degrees(np.mean(A[sel, 3]))), 1),
                        lean_deg=round(float(np.degrees(np.mean(np.abs(A[sel, 4])))), 1),
                        model_radius_m=round(LowLevel(P).min_turn_radius(v), 2)))
    return out


# ----------------------------------------------------------------------------- 5. parked on cross-slope with hatch open
def test_park(P):
    out = []
    for pct in (0, 2, 5, 8):
        a = np.arctan(pct / 100)
        m = mujoco.MjModel.from_xml_string(model.standalone_xml(P).replace(
            '<geom name="floor" type="plane" size="60 60 0.1"', f'<geom name="floor" type="plane" size="60 60 0.1" euler="{a} 0 0"'))
        d = mujoco.MjData(m); rb = Robot(m, P)
        # settle with brake on, then legs down, open lid, wait
        eqb, eqy = m.equality("brake_band").id, m.equality("brake_yoke").id
        for _ in range(250):
            mujoco.mj_step(m, d)
        m.eq_data[eqb][0] = d.qpos[rb.qa["pod_pitch"]]; m.eq_data[eqy][0] = d.qpos[rb.qa["drive"]]
        d.eq_active[eqb] = 1; d.eq_active[eqy] = 1
        p0 = d.qpos[:3].copy()
        for i in range(int(8 / m.opt.timestep)):
            for k in range(4):
                d.ctrl[rb.act[f"leg_{k+1}"]] = P.geo.leg_stroke
            if d.time > 2.0:
                d.ctrl[rb.act["lid"]] = max(-1.95, -1.95 * (d.time - 2.0) / 1.5)
            mujoco.mj_step(m, d)
        drift = float(np.linalg.norm(d.qpos[:2] - p0[:2]))
        s = rb.state(d)
        out.append(dict(cross_slope_pct=pct, drift_mm=round(drift * 1000, 1), pod_tilt_deg=round(abs(np.degrees(s["pod_roll"])), 1),
                        stable=bool(drift < 0.03)))
    return out


# ----------------------------------------------------------------------------- 6. energy
def test_energy(P):
    """Flat cruise at 1.4 m/s for 60 s: Wh/km including electronics, and runtimes."""
    m, d, rb = make(P); ll = LowLevel(P)
    E = P.e
    acc = dict(drive=0.0, steer=0.0, level=0.0)
    x = []
    def cb(t, s):
        if t < 10:
            return
        dt = m.opt.timestep
        tau = d.ctrl[rb.act["drive"]]; w = d.qvel[rb.va["drive"]]
        acc["drive"] += (max(tau * w, 0) / E.drivetrain_eff + 0.025 * tau * tau) * dt
        ts = d.actuator_force[rb.act["steer"]]; ws = d.qvel[rb.va["steer"]]
        acc["steer"] += (abs(ts * ws) / 0.6 + 0.05 * ts * ts + 0.3) * dt
        tl = d.ctrl[rb.act["pod_level"]]
        acc["level"] += (0.2 * tl * tl) * dt
        x.append(s["pos"][0])
    drive(m, d, rb, ll, lambda t, s: 1.4, 70, yaw_fn=lambda t, s: 0.05 * np.sin(t * 0.7), cb=cb)
    dur = 60.0
    dist = x[-1] - x[0]
    hotel = sum(E.hotel_w.values())
    p_motion = (acc["drive"] + acc["steer"] + acc["level"]) / dur
    p_drive_total = p_motion + hotel
    usable = E.battery_wh * E.usable_frac
    return dict(cruise_speed_mps=1.4, drive_power_W=round(acc["drive"] / dur, 1), steer_power_W=round(acc["steer"] / dur, 1),
                hotel_power_W=round(hotel, 1), total_while_driving_W=round(p_drive_total, 1),
                Wh_per_km=round(p_drive_total * dur / 3600 / (dist / 1000), 1),
                runtime_continuous_driving_h=round(usable / p_drive_total, 1),
                range_continuous_km=round(usable / p_drive_total * 1.4 * 3.6, 1),
                runtime_typical_duty_h=round(usable / (0.5 * p_drive_total + 0.5 * (hotel - 6.0)), 1),
                duty_note="typical duty = 50 % driving, 50 % parked/waiting with compute throttled (-6 W)")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    g = results["grade"]
    ax[0].plot([r["grade_pct"] for r in g], [r["pendulum_deg"] for r in g], "o-", color="#2a6fdb")
    for r in g:
        ax[0].annotate("ok" if r["climbs"] else "fail", (r["grade_pct"], r["pendulum_deg"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    fails = [r["grade_pct"] for r in g if not r["climbs"]]
    ax[0].set_xlabel("grade (%)"); ax[0].set_ylabel("pendulum swing to climb (deg)")
    ax[0].set_title(f"Hill climbing, 5 kg food (stalls at {min(fails):g}%)" if fails else "Hill climbing, 5 kg food")
    ax[0].axvline(8.33, color="#999", ls="--", lw=1); ax[0].text(8.5, 2, "ADA ramp 1:12", fontsize=8, color="#666")
    st = results["steps"]
    speeds = ["0.3", "0.7", "1.0", "1.4"]
    mx, pg = [], []
    for v in speeds:
        ok = [r["height_mm"] for r in st if r["v" + v]["passes"]]
        h = max(ok) if ok else 0
        mx.append(h)
        pg.append(next((r["v" + v]["peak_g"] for r in st if r["height_mm"] == h), 0))
    ax[1].bar(speeds, mx, color="#2a6fdb", alpha=0.85)
    for i, (h, g) in enumerate(zip(mx, pg)):
        ax[1].text(i, h + 1, f"{h} mm\n{g:.1f} g at pod", ha="center", fontsize=8)
    ax[1].set_xlabel("approach speed (m/s)"); ax[1].set_ylabel("highest lip it rolls over (mm)")
    ax[1].set_title("Sidewalk lips need momentum"); ax[1].set_ylim(0, max(mx) * 1.3 + 5)
    tr = results["turning"]
    ax[2].plot([r["speed_mps"] for r in tr], [r["turn_radius_m"] for r in tr], "o-", color="#2a6fdb", label="simulated")
    ax[2].plot([r["speed_mps"] for r in tr], [r["model_radius_m"] for r in tr], "--", color="#999", label="R/tan(lean) model")
    ax[2].set_xlabel("speed (m/s)"); ax[2].set_ylabel("min turn radius (m)"); ax[2].set_title("Lean steering: turns like a bike"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "tests.json"))), os.path.join(OUT, "tests.png")); sys.exit()
    P = Params()
    results = {}
    for name, fn in (("grade", test_grade), ("steps", test_steps), ("braking", test_braking), ("turning", test_turning),
                     ("park", test_park), ("energy", test_energy)):
        print("running", name, flush=True)
        results[name] = fn(P)
        print(json.dumps(results[name], indent=1), flush=True)
    results["mass_kg"] = dict(total_loaded=round(P.total_mass() + 0.48, 2), payload=P.m.payload,
                              pendulum_moment_kgm=round(P.pendulum_moment(), 3), static_grade_limit_deg=round(P.max_static_grade(), 1))
    with open(os.path.join(OUT, "tests.json"), "w") as f:
        json.dump(results, f, indent=1)
    plot(results, os.path.join(OUT, "tests.png"))
    print("saved")
