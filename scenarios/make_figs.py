"""Figures for the v2 report: curb filmstrips, stress-test dashboard, v1 vs v2.
python scenarios/make_figs.py   (needs out/stress.json, out_v2/curb_*.mp4)"""
import os, json
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out_v2")
S = json.load(open(os.path.join(ROOT, "out", "stress.json")))
INK, MUTED, ACC, ACC2, BAD = "#1f2933", "#7b8794", "#2f80ed", "#27ae60", "#d64545"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def filmstrip(mp4, out_png, n=6, title=""):
    frames = [f for f in imageio.get_reader(mp4)]
    # skip the long approach: keep the part where the legs are working
    idx = np.linspace(int(len(frames) * 0.38), len(frames) - 1, n).astype(int)
    pics = [frames[i][40:330, 120:560] for i in idx]
    fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.1))
    for a, p in zip(ax, pics):
        a.imshow(p); a.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def stress_dashboard(out_png):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    # grade
    g = S["grade"]
    up = [(r["grade_pct"], r["up_pendulum_deg"]) for r in g if r["up_ok"]]
    dn = [(r["grade_pct"], r.get("down_food_g")) for r in g if r.get("down_ok")]
    ax[0, 0].plot([u[0] for u in up], [u[1] for u in up], "o-", color=ACC, label="climbing: pendulum swing used (deg)")
    fails = [r["grade_pct"] for r in g if not r["up_ok"]]
    ax[0, 0].plot(fails, [62] * len(fails), "x", color=BAD, ms=9, label="can't climb / can't hold 1 m/s down")
    ax[0, 0].axhline(60, color=MUTED, ls=":"); ax[0, 0].text(0, 56, "swing limit", color=MUTED, fontsize=8)
    ax[0, 0].axvline(8.33, color=MUTED, ls=":"); ax[0, 0].text(8.6, 5, "ADA ramp", color=MUTED, fontsize=8)
    ax[0, 0].set_xlabel("grade (%)"); ax[0, 0].set_ylim(0, 70)
    ax[0, 0].set_title(f"Hills, 5 kg food: up and down to {max(u[0] for u in up):g}%", loc="left"); ax[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    # turning
    t = S["turning"]
    ax[0, 1].plot([r["speed_mps"] for r in t], [r["turn_radius_m"] for r in t], "o-", color=ACC, label="measured")
    ax[0, 1].plot([r["speed_mps"] for r in t], [r["planner_radius_m"] for r in t], "--", color=MUTED, label="planner's model")
    ax[0, 1].set_xlabel("speed (m/s)"); ax[0, 1].set_ylabel("tightest turn radius (m)")
    ax[0, 1].set_title("Turning", loc="left"); ax[0, 1].legend(frameon=False, fontsize=8)
    # braking
    b = S["braking"]
    for em, c, lab in ((False, ACC, "normal stop"), (True, BAD, "emergency stop")):
        rr = [r for r in b if r["emergency"] == em]
        ax[0, 2].plot([r["from_mps"] for r in rr], [r["stop_dist_m"] for r in rr], "o-", color=c, label=lab)
    ax[0, 2].set_xlabel("from speed (m/s)"); ax[0, 2].set_ylabel("stopping distance (m)")
    ax[0, 2].set_title("Braking (food ≤ 0.3 g)", loc="left"); ax[0, 2].legend(frameon=False, fontsize=8)
    # wind lane keeping
    wl = S["wind"]["lane"]
    for U, c in ((6.0, ACC2), (10.0, ACC), (14.0, BAD)):
        rr = [r for r in wl if r["mean_mps"] == U]
        ax[1, 0].plot([r["speed_mps"] for r in rr], [r["max_lateral_m"] for r in rr], "o-", color=c,
                      label=f"{U:.0f} m/s mean (gusts to {max(r['peak_gust_mps'] for r in rr):.0f})")
    ax[1, 0].set_yscale("log"); ax[1, 0].axhline(0.3, color=MUTED, ls=":")
    ax[1, 0].text(0.82, 0.33, "0.3 m lane budget", color=MUTED, fontsize=8)
    ax[1, 0].set_xlabel("speed (m/s)"); ax[1, 0].set_ylabel("worst sideways drift (m)")
    ax[1, 0].set_title("Crosswind while driving", loc="left"); ax[1, 0].legend(frameon=False, fontsize=8)
    # curbs
    cr = S["curb_robustness"]
    rows = cr["rows"]
    cats = [("12 cm up", True, 0.12), ("15 cm up", True, 0.15), ("12 cm down", False, 0.12), ("15 cm down", False, 0.15)]
    okr = [np.mean([r["ok"] for r in rows if r["up"] == u and r["h"] == h]) for _, u, h in cats]
    safe = [np.mean([r.get("safe", r["ok"]) for r in rows if r["up"] == u and r["h"] == h]) for _, u, h in cats]
    xx = np.arange(len(cats))
    ax[1, 1].bar(xx, safe, 0.6, color="#cfe0fb", label="safe outcome (done, or refused/backed off)")
    ax[1, 1].bar(xx, okr, 0.6, color=ACC, label="completed")
    ax[1, 1].set_xticks(xx); ax[1, 1].set_xticklabels([c[0] for c in cats]); ax[1, 1].set_ylim(0, 1.05)
    ax[1, 1].set_title(f"Curbs, randomised robots (n={cr['trials']})", loc="left"); ax[1, 1].legend(frameon=False, fontsize=8, loc="lower left")
    # monte carlo
    mc = S["montecarlo"]["pass_rates"]
    names = {"scurve_ok": "S-curve", "estop_ok": "E-stop ≤1.6 m", "wind_ok": "Wind 8+6 m/s", "ramp_ok": "12% ramp climb", "all_ok": "All four"}
    ax[1, 2].barh(list(names.values())[::-1], [mc[k] for k in list(names)[::-1]], color=ACC2)
    ax[1, 2].set_xlim(0, 1.05); ax[1, 2].set_title(f"Monte Carlo: {S['montecarlo']['n']} randomly built robots", loc="left")
    ax[1, 2].set_xlabel("pass rate")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def v1_v2(out_png):
    v1 = json.load(open(os.path.join(ROOT, "out", "delivery_summary.json"))) if False else None
    items = [("Cruise speed (m/s)", 1.4, 2.0), ("Normal stop (m)\nv1 from 1.5, v2 from 1.4 m/s", 1.81, S["braking"][0]["stop_dist_m"]),
             ("Steepest climb, loaded (%)", 14, max(r["grade_pct"] for r in S["grade"] if r["up_ok"])),
             ("Curb height handled (cm)", 0, 15), ("Runtime, driving (h)", 12.8, S["energy"][0]["runtime_driving_h"]),
             ("BOM (USD, hundreds)", 16.5, 21.2)]
    fig, ax = plt.subplots(1, len(items), figsize=(15, 2.8))
    for a, (lab, a1, a2) in zip(ax, items):
        a.bar([0, 1], [a1, a2], color=[MUTED, ACC], width=0.6)
        a.set_xticks([0, 1]); a.set_xticklabels(["v1", "v2"]); a.set_title(lab, fontsize=9, loc="left")
        for x, v in ((0, a1), (1, a2)):
            a.text(x, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        a.set_yticks([])
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


if __name__ == "__main__":
    filmstrip(os.path.join(OUT, "curb_up_12cm.mp4"), os.path.join(OUT, "curb_up.png"), title="Up a 12 cm curb: lift on the legs, roll to the edge, tuck the front legs, roll up")
    filmstrip(os.path.join(OUT, "curb_down_12cm.mp4"), os.path.join(OUT, "curb_down.png"), title="Down a 12 cm curb: front legs reach down, cart forward, kneel, land")
    stress_dashboard(os.path.join(OUT, "stress.png"))
    v1_v2(os.path.join(OUT, "v1_v2.png"))
    print("figures written to", OUT)
