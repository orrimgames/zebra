"""Fit the drive model: acceleration vs. pendulum angle (-> M_EFF) and the
rolling-resistance angle. Holds the pendulum at fixed world angles with a
stiff PD loop and measures the ball's acceleration from rest."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, mujoco
from orb.harness import Rig
from orb.params import Params

P = Params()
res = []
for deg in (4, 8, 12, 16):
    r = Rig(P, realistic=False)
    th0 = np.radians(deg)
    log = []
    def drive_ov(t, s, c):
        u = r.ll.mgl * np.sin(th0) + 120 * (th0 - s["th_y"]) - 14 * s["th_y_rate"]
        return -float(np.clip(u, -40, 40))
    r.drive_override = drive_ov
    r.run(2.5, lambda t, s: 0.0, cb=lambda t, s: log.append((t, s["v"], s["th_y"])))
    L = np.array(log); sel = (L[:, 0] > 0.6) & (L[:, 0] < 2.3)
    a = np.polyfit(L[sel, 0], L[sel, 1], 1)[0]
    Meff = r.ll.mgl * np.sin(th0) / (P.geo.R * a)
    res.append((deg, a, Meff))
    print(f"pendulum {deg:2d} deg -> a = {a:.3f} m/s^2  -> M_eff(no RR) = {Meff:.1f} kg")
# rolling resistance: angle needed at steady 1.5 m/s
r = Rig(P, realistic=False); log = []
r.run(12, lambda t, s: 1.5, cb=lambda t, s: log.append((t, s["th_y"], s["v"])))
L = np.array(log); th_rr = np.median(L[L[:, 0] > 8, 1])
F_rr = r.ll.mgl * np.sin(th_rr) / P.geo.R
print(f"steady 1.5 m/s: pendulum {np.degrees(th_rr):.2f} deg -> resist {F_rr:.2f} N (Crr={F_rr/(P.total_mass()*9.81):.4f})")
# corrected M_eff
for deg, a, _ in res:
    th0 = np.radians(deg)
    print(f"  {deg} deg corrected M_eff = {(r.ll.mgl*np.sin(th0) - F_rr*P.geo.R)/(P.geo.R*a):.1f}")
