"""Open-loop steering map: tray swing -> steady lean -> yaw rate, at several
speeds. Fits cff(v) = A + B v^2 (swing needed per unit lean)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from orb.harness import Rig
from orb.params import Params

P = Params()
rows = []
for v in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4):
    for sw in (6, 12, 18):
        r = Rig(P, realistic=False)
        r.steer_override = lambda t, s, sw=sw: np.radians(sw) * min(1.0, max(0.0, (t - 3.0) / 0.5))
        log = []
        r.run(9.0, lambda t, s: v, cb=lambda t, s: log.append((t, s["lean"], s["yaw_rate"], s["v"])))
        L = np.array(log); sel = L[:, 0] > 6.5
        lam = -np.mean(L[sel, 1]); yr = np.mean(L[sel, 2]); vm = np.mean(L[sel, 3])
        rows.append((v, sw, lam, yr, vm))
        print(f"v={v:.1f} swing={sw:2d} deg  lean={np.degrees(lam):5.2f} deg  yaw={yr:+.3f}  radius={vm/max(abs(yr),1e-6):6.2f} m  ratio={np.radians(sw)/max(lam,1e-6):5.2f}", flush=True)
A = np.array(rows)
ratio = np.radians(A[:, 1]) / np.maximum(A[:, 2], 1e-6)
M = np.stack([np.ones(len(A)), A[:, 4] ** 2], 1)
coef = np.linalg.lstsq(M, ratio, rcond=None)[0]
print("cff(v) = %.3f + %.3f v^2" % tuple(coef))
