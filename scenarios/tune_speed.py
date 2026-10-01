"""Sweep the speed-loop gains on the realistic rig (delayed/noisy sensors):
step 0 -> 2.0 m/s, cruise, brake to 0. Score = pendulum wobble + speed error."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from orb.harness import Rig
from orb.params import Params

P = Params()
def trial(Kp, Kd, Kv, Ki, tau_d=0.3, vmax=2.0, verbose=False):
    r = Rig(P, realistic=True, seed=1)
    r.ll.Kp_th, r.ll.Kd_th, r.ll.Kv, r.ll.Ki_v = Kp, Kd, Kv, Ki
    log = []
    vf = lambda t, s: vmax if t < 10 else 0.0
    r.run(15, vf, cb=lambda t, s: log.append((t, s["v"], s["th_y"], abs(s["lean"]))))
    L = np.array(log)
    cru = (L[:, 0] > 6) & (L[:, 0] < 10)
    wob = np.degrees(np.std(L[cru, 2]))
    verr = np.sqrt(np.mean((L[cru, 1] - vmax) ** 2))
    stop = L[L[:, 0] > 13.5, 1]
    lean = np.degrees(L[:, 3].max())
    return wob, verr, float(np.abs(stop).max()), lean
res = []
for Kp, Kd, Kv, Ki in itertools.product((95, 150, 220), (11, 18, 28), (0.5, 0.8), (0.2, 0.4)):
    w, e, st, ln = trial(Kp, Kd, Kv, Ki)
    sc = w + 20 * e + 10 * st
    res.append((sc, w, e, st, ln, Kp, Kd, Kv, Ki))
    print("score %6.2f wobble %5.2f deg  verr %.3f  stop-resid %.3f lean %.1f | Kp %g Kd %g Kv %g Ki %g" % res[-1], flush=True)
res.sort()
print("BEST")
for r in res[:6]:
    print("score %6.2f wobble %5.2f deg  verr %.3f  stop-resid %.3f lean %.1f | Kp %g Kd %g Kv %g Ki %g" % r)
