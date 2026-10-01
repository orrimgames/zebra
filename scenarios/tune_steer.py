"""Steering-loop tuning on the realistic rig: line following in a gusty
crosswind + turn-in steps. Prints the best gain sets."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from orb.harness import Rig
from orb.params import Params
from orb.wind import Wind

P = Params()

def pursuit(r, v_ref, y_line=0.0):
    def yaw(t, s):
        st = r.rb.state(r.d)
        Ld = 0.9 + 0.7 * max(abs(st["v"]), 0.3)
        a = np.arctan2(y_line - st["pos"][1], Ld) - st["heading"]
        yc = 2 * max(abs(st["v"]), 0.3) * np.sin(a) / Ld
        mx = r.ll.max_yaw_rate(max(abs(st["v"]), 0.3))
        return float(np.clip(yc, -mx, mx))
    return yaw

def line_wind(g, v, U=10.0, gust=8.0, seed=2):
    w = Wind(U, np.pi / 2, turb=0.15, gusts=[(9.0, 3.0, gust, 0.0)], seed=seed)
    r = Rig(P, realistic=True, seed=seed, wind=w)
    r.ll.steer_Kp, r.ll.steer_Ki, r.ll.steer_Kd, r.ll.yaw_slew = g
    log = []
    r.run(16, lambda t, s: v, yaw_fn=pursuit(r, v), cb=lambda t, s: log.append((t, s["pos"][1], s["lean"], s["lean_rate"])))
    L = np.array(log); sel = L[:, 0] > 3
    return np.abs(L[sel, 1]).max(), np.degrees(np.abs(L[sel, 2]).max()), np.std(L[sel, 3])

def turn(g, v):
    r = Rig(P, realistic=True, seed=3)
    r.ll.steer_Kp, r.ll.steer_Ki, r.ll.steer_Kd, r.ll.yaw_slew = g
    mx = r.ll.max_yaw_rate(v)
    log = []
    r.run(12, lambda t, s: v, yaw_fn=lambda t, s: (mx if 4 < t < 9 else 0.0), cb=lambda t, s: log.append((t, s["yaw_rate"], s["lean"])))
    L = np.array(log)
    on = (L[:, 0] > 6.5) & (L[:, 0] < 9)
    trk = np.sqrt(np.mean((L[on, 1] - mx) ** 2)) / mx
    after = L[L[:, 0] > 10.5]
    return trk, np.degrees(np.abs(after[:, 2]).max())

if __name__ == "__main__":
  res = []
  for g in itertools.product((0.4, 0.8, 1.5), (0.6, 1.2, 2.5), (0.6, 1.0, 1.6), (0.6, 1.0)):
      sc, parts = 0.0, []
      for v in (0.8, 2.0):
          dy, ln, lr = line_wind(g, v)
          tk, res_lean = turn(g, v)
          sc += 3 * dy + 0.05 * ln + 2 * lr + tk + 0.2 * res_lean
          parts += [dy, ln, tk]
      res.append((sc, g, parts))
      print("%.3f" % sc, g, " ".join("%.3f" % p for p in parts), flush=True)
  res.sort(key=lambda x: x[0])
  print("BEST")
  for r in res[:6]:
      print("%.3f" % r[0], r[1], " ".join("%.3f" % p for p in r[2]))
