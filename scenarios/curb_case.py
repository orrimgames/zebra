"""Re-run one curb_robustness case: python curb_case.py i up h [-v]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from orb.harness import perturbed
from curb_test import curb_run

def case(i):
    rng = np.random.default_rng(7)
    for k in range(i + 1):
        Pp = perturbed(rng, level=0.7); mu = float(rng.uniform(0.55, 0.9))
    return Pp, mu

if __name__ == "__main__":
    i, up, h = int(sys.argv[1]), sys.argv[2] == "up", float(sys.argv[3])
    Pp, mu = case(i)
    r = curb_run(h, up, P=Pp, seed=100 + i, edge_noise=0.06, mu_edge=mu, verbose="-v" in sys.argv,
                 video=sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None)
    print(json.dumps({k: v for k, v in r.items()}, default=str))
