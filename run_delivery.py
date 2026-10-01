"""Run the full campus delivery: dining hall -> sidewalks -> plaza ramps ->
crosswalk -> dorm door, with onboard camera perception, then park, open the
hatch and hand off the food.  Writes out/delivery.mp4 + metrics."""
import os, sys, json, argparse
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")   # headless Linux; desktops use the default GL
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orb.sim import Sim

ap = argparse.ArgumentParser()
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--T", type=float, default=170)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--payload", type=float, default=5.0, help="food mass kg")
ap.add_argument("--out", default="out")
ap.add_argument("--start", type=float, nargs=3, default=None, help="x y yaw (debug: start part-way)")
a = ap.parse_args()
sim = Sim(video=not a.no_video, seed=a.seed, out_dir=a.out, payload=a.payload, start=a.start)
summ = sim.run(T=a.T)
print(json.dumps(summ, indent=1, default=float))
