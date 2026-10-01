# ZEBRA v2: a campus food-delivery sphere, physically simulated in MuJoCo

Zebra is a 650 mm glowing ball that rolls down campus sidewalks at up to 2 m/s, sees with
its own cameras and radar, waits for cars, yields to people, climbs 3–15 cm curbs
on four little legs when a curb ramp is blocked, rides out wind, and opens like a
clamshell at the dorm door. Everything is simulated physics, run closed-loop:
the drive, steering, legs, food pod, sensors (with delay, noise and bias),
perception, localisation, planning and an independent safety computer.

## Run it

```bash
pip install mujoco numpy scipy opencv-python pandas matplotlib imageio imageio-ffmpeg
python view.py                          # watch the autonomous delivery live (macOS: mjpython view.py)
python run_delivery.py --T 260 --out out_v2      # headless: out_v2/delivery.mp4 + delivery_summary.json
python run_delivery.py --no-video --start 52 23.9 0   # start part-way (here: 8 m from the door)
python scenarios/stress_suite.py        # every stress test -> out/stress.json  (~25 min on 1 core)
python scenarios/stress_suite.py curbs curb_robustness   # or just some sections
python scenarios/curb_test.py           # curb climbs 5/10/15 cm, up and down
python scenarios/curb_case.py 2 up 0.12 -v   # replay one randomised curb trial, verbose
python scenarios/make_figs.py           # figures in out_v2/
```

Stress-suite sections: `grade lips curbs curb_robustness braking turning wind park energy montecarlo`.

## What changed from v1

| v1 | v2 | Why |
|---|---|---|
| 1.4 m/s cruise | 2.0 m/s cruise, slows itself for people, corners, blind spots, wind | faster deliveries without being the scary thing on the sidewalk |
| 3 kg Li-ion steering pendulum, ±28° | 5.8 kg steering tray around a 5.2 kg LiFePO4 12S3P pack, ±32°, at 0.255 m | 2.1x the lean moment: steering authority for 2 m/s and crosswinds; LFP is the safe chemistry; 690 Wh |
| no curbs (ramps only) | 4 telescoping series-elastic legs with driven wheels: "cart mode" climbs 3–15 cm curbs up and down | blocked or missing curb ramps are everyday campus life |
| rigid pod | pod on 7 Hz elastomer isolation | 5 cm lips gave 3–9 g (raw peaks); now 1–2.6 g over 15–30 mm (10 ms-filtered, so not strictly like-for-like) |
| cameras + ToF | + 2 x 24 GHz radar (crosswalk gap acceptance), 2 down-looking cliff ToFs, rear view | sees cars before they're visible, finds drop-off edges to ~1 cm |
| one computer | + independent safety MCU (own ToF, tilt, heartbeat watchdog) | stops the robot if the main computer hangs or misjudges |
| Jetson Orin Nano | Raspberry Pi 5 + AI HAT+ (Hailo-8) | the Jetson went to $399; Pi+Hailo is cheaper and lower power |
| band imbalance from the hatch | 0.41 kg counterweight opposite the hatch | removed a 0.67 Hz wobble at speed |
| legs pushed blindly at park (sim bug: they never touched down) | stop on the speed loop → brake → legs find the ground one by one → equal preload | parks on 12% cross-slopes and in 25 m/s gusts |

## Onboard software (`orb/`)

| File | Role |
|---|---|
| `params.py` | Every physical number (masses, geometry, friction, motors, legs, aero, power). |
| `model.py`, `meshes.py` | Robot MJCF (band, counterweight, lid, yoke + steering tray, caps with sensors, isolated pod, legs + wheels). |
| `world.py` | Campus: sidewalks, plaza with ADA ramps, road with 12 cm curbs, a curb ramp blocked by a fallen scooter, hedge blind spot, pedestrians, a child, a bike coming from behind, cars. |
| `control.py` | Speed loop (pendulum cascade with slope/push observer), lean steering with speed-scheduled gains and braking fade, pod leveling, leg control (force mode, parking touchdown, wheel locks), sensor model (delay/noise/bias). |
| `curb.py` | Curb climber state machine (up: lift → cart → handoff → climb → mount; down: legs → cart → kneel → settle → lower → land), contact- and cliff-ToF-based edge finding, traction control, surge catch, watchdogs. |
| `perception.py` | Cap cameras (RGB + noisy depth), classification, bird's-eye grid to 9 m with obstacles, steps and curbs. |
| `planner.py` | Route, localisation, "common sense" speed policy, people tracking/prediction, keep-right and rear-bike etiquette, crosswalk radar gap acceptance, curb-ramp blocked → curb climb, wind brace, delivery state machine. |
| `safety.py` | Independent safety MCU: ToF rays, tilt, planner heartbeat; latches an emergency stop. |
| `wind.py` | Gauss-Markov turbulence + gusts; sphere drag and lid sail. |
| `harness.py` | Test rig with the real control loop at 250 Hz on delayed/noisy sensors; `perturbed()` builds randomly-wrong robots for Monte Carlo. |
| `sim.py` | Full closed loop, video with overlays (wind gauge, safety status), energy, metrics. |

## Headline results (sim)

See `out/stress.json` and the design spec for the full set.

- Turning radius 1.56 m at 0.4 m/s, 6.4 m at 2 m/s (planner model within 5%).
- Emergency stop 0.83 m from 1.4 m/s, 1.30 m from 2.0 m/s, food ≤0.3 g.
- Climbs and descends 22% grades with 5 kg; fails at 25%.
- Curbs 3–15 cm, up and down: 10/10 nominal; 60 randomised trials (wet curbs, ±3 cm edge error, 0–7 kg, built-robot spread): 90% completed, 98% safe outcome, worst food 2.1 g.
- Wind: full speed up to 5 m/s mean (planner caps 1.7 m/s at 6, 1.4 at 7, 1.0 at 9.5); ≤1.4 m/s holds the lane (0.16 m) at 10 m/s mean with 16 m/s gusts; braces (stops, legs down) above 10 m/s mean or 15 m/s gusts. Parked and braced: no tip-over in 25 m/s gusts from any side, 30 m/s side-on.
- 40 randomly built robots (mass, CG, friction, motor torque, 4–16 ms sensor delay): 100% pass the S-curve, e-stop, gusty-lane and 12% ramp-climb tests.
- 19 h continuous driving at 1.4 m/s (30.7 W), ~26 h typical day.
- Full campus delivery (`out_v2/delivery.mp4`): 85 m in 157 s through a child stepping out from
  behind a hedge (emergency stop, 0.87 m clearance), oncoming walkers, a bike from behind, a crosswalk
  with two cars (radar gap acceptance) and a curb ramp blocked by a fallen scooter (lined up and
  climbed the 12 cm curb next to it in 16 s). 0 contacts, peak food 0.91 g, gusts to 13 m/s.

## Honest caveats

Perception in the sim classifies pixels by colour; the real robot needs a learned
segmentation/stereo network trained and validated on real campus footage. Pedestrians
follow scripts. Contacts, tread, ground friction and the roller-track drag are modelled,
not measured. The randomised tests make the controllers robust to being wrong about these
things, but every number is a target to verify on hardware (see the sim-to-real plan in the spec).
