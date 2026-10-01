# ZEBRA: a campus food-delivery sphere, physically simulated in MuJoCo

Zebra is a 650 mm glowing ball that rolls down campus sidewalks, sees with its own
cameras, waits for cars at crosswalks, yields to people, and opens like a
clamshell at the dorm door. Everything here is simulated physics: the drive,
steering, food pod, cameras, perception, localisation and decision logic all
run closed-loop in MuJoCo.

## Run it

```bash
pip install mujoco numpy scipy opencv-python pandas matplotlib imageio imageio-ffmpeg
python view.py              # watch the full autonomous delivery live (macOS: use `mjpython view.py`)
python view.py --teleop     # drive it yourself (arrows, space, O = hatch)
python run_delivery.py      # headless: renders out/delivery.mp4 + out/delivery_summary.json
python scenarios/test_suite.py   # engineering tests -> out/tests.json + out/tests.png
```

`run_delivery.py --payload 0` runs it empty; `--seed N` changes sensor noise.

## How the robot works

| Part | What it does |
|---|---|
| **Rolling band** (centre ±30° of the sphere) | The only part that touches the ground. Translucent, with a TPU tread and a flush load-bearing hatch. |
| **Side caps** (smoked, stationary) | Never rotate or touch the ground, so the camera lenses stay clean. The band rides on roller tracks on the caps (no through-axle, so the centre is free for food). |
| **Drive pendulum** | Motors, compute, ballast. Torque between it and an internal ring gear rolls the band. |
| **Battery pendulum** (3 kg, swings sideways) | Leans the whole ball; a leaning ball turns like a bike at `yaw rate = v·tan(lean)/R`. |
| **Food pod** (~19 L, insulated) | Hangs on a 2-axis gimbal: pitch is actively leveled by a tiny motor, roll swings free and damped. Food stays within a few degrees of level. |
| **Eyes** | LED panel on the level frame shines through the band, so the face stays forward while the ball rolls. Blinks, looks at people it yields to, squints happily at delivery. |
| **Park** | Spring brake locks band to frame; 4 legs drop from the caps; the robot rolls to put the hatch exactly on top before parking. |

## Onboard software (`orb/`)

| File | Role |
|---|---|
| `params.py` | Every physical number (masses, geometry, friction, motors, power budget). |
| `model.py`, `meshes.py` | Robot MJCF, procedural meshes (band with hatch cut-out, caps, pod). |
| `world.py` | Campus: sidewalks, raised plaza with 1:12 ADA ramps, road with 12 cm curbs + curb ramps + crosswalk, 1–2 cm heaved slabs, bins, scooter, scripted pedestrians and cars. |
| `control.py` | Speed loop (speed → pendulum angle → torque, with accel feed-forward and a slope observer); lean-steering loop with roll damping; pod leveling. |
| `perception.py` | Renders the cap cameras (RGB + noisy stereo depth), classifies pavement/grass/asphalt/paint, builds a bird's-eye grid with obstacles and curb/step edges. |
| `planner.py` | Route with rounded corners, curvature speed limits, map-relative localisation (odometry + GNSS + sidewalk-edge and pavement-end fixes), obstacle-aware lane offset, pedestrian tracking and prediction, crosswalk gate, stall recovery (back up and go around, or take a lip with a run-up), delivery state machine. |
| `sim.py` | Closed loop, energy accounting, metrics, chase-camera video with camera feeds and perception overlay. |

`scenarios/` holds the tuning and calibration scripts used to get here
(steering map, roll-stability sweeps, ramp speed control).
