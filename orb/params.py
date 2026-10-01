"""All physical parameters for the ORB campus delivery sphere.

Units: SI (m, kg, s, N, rad). Every number the sim uses lives here so the
design can be re-tuned from one place and re-verified by the test scripts.

Architecture (one sentence): a 650 mm sphere whose centre *band* rolls on
the ground, driven by an internal pendulum ("yoke") through an internal ring
gear, while the food pod and the two side caps (cameras, eyes) hang level on
roller tracks and never rotate with the band.
"""
from dataclasses import dataclass, field
import math


@dataclass
class Geometry:
    R: float = 0.325                 # outer shell radius (650 mm sphere)
    wall: float = 0.005              # band wall thickness (rotomoulded LLDPE)
    band_lat: float = math.radians(30)   # band spans +/-30 deg latitude; caps beyond
    # hatch (a flush, load-bearing door in the band)
    hatch_lon: float = math.radians(34)  # half-angle along rolling direction
    hatch_lat: float = math.radians(25)  # half-angle across (inside band)
    # food pod: a drum section hanging on the axle, radius < battery swing radius
    pod_r: float = 0.200             # drum radius about the axle
    pod_top: float = 0.085           # flat top (z, body frame)
    pod_half_w: float = 0.150        # half width along axle
    pod_insul: float = 0.025         # EPP insulation thickness
    # cameras in the stationary caps (stereo pair, 0.40 m baseline)
    cam_lat: float = math.radians(38)
    cam_pitch_down: float = math.radians(18)
    cam_yaw_out: float = math.radians(12)
    cam_fovy: float = 78.0           # rectified crop of a ~190 deg fisheye
    # battery (steering) pendulum
    batt_r: float = 0.262            # distance of pack CG from centre
    steer_max: float = math.radians(28)
    # park legs (4, out of the stationary caps)
    leg_x: float = 0.11
    leg_y: float = 0.215
    leg_stroke: float = 0.110


@dataclass
class Masses:
    band: float = 3.20       # band + tread + ring gear + roller track
    lid: float = 0.40
    body: float = 4.40       # pod shell/liner/insulation + 2 caps + cameras + eye panel + lock
    body_cg_z: float = -0.035
    yoke: float = 4.70       # 2 motors, arms, tray, compute, steel ballast
    yoke_cg_r: float = 0.154
    battery: float = 3.00    # 36 V 10S3P 21700 pack, 432 Wh
    payload: float = 5.0     # food (default: loaded)
    payload_cg_z: float = -0.075


@dataclass
class Contact:
    # rubber-like TPU tread on concrete
    slide: float = 0.9
    torsion: float = 0.012     # [m] ~ (2/3)*mu*patch radius  -> enables lean steering
    roll: float = 0.0035       # [m] -> rolling resistance coeff ~ roll/R ~ 0.011


@dataclass
class Drive:
    motor_peak_torque: float = 22.0      # at the band, both motors, [Nm] (current-limited)
    motor_cont_torque: float = 12.0
    steer_kp: float = 180.0              # battery swing servo (position loop) [Nm/rad]
    steer_kv: float = 14.0
    steer_torque: float = 12.0
    body_damping: float = 0.025          # roller-track viscous drag (12 x 608 rollers) [Nm s/rad]
    body_friction: float = 0.04          # roller-track breakaway [Nm]
    yoke_damping: float = 0.02
    lid_kp: float = 25.0
    leg_kp: float = 3000.0


@dataclass
class Electrical:
    battery_wh: float = 432.0            # 10S3P, 36 V nominal, 12 Ah
    usable_frac: float = 0.85            # keep 15 % reserve (return-to-base margin)
    drivetrain_eff: float = 0.72         # motor+ESC+gear, at the typical low-torque point
    regen_eff: float = 0.0               # not relied on
    hotel_w: dict = field(default_factory=lambda: {
        "compute (Orin Nano, perception)": 11.0,
        "cameras x4": 3.2,
        "LTE + GNSS": 2.5,
        "eyes + glow LEDs (avg)": 2.0,
        "motor drivers idle / BMS / misc": 2.5,
    })


@dataclass
class Params:
    geo: Geometry = field(default_factory=Geometry)
    m: Masses = field(default_factory=Masses)
    c: Contact = field(default_factory=Contact)
    d: Drive = field(default_factory=Drive)
    e: Electrical = field(default_factory=Electrical)
    timestep: float = 0.002

    def total_mass(self):
        m = self.m
        return m.band + m.lid + m.body + m.yoke + m.battery + m.payload

    def pendulum_moment(self):
        """kg*m of drive pendulum about the axle (sets max hill-climb torque)."""
        return self.m.yoke * self.m.yoke_cg_r + self.m.battery * self.geo.batt_r

    def max_static_grade(self):
        """Largest slope (deg) the pendulum can hold at a 90 deg swing."""
        s = self.pendulum_moment() / (self.total_mass() * self.geo.R)
        return math.degrees(math.asin(min(1.0, s)))


P = Params()
