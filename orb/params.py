"""All physical parameters for the ORB campus delivery sphere (v2).

Units: SI (m, kg, s, N, rad). Every number the sim uses lives here so the
design can be re-tuned from one place and re-verified by the test scripts.

Architecture (one sentence): a 650 mm sphere whose centre *band* rolls on
the ground, driven by an internal pendulum ("yoke") through an internal ring
gear, while the food pod and the two side caps (cameras, eyes, legs) hang
level on roller tracks and never rotate with the band.

v2 changes (why):
  * battery is now a 12S3P LiFePO4 pack (5.2 kg, 690 Wh) that rides on a
    curved rail at the bottom of the pendulum and IS the steering mass:
    LFP is heavier per Wh, which here is a feature (it is ballast that also
    stores energy), it cannot thermally run away, and it lasts 3000+ cycles.
    Steering moment x2.1, drive pendulum moment +33 %, no dead steel ballast.
  * drive motors sit at the bottom of the pendulum next to the ring gear
    (low CG) instead of at the hub.
  * food pod is a bowl concentric with the shell, hung on elastomer mounts
    (vertical isolation) inside its roll gimbal.
  * four telescoping series-elastic legs (0.31 m stroke) with fixed wheels,
    all four driven by tiny gearmotors: they park the robot AND
    turn it into a 4-wheeled cart that lifts itself over curbs and steps up
    to 15 cm.
"""
from dataclasses import dataclass, field
import math


@dataclass
class Geometry:
    R: float = 0.325                 # outer shell radius (650 mm sphere)
    wall: float = 0.005              # band wall thickness
    band_lat: float = math.radians(30)   # band spans +/-30 deg latitude; caps beyond
    # hatch (a flush, load-bearing door in the band)
    hatch_lon: float = math.radians(34)  # half-angle along rolling direction
    hatch_lat: float = math.radians(25)  # half-angle across (inside band)
    # food pod: insulated bowl concentric with the shell, flat open top
    pod_r: float = 0.205             # outer radius of the bowl
    pod_top: float = 0.085           # flat rim height (z, body frame)
    pod_insul: float = 0.025         # EPP insulation thickness
    pod_iso_hz: float = 7.0          # elastomer mounts: vertical natural frequency
    pod_iso_zeta: float = 0.35
    pod_iso_travel: float = 0.012    # +/- before the bump stops
    # cameras in the stationary caps (stereo pair, 0.40 m baseline)
    cam_lat: float = math.radians(38)
    cam_pitch_down: float = math.radians(18)
    cam_yaw_out: float = math.radians(12)
    cam_fovy: float = 78.0           # rectified crop of a ~190 deg fisheye
    # steering tray (LFP pack + BMS) on a curved rail under the pod
    tray_r: float = 0.255            # CG radius from the centre
    steer_max: float = math.radians(32)
    # telescoping legs (4, out of the stationary caps) with fixed caster wheels
    leg_x: float = 0.12
    leg_y: float = 0.235
    leg_stroke: float = 0.31
    leg_speed: float = 0.12          # m/s (lead screw)
    leg_wheel_r: float = 0.025

    def leg_exit_depth(self):
        """Depth below the centre where a leg leaves the cap surface."""
        return math.sqrt(self.R ** 2 - self.leg_x ** 2 - self.leg_y ** 2)


@dataclass
class Masses:
    band: float = 3.20       # band + tread + 2 ring gears + roller tracks
    lid: float = 0.40
    band_balance: float = 0.41   # steel strip bonded in the band opposite the hatch (statically balances the lid)
    caps: float = 2.55       # 2 caps + cameras + radar + eye panel + electronics + leg tubes
    caps_cg_z: float = -0.02
    pod: float = 1.55        # EPP bowl + PP liner + lock (on the isolators)
    leg: float = 0.17        # moving stage + caster wheel, each
    yoke: float = 2.20       # 2 BLDC + gearheads + C-arms + drivers, motors at the bottom
    yoke_cg_r: float = 0.240
    battery: float = 5.80    # steering tray: 12S3P LFP 32700 (36 cells) + BMS + rail carriage
    payload: float = 5.0     # food (default: loaded)
    payload_cg_z: float = -0.085


@dataclass
class Contact:
    # rubber-like TPU tread on concrete
    slide: float = 0.9
    torsion: float = 0.012     # [m] ~ (2/3)*mu*patch radius  -> enables lean steering
    roll: float = 0.0035       # [m] -> rolling resistance coeff ~ roll/R ~ 0.011


@dataclass
class Drive:
    motor_peak_torque: float = 32.0      # at the band, both motors, [Nm] (current-limited)
    motor_cont_torque: float = 14.0
    motor_lag: float = 0.006             # torque response time constant (current loop + filter) [s]
    steer_kp: float = 320.0              # tray swing servo (position loop) [Nm/rad]
    steer_kv: float = 22.0
    steer_torque: float = 26.0
    steer_rate: float = 5.0              # rad/s slew limit
    body_damping: float = 0.025          # roller-track viscous drag [Nm s/rad]
    body_friction: float = 0.04          # roller-track breakaway [Nm]
    yoke_damping: float = 0.02
    lid_kp: float = 25.0
    level_torque: float = 30.0           # cap/pod leveling gearmotor (holds the caps level while legs push)
    leg_k: float = 9000.0                # series spring in each leg [N/m]
    leg_force: float = 420.0             # lead-screw thrust limit per leg [N]
    wheel_torque: float = 0.8            # leg-wheel gearmotors, at the wheel [Nm] (-> 32 N traction each)


@dataclass
class Aero:
    rho: float = 1.2
    cd_sphere: float = 0.50              # subcritical sphere (conservative; drag crisis would lower it)
    cd_lid: float = 1.2                  # flat-ish plate
    lid_area: float = 0.105


@dataclass
class Electrical:
    battery_wh: float = 690.0            # 12S3P LFP 32700 6 Ah, 38.4 V nominal, 18 Ah
    usable_frac: float = 0.85            # keep 15 % reserve (return-to-base margin)
    drivetrain_eff: float = 0.75         # motor+ESC+gear at the typical low-torque point
    regen_eff: float = 0.0               # not relied on
    hotel_w: dict = field(default_factory=lambda: {
        "compute (Orin Nano, perception)": 11.0,
        "cameras x4 + IR": 3.6,
        "radar x2": 1.6,
        "LTE + GNSS": 2.5,
        "eyes + glow LEDs (avg)": 2.0,
        "safety MCU, drivers idle, BMS, misc": 2.8,
    })


@dataclass
class Params:
    geo: Geometry = field(default_factory=Geometry)
    m: Masses = field(default_factory=Masses)
    c: Contact = field(default_factory=Contact)
    d: Drive = field(default_factory=Drive)
    a: Aero = field(default_factory=Aero)
    e: Electrical = field(default_factory=Electrical)
    timestep: float = 0.002
    control_dt: float = 0.004            # 250 Hz motor-control loop on the robot
    sensor_delay: float = 0.004          # one control period of IMU/encoder latency

    def total_mass(self):
        m = self.m
        return m.band + m.lid + m.band_balance + m.caps + m.pod + 4 * m.leg + m.yoke + m.battery + m.payload

    def pendulum_moment(self):
        """kg*m of drive pendulum about the axle (sets max hill-climb torque)."""
        return self.m.yoke * self.m.yoke_cg_r + self.m.battery * self.geo.tray_r

    def steer_moment(self):
        """kg*m of sideways CG shift the tray can make (sets turn + crosswind authority)."""
        return self.m.battery * self.geo.tray_r * math.sin(self.geo.steer_max)

    def cg_depth(self):
        m = self.m
        mom = (self.pendulum_moment() + m.caps * -m.caps_cg_z + m.pod * 0.08 + 4 * m.leg * 0.19
               + m.payload * -m.payload_cg_z)
        return mom / self.total_mass()

    def max_static_grade(self, swing=math.radians(90)):
        """Largest slope (deg) the pendulum can hold at a given swing."""
        s = self.pendulum_moment() * math.sin(swing) / (self.total_mass() * self.geo.R)
        return math.degrees(math.asin(min(1.0, s)))

    def drag_force(self, wind):
        g = self.geo
        return 0.5 * self.a.rho * self.a.cd_sphere * math.pi * g.R ** 2 * wind * abs(wind)


P = Params()
