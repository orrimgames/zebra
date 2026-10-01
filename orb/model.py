"""MJCF for the ORB delivery sphere (v2).

Kinematic tree (all big hinges about the shell's axle = shell-local y):

  shell  (free joint; exact sphere contact, rolling band visual, hatch cut-out)
   |- lid      hinge  (flush load-bearing door in the band)
   |- yoke     hinge  "drive"  <- motor torque between band and pendulum
   |    '- battery   hinge "steer" (about x): LFP tray on a curved rail -> lean -> turn
   '- body     hinge  "pod_pitch" (leveled by a small motor): side caps, cameras, eyes, legs
        |- pod       hinge "pod_roll" (passive, damped gimbal)
        |    '- pod_iso  slide "pod_iso" (elastomer mounts) -> payload
        '- leg_1..4  slide (lead screw + series spring)  '- wheel_k  hinge (fixed caster)
"""
import numpy as np
from .params import P as DEFAULT
from . import meshes


def _f(v):
    return " ".join(f"{x:.5f}" for x in v)


def camera_axes(fwd_yaw, pitch_down):
    f = np.array([np.cos(pitch_down) * np.cos(fwd_yaw), np.cos(pitch_down) * np.sin(fwd_yaw), -np.sin(pitch_down)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u


def build(P=DEFAULT, pos=(0, 0), yaw=0.0, z0=0.0):
    g, m, c, d = P.geo, P.m, P.c, P.d
    R = g.R
    hinge = meshes._sph(R, -g.hatch_lon, 0.0)          # lid hinge on trailing hatch edge
    hinge[1] = 0.0

    assets = []
    assets.append(meshes.band_mesh(R, g.wall, g.band_lat, g.hatch_lon, g.hatch_lat).mjcf("orb_band"))
    assets.append(meshes.lid_mesh(R, g.wall, g.hatch_lon, g.hatch_lat, hinge).mjcf("orb_lid"))
    assets.append(meshes.cap_mesh(R, 0.004, g.band_lat, +1).mjcf("orb_capL"))
    assets.append(meshes.cap_mesh(R, 0.004, g.band_lat, -1).mjcf("orb_capR"))
    assets.append(meshes.bowl_mesh(g.pod_r, g.pod_top, g.pod_insul).mjcf("orb_pod"))
    for sd, nm in ((+1, "orb_seamL"), (-1, "orb_seamR")):
        assets.append(meshes.seam_ring_mesh(R - 0.0015, g.band_lat + np.radians(0.05), g.band_lat + np.radians(0.55), sd).mjcf(nm))
    assets.append("""
    <material name="orb_band" rgba="0.93 0.95 0.98 0.78" emission="0.35" specular="0.6" shininess="0.8" reflectance="0.05"/>
    <material name="orb_lid" rgba="0.93 0.95 0.98 0.82" emission="0.35" specular="0.6" shininess="0.8"/>
    <material name="orb_cap" rgba="0.07 0.08 0.10 1" specular="1" shininess="1" reflectance="0.25"/>
    <material name="orb_pod" rgba="0.18 0.19 0.22 1" specular="0.3"/>
    <material name="orb_liner" rgba="0.85 0.87 0.88 1" specular="0.4"/>
    <material name="orb_eye" rgba="0.35 0.95 1.0 1" emission="1"/>
    <material name="orb_accent" rgba="0.35 0.95 1.0 1" emission="0.8"/>
    <material name="orb_metal" rgba="0.5 0.52 0.55 1" specular="0.8"/>
    <material name="orb_batt" rgba="0.15 0.35 0.75 1"/>
    <material name="orb_glass" rgba="0.02 0.02 0.03 1" specular="1" shininess="1" reflectance="0.4"/>
    <material name="orb_tire" rgba="0.08 0.08 0.08 1"/>
    """)

    vis = 'contype="0" conaffinity="0" group="1"'
    # ---- cameras (stereo pair on the front of the caps) + side crops + rear pair
    cams, camgeoms = [], []
    for side, name in ((+1, "cam_left"), (-1, "cam_right")):
        p = meshes._sph(R + 0.002, np.pi / 2, side * g.cam_lat)   # lon=90deg -> +x (forward)
        rr, uu = camera_axes(side * g.cam_yaw_out, g.cam_pitch_down)
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
        n = p / np.linalg.norm(p)
        camgeoms.append(f'<geom type="cylinder" size="0.022 0.0015" pos="{_f(p - 0.001*n)}" zaxis="{_f(n)}" material="orb_glass" {vis}/>')
        camgeoms.append(f'<geom type="cylinder" size="0.028 0.001" pos="{_f(p - 0.0018*n)}" zaxis="{_f(n)}" material="orb_accent" {vis}/>')
    for side, name in ((+1, "cam_left_side"), (-1, "cam_right_side")):
        p = meshes._sph(R + 0.002, np.pi / 2, side * g.cam_lat)
        rr, uu = camera_axes(side * np.radians(62), np.radians(8))
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
    for side, name in ((+1, "cam_rear_left"), (-1, "cam_rear_right")):
        p = meshes._sph(R + 0.002, -np.pi / 2, side * g.cam_lat)
        rr, uu = camera_axes(np.pi - side * g.cam_yaw_out, g.cam_pitch_down)
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
        n = p / np.linalg.norm(p)
        camgeoms.append(f'<geom type="cylinder" size="0.016 0.0015" pos="{_f(p - 0.001*n)}" zaxis="{_f(n)}" material="orb_glass" {vis}/>')
    # ToF / radar sites (rays are cast from these by the sensor layer)
    sites = []
    for k, (side, lon_deg) in enumerate(((+1, 90), (-1, 90), (+1, 100), (-1, 100))):
        # near-field ToF in the cap fronts, below the cameras (two per cap)
        p = meshes._sph(R + 0.002, np.radians(lon_deg), side * np.radians(33))
        p[2] -= 0.06
        sites.append(f'<site name="tof_{k}" pos="{_f(p)}" size="0.005" rgba="0 0 0 0"/>')
    for side, name in ((+1, "radar_left"), (-1, "radar_right")):
        p = meshes._sph(R + 0.002, np.pi / 2, side * np.radians(55))
        sites.append(f'<site name="{name}" pos="{_f(p)}" size="0.005" rgba="0 0 0 0"/>')

    # ---- eyes: LED panel shines through the band; rendered at the band surface
    eyes = []
    for side, name in ((+1, "eye_left"), (-1, "eye_right")):
        lat, lon = side * np.radians(10.5), np.radians(90 - 11)
        p = meshes._sph(R + 0.0015, lon, lat)
        n = p / np.linalg.norm(p)
        eyes.append(f'<geom name="{name}" type="ellipsoid" size="0.036 0.026 0.0015" pos="{_f(p)}" zaxis="{_f(n)}" material="orb_eye" {vis}/>')
    rings = [f'<geom name="ring_L" type="mesh" mesh="orb_seamL" material="orb_accent" {vis}/>',
             f'<geom name="ring_R" type="mesh" mesh="orb_seamR" material="orb_accent" {vis}/>']

    # ---- legs: lead screw + series spring, fixed caster wheel at the foot
    legs, leg_act, excl = [], [], []
    k = 0
    ex = g.leg_exit_depth()
    wr = g.leg_wheel_r
    for sx in (+1, -1):
        for sy in (+1, -1):
            k += 1
            x, y = sx * g.leg_x, sy * g.leg_y
            zc = -ex + wr + 0.002          # wheel centre when retracted (flush with the cap)
            legs.append(f"""
        <body name="leg_{k}" pos="{x:.4f} {y:.4f} {zc:.4f}">
          <inertial pos="0 0 0.05" mass="{m.leg - 0.05:.3f}" diaginertia="2e-4 2e-4 2e-5"/>
          <joint name="leg_{k}" type="slide" axis="0 0 -1" range="0 {g.leg_stroke + 0.01:.3f}" damping="60" armature="0.08"/>
          <geom type="capsule" fromto="0 0 0.03 0 0 0.17" size="0.012" material="orb_metal" {vis}/>
          <geom type="box" size="0.012 0.022 0.018" pos="0 0 0.012" material="orb_metal" {vis}/>
          <body name="wheel_{k}">
            <inertial pos="0 0 0" mass="0.05" diaginertia="2e-5 3e-5 2e-5"/>
            <joint name="wheel_{k}" type="hinge" axis="0 1 0" damping="0.0008"/>
            <geom name="foot_{k}" type="capsule" size="{wr} 0.006" zaxis="0 1 0" material="orb_tire" friction="1.0 0.01 0.002" condim="3"/>
          </body>
        </body>""")
            leg_act.append(f'<position name="leg_{k}" joint="leg_{k}" kp="{d.leg_k}" kv="350" ctrlrange="0 {g.leg_stroke + 0.01:.3f}" forcerange="{-d.leg_force} {d.leg_force}"/>')
            if True:     # all four leg wheels are driven (tiny geared DC motors, effectively self-braking)
                leg_act.append(f'<velocity name="wheel_{k}" joint="wheel_{k}" kv="3.0" ctrlrange="-12 12" forcerange="{-d.wheel_torque} {d.wheel_torque}"/>')
            excl.append(f'<exclude body1="shell" body2="leg_{k}"/>')
            excl.append(f'<exclude body1="shell" body2="wheel_{k}"/>')

    # inertias
    Ib = m.band * R**2
    band_I = f"{0.55*Ib:.4f} {0.92*Ib:.4f} {0.55*Ib:.4f}"
    x0, y0 = pos
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    mp = m.pod + m.payload
    k_iso = mp * (2 * np.pi * g.pod_iso_hz) ** 2
    c_iso = 2 * g.pod_iso_zeta * np.sqrt(k_iso * mp)
    sref = 9.81 * mp / k_iso
    tr = g.tray_r
    body = f"""
    <body name="shell" pos="{x0} {y0} {R + z0}" quat="{qw:.6f} 0 0 {qz:.6f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{m.band}" diaginertia="{band_I}"/>
      <geom name="shell_contact" type="sphere" size="{R}" rgba="1 1 1 0" group="3"
            friction="{c.slide} {c.torsion} {c.roll}" condim="6" priority="1"/>
      <geom type="mesh" mesh="orb_band" material="orb_band" {vis}/>
      <site name="hatch_center" pos="0 0 {R:.4f}" size="0.01" rgba="1 0 0 0"/>
      <body name="balance" pos="0 0 {-(R - 0.012):.4f}">
        <inertial pos="0 0 0" mass="{m.band_balance}" diaginertia="1e-4 1e-4 1e-4"/>
      </body>
      <body name="lid" pos="{_f(hinge)}">
        <inertial pos="{_f(meshes._sph(R, 0, 0) - hinge)}" mass="{m.lid}" diaginertia="0.004 0.004 0.006"/>
        <joint name="lid" type="hinge" axis="0 1 0" range="-2.05 0" damping="0.4" armature="0.01"/>
        <geom type="mesh" mesh="orb_lid" material="orb_lid" {vis}/>
        <site name="lid_center" pos="{_f(meshes._sph(R, 0, 0) - hinge)}" size="0.01" rgba="0 0 0 0"/>
      </body>

      <body name="yoke">
        <inertial pos="0 0 {-m.yoke_cg_r}" mass="{m.yoke}" diaginertia="0.03 0.012 0.03"/>
        <joint name="drive" type="hinge" axis="0 1 0" damping="{d.yoke_damping}" armature="0.03"/>
        <geom type="cylinder" size="0.035 0.02" pos="0 0.285 0" zaxis="0 1 0" material="orb_metal" {vis}/>
        <geom type="cylinder" size="0.035 0.02" pos="0 -0.285 0" zaxis="0 1 0" material="orb_metal" {vis}/>
        <geom type="capsule" fromto="0 0.275 0 0 0.20 -0.17" size="0.012" material="orb_metal" {vis}/>
        <geom type="capsule" fromto="0 -0.275 0 0 -0.20 -0.17" size="0.012" material="orb_metal" {vis}/>
        <geom type="capsule" fromto="0 0.20 -0.17 0 0.155 -0.245" size="0.012" material="orb_metal" {vis}/>
        <geom type="capsule" fromto="0 -0.20 -0.17 0 -0.155 -0.245" size="0.012" material="orb_metal" {vis}/>
        <geom type="cylinder" size="0.038 0.03" pos="0.0 0.15 -0.25" zaxis="0 1 0" material="orb_metal" {vis}/>
        <geom type="cylinder" size="0.038 0.03" pos="0.0 -0.15 -0.25" zaxis="0 1 0" material="orb_metal" {vis}/>
        <body name="battery">
          <inertial pos="0 0 {-tr}" mass="{m.battery}" diaginertia="0.0233 0.0194 0.0388"/>
          <joint name="steer" type="hinge" axis="1 0 0" range="{-g.steer_max:.4f} {g.steer_max:.4f}" damping="1.5" armature="0.03"/>
          <geom type="box" size="0.096 0.10 0.030" pos="0 0 {-tr:.4f}" material="orb_batt" {vis}/>
        </body>
      </body>

      <body name="body">
        <inertial pos="0 0 {m.caps_cg_z:.4f}" mass="{m.caps:.3f}" diaginertia="0.17 0.10 0.17"/>
        <joint name="pod_pitch" type="hinge" axis="0 1 0" damping="{d.body_damping}" frictionloss="{d.body_friction}" armature="0.01"/>
        <site name="imu" pos="0 0 0" size="0.01"/>
        {''.join(sites)}
        <geom type="mesh" mesh="orb_capL" material="orb_cap" {vis}/>
        <geom type="mesh" mesh="orb_capR" material="orb_cap" {vis}/>
        {''.join(rings)}
        {''.join(camgeoms)}
        {''.join(eyes)}
        {''.join(cams)}
        <body name="pod">
          <!-- food bowl on a passive, damped roll gimbal (+/-12 deg) ... -->
          <inertial pos="0 0 -0.08" mass="0.25" diaginertia="0.004 0.004 0.004"/>
          <joint name="pod_roll" type="hinge" axis="1 0 0" range="-0.21 0.21" damping="0.6" armature="0.005"/>
          <body name="pod_iso">
            <!-- ... hung on elastomer mounts (vertical isolation) -->
            <inertial pos="0 0 -0.08" mass="{m.pod - 0.25:.3f}" diaginertia="0.03 0.03 0.03"/>
            <joint name="pod_iso" type="slide" axis="0 0 1" range="{-g.pod_iso_travel} {g.pod_iso_travel}" stiffness="{k_iso:.1f}" springref="{sref:.5f}" damping="{c_iso:.1f}" solreflimit="0.004 1"/>
            <site name="pod_imu" pos="0 0 -0.06" size="0.01"/>
            <geom type="mesh" mesh="orb_pod" material="orb_pod" {vis}/>
            <geom type="cylinder" size="{np.sqrt((g.pod_r-g.pod_insul)**2-0.03**2)-0.01:.3f} 0.002" pos="0 0 0.03" material="orb_liner" {vis}/>
            <body name="payload" pos="0 0 {m.payload_cg_z}">
              <inertial pos="0 0 0" mass="{max(m.payload, 1e-3)}" diaginertia="0.02 0.02 0.02"/>
              <geom name="food_bag" type="box" size="0.10 0.08 0.07" pos="-0.03 0 0.0" rgba="0.72 0.55 0.35 1" {vis}/>
              <geom name="food_cup" type="cylinder" size="0.042 0.065" pos="0.10 0.04 0.02" rgba="0.95 0.95 0.95 1" {vis}/>
              <geom name="food_cup_lid" type="cylinder" size="0.044 0.006" pos="0.10 0.04 0.088" rgba="0.2 0.2 0.2 1" {vis}/>
            </body>
          </body>
        </body>
        {''.join(legs)}
      </body>
    </body>"""

    actuators = f"""
    <general name="pod_level" tendon="pod_vs_yoke" dyntype="filter" dynprm="0.01" gainprm="1" ctrllimited="true" ctrlrange="{-d.level_torque} {d.level_torque}"/>
    <general name="drive" joint="drive" dyntype="filter" dynprm="{d.motor_lag}" gainprm="1" ctrllimited="true" ctrlrange="{-d.motor_peak_torque} {d.motor_peak_torque}"/>
    <position name="steer" joint="steer" kp="{d.steer_kp}" kv="{d.steer_kv}" ctrlrange="{-g.steer_max:.4f} {g.steer_max:.4f}" forcerange="{-d.steer_torque} {d.steer_torque}"/>
    <position name="lid" joint="lid" kp="{d.lid_kp}" kv="2" ctrlrange="-2.05 0" forcerange="-6 6"/>
    {''.join(leg_act)}"""

    sensors = """
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <accelerometer name="pod_acc" site="pod_imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <jointpos name="enc_drive" joint="drive"/>
    <jointpos name="enc_pod" joint="pod_pitch"/>
    <jointvel name="encv_drive" joint="drive"/>
    <jointvel name="encv_pod" joint="pod_pitch"/>
    """
    contact = "".join(excl)
    tendon = """<fixed name="pod_vs_yoke"><joint joint="pod_pitch" coef="1"/><joint joint="drive" coef="-1"/></fixed>"""
    equality = """
    <joint name="brake_band" joint1="pod_pitch" polycoef="0 1 0 0 0" active="false" solref="0.01 1"/>
    <joint name="brake_yoke" joint1="drive" polycoef="0 1 0 0 0" active="false" solref="0.01 1"/>
    <joint name="lock_level" joint1="pod_pitch" joint2="drive" polycoef="0 1 0 0 0" active="false" solref="0.008 1"/>"""
    return dict(assets="".join(assets), body=body, actuators=actuators, sensors=sensors, contact=contact, tendon=tendon, equality=equality)


def standalone_xml(P=DEFAULT, floor_extra="", extra_bodies="", floor_euler=None, pos=(0, 0), yaw=0.0):
    r = build(P, pos=pos, yaw=yaw)
    fe = f'euler="{floor_euler[0]} {floor_euler[1]} {floor_euler[2]}"' if floor_euler is not None else ""
    return f"""
<mujoco model="orb_test">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{P.timestep}" integrator="implicitfast" cone="elliptic" impratio="3"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <default><geom friction="0.9 0.012 0.0035"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.56 0.58" rgb2="0.5 0.51 0.53" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="40 40"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.92" rgb2="0.92 0.95 1.0" width="256" height="256"/>
    {r['assets']}
  </asset>
  <worldbody>
    <light pos="2 -2 6" dir="-0.3 0.3 -1" directional="true" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="80 80 0.1" material="grid" friction="0.9 0.012 0.0035" {fe}/>
    {floor_extra}
    {r['body']}
    {extra_bodies}
  </worldbody>
  <contact>{r['contact']}</contact>
  <tendon>{r['tendon']}</tendon>
  <equality>{r['equality']}</equality>
  <actuator>{r['actuators']}</actuator>
  <sensor>{r['sensors']}</sensor>
</mujoco>"""
