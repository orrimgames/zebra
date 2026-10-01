"""MJCF for the ORB delivery sphere.

Kinematic tree (all hinges about the shell's axle = shell-local y):

  shell  (free joint; exact sphere contact, rolling band visual, hatch cut-out)
   |- lid      hinge  (flush load-bearing door in the band)
   |- yoke     hinge  "drive"  <- motor torque between band and pendulum
   |    '- battery   hinge "steer" (about x): swings the 3 kg pack sideways -> lean -> turn
   '- body     hinge  "pod_pitch" (passive, damped): food pod + side caps + cameras + eyes
        |- payload   (food mass)
        '- leg_1..4  slide joints (park legs out of the stationary caps)
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


def build(P=DEFAULT, pos=(0, 0), yaw=0.0):
    g, m, c, d = P.geo, P.m, P.c, P.d
    R = g.R
    hinge = meshes._sph(R, -g.hatch_lon, 0.0)          # lid hinge on trailing hatch edge
    hinge[1] = 0.0

    assets = []
    assets.append(meshes.band_mesh(R, g.wall, g.band_lat, g.hatch_lon, g.hatch_lat).mjcf("orb_band"))
    assets.append(meshes.lid_mesh(R, g.wall, g.hatch_lon, g.hatch_lat, hinge).mjcf("orb_lid"))
    assets.append(meshes.cap_mesh(R, 0.004, g.band_lat, +1).mjcf("orb_capL"))
    assets.append(meshes.cap_mesh(R, 0.004, g.band_lat, -1).mjcf("orb_capR"))
    assets.append(meshes.pod_mesh(g.pod_r, g.pod_top, g.pod_half_w, g.pod_insul).mjcf("orb_pod"))
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
    """)

    vis = 'contype="0" conaffinity="0" group="1"'
    # ---- camera frames (stereo pair on the front of the caps)
    cams, camgeoms = [], []
    for side, name in ((+1, "cam_left"), (-1, "cam_right")):
        p = meshes._sph(R + 0.002, np.pi / 2, side * g.cam_lat)   # lon=90deg -> +x (forward)
        rr, uu = camera_axes(side * g.cam_yaw_out, g.cam_pitch_down)
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
        n = p / np.linalg.norm(p)
        camgeoms.append(f'<geom type="cylinder" size="0.022 0.0015" pos="{_f(p - 0.001*n)}" zaxis="{_f(n)}" material="orb_glass" {vis}/>')
        camgeoms.append(f'<geom type="cylinder" size="0.028 0.001" pos="{_f(p - 0.0018*n)}" zaxis="{_f(n)}" material="orb_accent" {vis}/>')
    # side crops of the same front fisheyes (used when checking a road before crossing)
    for side, name in ((+1, "cam_left_side"), (-1, "cam_right_side")):
        p = meshes._sph(R + 0.002, np.pi / 2, side * g.cam_lat)
        rr, uu = camera_axes(side * np.radians(62), np.radians(8))
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
    # rear fisheyes (for reversing; not rendered by the sim loop by default)
    for side, name in ((+1, "cam_rear_left"), (-1, "cam_rear_right")):
        p = meshes._sph(R + 0.002, -np.pi / 2, side * g.cam_lat)
        rr, uu = camera_axes(np.pi - side * g.cam_yaw_out, g.cam_pitch_down)
        cams.append(f'<camera name="{name}" pos="{_f(p)}" xyaxes="{_f(rr)} {_f(uu)}" fovy="{g.cam_fovy}"/>')
        n = p / np.linalg.norm(p)
        camgeoms.append(f'<geom type="cylinder" size="0.016 0.0015" pos="{_f(p - 0.001*n)}" zaxis="{_f(n)}" material="orb_glass" {vis}/>')

    # ---- eyes: LED panel on the pod front shines through the band; rendered as
    # glowing ellipsoids just at the band surface (they ride on the level body,
    # so the face stays forward while the band rolls underneath it).
    eyes = []
    for side, name in ((+1, "eye_left"), (-1, "eye_right")):
        lat, lon = side * np.radians(10.5), np.radians(90 - 11)
        p = meshes._sph(R + 0.0015, lon, lat)
        n = p / np.linalg.norm(p)
        eyes.append(f'<geom name="{name}" type="ellipsoid" size="0.036 0.026 0.0015" pos="{_f(p)}" zaxis="{_f(n)}" material="orb_eye" {vis}/>')
    # status glow ring in each cap seam (thin strip, lit from inside)
    rings = [f'<geom name="ring_L" type="mesh" mesh="orb_seamL" material="orb_accent" {vis}/>',
             f'<geom name="ring_R" type="mesh" mesh="orb_seamR" material="orb_accent" {vis}/>']

    # ---- legs
    legs, leg_act, excl = [], [], []
    k = 0
    for sx in (+1, -1):
        for sy in (+1, -1):
            k += 1
            x, y = sx * g.leg_x, sy * g.leg_y
            z0 = -np.sqrt(R**2 - x**2 - y**2) + 0.055
            legs.append(f"""
        <body name="leg_{k}" pos="{x:.4f} {y:.4f} {z0:.4f}">
          <inertial pos="0 0 -0.03" mass="0.12" diaginertia="4e-5 4e-5 1e-5"/>
          <joint name="leg_{k}" type="slide" axis="0 0 -1" range="0 {g.leg_stroke + 0.015:.3f}" damping="40" armature="0.05"/>
          <geom type="capsule" fromto="0 0 0.0 0 0 -0.030" size="0.011" material="orb_metal" contype="0" conaffinity="0" group="1"/>
          <geom name="foot_{k}" type="sphere" size="0.018" pos="0 0 -0.032" rgba="0.1 0.1 0.1 1" friction="1.2 0.02 0.001" condim="4"/>
        </body>""")
            leg_act.append(f'<position name="leg_{k}" joint="leg_{k}" kp="{d.leg_kp}" kv="120" ctrlrange="0 {g.leg_stroke + 0.015:.3f}" forcerange="-400 400"/>')
            excl.append(f'<exclude body1="shell" body2="leg_{k}"/>')

    # inertias
    Ib = m.band * R**2
    band_I = f"{0.55*Ib:.4f} {0.92*Ib:.4f} {0.55*Ib:.4f}"
    x0, y0 = pos
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    body = f"""
    <body name="shell" pos="{x0} {y0} {R}" quat="{qw:.6f} 0 0 {qz:.6f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{m.band}" diaginertia="{band_I}"/>
      <geom name="shell_contact" type="sphere" size="{R}" rgba="1 1 1 0" group="3"
            friction="{c.slide} {c.torsion} {c.roll}" condim="6" priority="1"/>
      <geom type="mesh" mesh="orb_band" material="orb_band" {vis}/>
      <site name="hatch_center" pos="0 0 {R:.4f}" size="0.01" rgba="1 0 0 0"/>
      <body name="lid" pos="{_f(hinge)}">
        <inertial pos="{_f(meshes._sph(R, 0, 0) - hinge)}" mass="{m.lid}" diaginertia="0.004 0.004 0.006"/>
        <joint name="lid" type="hinge" axis="0 1 0" range="-2.05 0" damping="0.4" armature="0.01"/>
        <geom type="mesh" mesh="orb_lid" material="orb_lid" {vis}/>
      </body>

      <body name="yoke">
        <inertial pos="0 0 {-m.yoke_cg_r}" mass="{m.yoke}" diaginertia="0.09 0.10 0.06"/>
        <joint name="drive" type="hinge" axis="0 1 0" damping="{d.yoke_damping}" armature="0.02"/>
        <geom type="cylinder" size="0.05 0.022" pos="0 0.255 0" zaxis="0 1 0" material="orb_metal" {vis}/>
        <geom type="cylinder" size="0.05 0.022" pos="0 -0.255 0" zaxis="0 1 0" material="orb_metal" {vis}/>
        <geom type="box" size="0.018 0.008 0.12" pos="0 0.232 -0.13" material="orb_metal" {vis}/>
        <geom type="box" size="0.018 0.008 0.12" pos="0 -0.232 -0.13" material="orb_metal" {vis}/>
        <geom type="box" size="0.10 0.20 0.012" pos="0 0 -0.285" euler="0 0 0" material="orb_metal" {vis}/>
        <body name="battery">
          <inertial pos="0 0 {-g.batt_r}" mass="{m.battery}" diaginertia="0.0058 0.011 0.015"/>
          <joint name="steer" type="hinge" axis="1 0 0" range="{-g.steer_max:.4f} {g.steer_max:.4f}" damping="1.5" armature="0.02"/>
          <geom type="box" size="0.10 0.07 0.028" pos="0 0 {-g.batt_r:.4f}" material="orb_batt" {vis}/>
        </body>
      </body>

      <body name="body">
        <inertial pos="0 0 {m.body_cg_z + 0.025:.4f}" mass="{m.body - 1.8:.3f}" diaginertia="0.13 0.08 0.13"/>
        <joint name="pod_pitch" type="hinge" axis="0 1 0" damping="{d.body_damping}" frictionloss="{d.body_friction}" armature="0.01"/>
        <site name="imu" pos="0 0 0" size="0.01"/>
        <geom type="mesh" mesh="orb_capL" material="orb_cap" {vis}/>
        <geom type="mesh" mesh="orb_capR" material="orb_cap" {vis}/>
        {''.join(rings)}
        {''.join(camgeoms)}
        {''.join(eyes)}
        {''.join(cams)}
        <body name="pod">
          <!-- food pod on a passive, damped roll gimbal (+/-12 deg): stays level in lean-steered turns -->
          <inertial pos="0 0 -0.07" mass="1.8" diaginertia="0.03 0.03 0.035"/>
          <joint name="pod_roll" type="hinge" axis="1 0 0" range="-0.21 0.21" damping="0.6" armature="0.005"/>
          <site name="pod_imu" pos="0 0 -0.06" size="0.01"/>
          <geom type="mesh" mesh="orb_pod" material="orb_pod" {vis}/>
          <geom type="box" size="{g.pod_r - g.pod_insul - 0.02:.3f} {g.pod_half_w - g.pod_insul - 0.002:.3f} 0.002" pos="0 0 {-(g.pod_r - g.pod_insul) + 0.03:.3f}" material="orb_liner" {vis}/>
          <body name="payload" pos="0 0 {m.payload_cg_z}">
            <inertial pos="0 0 0" mass="{max(m.payload, 1e-3)}" diaginertia="0.02 0.02 0.02"/>
            <geom name="food_bag" type="box" size="0.12 0.09 0.075" pos="-0.03 0 0.0" rgba="0.72 0.55 0.35 1" {vis}/>
            <geom name="food_cup" type="cylinder" size="0.045 0.07" pos="0.12 0.05 0.005" rgba="0.95 0.95 0.95 1" {vis}/>
            <geom name="food_cup_lid" type="cylinder" size="0.047 0.006" pos="0.12 0.05 0.078" rgba="0.2 0.2 0.2 1" {vis}/>
          </body>
        </body>
        {''.join(legs)}
      </body>
    </body>"""

    actuators = f"""
    <motor name="pod_level" tendon="pod_vs_yoke" gear="1" ctrlrange="-4 4"/>
    <motor name="drive" joint="drive" gear="1" ctrlrange="{-d.motor_peak_torque} {d.motor_peak_torque}"/>
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
    <joint name="brake_yoke" joint1="drive" polycoef="0 1 0 0 0" active="false" solref="0.01 1"/>"""
    return dict(assets="".join(assets), body=body, actuators=actuators, sensors=sensors, contact=contact, tendon=tendon, equality=equality)


def standalone_xml(P=DEFAULT, floor_extra="", extra_bodies=""):
    r = build(P)
    return f"""
<mujoco model="orb_test">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{P.timestep}" integrator="implicitfast" cone="elliptic" impratio="3"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.56 0.58" rgb2="0.5 0.51 0.53" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="40 40"/>
    {r['assets']}
  </asset>
  <worldbody>
    <light pos="2 -2 6" dir="-0.3 0.3 -1" directional="true" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="60 60 0.1" material="grid" friction="0.9 0.012 0.0035"/>
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
