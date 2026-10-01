"""Procedural campus: sidewalks, raised plaza with 1:12 ADA ramps, a road with
curb ramps + crosswalk, sidewalk lips, static obstacles, scripted pedestrians
and cars (mocap), buildings, trees, lamps.

Ground levels:  grass top z=0, sidewalks z=+0.004, road z=-0.12 (12 cm curb),
plaza z=+0.304.
"""
import numpy as np
from dataclasses import dataclass, field

ROAD_Z = -0.12
SW_Z = 0.004
PLAZA_Z = 0.304
SW_W = 2.4          # sidewalk width
RAMP_Y0, RAMP_Y1 = 23.3, 24.7   # 1.4 m wide ADA curb ramps at the crosswalk


@dataclass
class Agent:
    name: str
    kind: str                    # 'ped' or 'car'
    path: list                   # list of (x, y)
    speed: float
    trigger: object = None       # f(robot_xy, t) -> bool
    z: float = 0.0
    started: float = None
    color: tuple = (0.3, 0.3, 0.6)
    idle_yaw: float = 0.0
    loop: bool = False
    scale: float = 1.0           # people: 1.0 adult, ~0.65 child

    def pose(self, t):
        if self.started is None:
            p = np.array(self.path[0], float)
            if len(self.path) > 1:
                dvec = np.array(self.path[1]) - p
                return p, float(np.arctan2(dvec[1], dvec[0])), 0.0
            return p, self.idle_yaw, 0.0
        s = (t - self.started) * self.speed
        pts = np.array(self.path, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = seg.sum()
        if self.loop and total > 0:
            s = s % total
        for i, L in enumerate(seg):
            if s <= L:
                p = pts[i] + (pts[i + 1] - pts[i]) * (s / L)
                dvec = pts[i + 1] - pts[i]
                return p, float(np.arctan2(dvec[1], dvec[0])), s
            s -= L
        dvec = pts[-1] - pts[-2]
        return pts[-1], float(np.arctan2(dvec[1], dvec[0])), None   # finished


@dataclass
class CampusMap:
    route: list                           # waypoint list [(x, y, tag)]
    sidewalks: list                       # [(ax, ay, bx, by, width)]
    road_poly: tuple                      # (xmin, xmax, ymin, ymax)
    crosswalk: tuple                      # (x_entry, x_exit, y)
    delivery: tuple                       # (x, y)
    door: tuple
    bumps: list                           # [(x0,y0,x1,y1,height)]
    slow_zones: list = field(default_factory=list)   # ramps: (xmin, xmax, ymin, ymax, v_max)
    curbs: list = field(default_factory=list)        # plain curb faces beside the far ramp: (x_face, ymin, ymax, height)
    agents: list = field(default_factory=list)


def _box(name, pos, size, mat=None, rgba=None, euler=None, contact=True, extra=""):
    col = "" if contact else 'contype="0" conaffinity="0"'
    m = f'material="{mat}"' if mat else f'rgba="{" ".join(map(str, rgba))}"'
    e = f'euler="{euler[0]:.5f} {euler[1]:.5f} {euler[2]:.5f}"' if euler is not None else ""
    return (f'<geom name="{name}" type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.4f}" '
            f'size="{size[0]:.3f} {size[1]:.3f} {size[2]:.4f}" {m} {e} {col} {extra}/>')


def build_world(robot_xml, seed=3):
    rng = np.random.default_rng(seed)
    G = []   # static geoms
    A = []   # extra assets
    B = []   # mocap bodies

    A.append("""
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.92" rgb2="0.92 0.95 1.0" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="flat" rgb1="0.33 0.55 0.24" rgb2="0.26 0.45 0.18" mark="random" random="0.35" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="1 1" texuniform="true"/>
    <texture name="asphalt" type="2d" builtin="flat" rgb1="0.22 0.22 0.23" rgb2="0.30 0.30 0.31" mark="random" random="0.4" width="256" height="256"/>
    <material name="asphalt" texture="asphalt" texrepeat="2 2" texuniform="true"/>
    <texture name="concrete" type="2d" builtin="checker" rgb1="0.74 0.73 0.70" rgb2="0.70 0.69 0.66" width="256" height="256"/>
    <material name="concrete" texture="concrete" texrepeat="0.8 0.8" texuniform="true" specular="0.1"/>
    <texture name="pavers" type="2d" builtin="checker" rgb1="0.72 0.66 0.60" rgb2="0.66 0.60 0.55" width="256" height="256"/>
    <material name="pavers" texture="pavers" texrepeat="2 2" texuniform="true"/>
    <material name="brick" rgba="0.62 0.34 0.26 1"/>
    <material name="brick2" rgba="0.78 0.74 0.66 1"/>
    <material name="window" rgba="0.25 0.35 0.45 1" specular="1" reflectance="0.2"/>
    <material name="white_paint" rgba="0.95 0.95 0.95 1"/>
    <material name="yellow_paint" rgba="0.95 0.8 0.2 1"/>
    <material name="trunk" rgba="0.36 0.25 0.16 1"/>
    <material name="leaf" rgba="0.20 0.42 0.18 1"/>
    <material name="metal_dark" rgba="0.15 0.16 0.17 1"/>
    <material name="wood" rgba="0.55 0.38 0.22 1"/>
    <material name="skin" rgba="0.80 0.62 0.50 1"/>
    <material name="lamp_glow" rgba="1 0.95 0.8 1" emission="0.6"/>
    <material name="sign_glow" rgba="0.1 0.45 0.7 1" emission="0.5"/>
    <material name="headlamp" rgba="1 1 0.85 1" emission="1"/>
    """)

    # ---------------- ground: asphalt plane at road level, grass blocks above it
    G.append(f'<geom name="road_plane" type="plane" pos="0 0 {ROAD_Z}" size="200 200 0.1" material="asphalt"/>')
    gt = 0.5 * (0.0 - ROAD_Z) + 0.3   # grass block half-height (extends well below)
    def grass(x0, x1, y0, y1, nm):
        G.append(_box(nm, ((x0 + x1) / 2, (y0 + y1) / 2, 0.0 - gt), ((x1 - x0) / 2, (y1 - y0) / 2, gt), mat="grass"))
    grass(-60, 38.5, -60, 80, "grass_w")
    grass(48.5, 110, -60, 80, "grass_e")
    # curb strips left/right of the curb ramps: grass, with a paved apron
    # beside each 1.4 m curb ramp (plain 12.4 cm curb face toward the road)
    grass(38.5, 40.0, -60, 22.8, "curb_w_s"); grass(38.5, 40.0, 25.2, 80, "curb_w_n")
    grass(47.0, 48.5, -60, 21.0, "curb_e_s"); grass(47.0, 48.5, 26.4, 80, "curb_e_n")
    for nm, x0, x1 in (("apron_e", 47.0, 48.5),):
        for k, (y0, y1) in enumerate(((21.0, RAMP_Y0), (RAMP_Y1, 26.4))):
            G.append(_box(f"{nm}{k}", ((x0 + x1) / 2, (y0 + y1) / 2, SW_Z - 0.2), ((x1 - x0) / 2, (y1 - y0) / 2, 0.2), mat="concrete"))

    # ---------------- sidewalks
    sidewalks = []
    def walk(x0, y0, x1, y1, nm, w=SW_W, mat="concrete"):
        sidewalks.append((x0, y0, x1, y1, w))
        if abs(y1 - y0) < 1e-6:     # east-west
            G.append(_box(nm, ((x0 + x1) / 2, y0, SW_Z - 0.02), (abs(x1 - x0) / 2, w / 2, 0.02), mat=mat))
        else:
            G.append(_box(nm, (x0, (y0 + y1) / 2, SW_Z - 0.02), (w / 2, abs(y1 - y0) / 2, 0.02), mat=mat))
    walk(-3.0, 0, 31.2, 0, "sw_S1")
    walk(30, -1.2, 30, 6.0, "sw_S2a")
    walk(30, 19.2, 30, 25.2, "sw_S2b")
    walk(28.8, 24, 38.5, 24, "sw_S3a")
    walk(48.5, 24, 66, 24, "sw_S3b")
    walk(15, 1.2, 15, 22, "sw_branch")                  # branch the robot must NOT take
    walk(60, 25.2, 60, 26.4, "sw_door", w=2.0)
    walk(-12, -5, 12, -5, "sw_far", w=2.0)              # background walk
    sidewalks = sidewalks[:6]

    # ---------------- plaza with 1:12 ADA ramps (rise 0.30 m over 3.6 m)
    a = np.arctan2(0.30, 3.6)
    L = 3.6 / np.cos(a)
    for nm, yc, sgn in (("ramp_up", 7.8, +1), ("ramp_dn", 17.4, -1)):
        top_c = np.array([30, yc, SW_Z + 0.15])
        n = np.array([0, -sgn * np.sin(a), np.cos(a)])
        c = top_c - n * 0.10
        G.append(_box(nm, c, (SW_W / 2, L / 2 + 0.01, 0.10), mat="concrete", euler=(sgn * a, 0, 0)))
    G.append(_box("plaza", (30, 12.6, PLAZA_Z - 0.2), (4.5, 3.0, 0.2), mat="pavers"))
    # side walls under ramps (so there is a drop-off edge to see)
    # ---------------- curb ramps at the crosswalk (drop 0.124 m over 1.5 m)
    for nm, x0, x1, hi_at_x0 in (("cramp_w", 38.5, 40.0, True), ("cramp_e", 47.0, 48.5, False)):
        dz = SW_Z - ROAD_Z
        ang = np.arctan2(dz, x1 - x0)
        top_c = np.array([(x0 + x1) / 2, 24, (SW_Z + ROAD_Z) / 2])
        sg = 1 if hi_at_x0 else -1       # surface descends toward +x when hi at x0
        n = np.array([sg * np.sin(ang), 0, np.cos(ang)])
        c = top_c - n * 0.10
        Lr = (x1 - x0) / np.cos(ang)
        ry0, ry1 = (RAMP_Y0, RAMP_Y1) if nm == "cramp_e" else (22.8, 25.2)   # west ramp: full sidewalk width
        c[1] = 0.5 * (ry0 + ry1)
        G.append(_box(nm, c, (Lr / 2 + 0.01, (ry1 - ry0) / 2, 0.10), mat="concrete", euler=(0, sg * ang, 0)))
    # an e-scooter someone knocked over, lying across the east curb ramp
    G.append(_box("fallen_scooter_deck", (47.85, 24.0, -0.02), (0.09, 0.56, 0.045), rgba=(0.1, 0.1, 0.1, 1), euler=(0.12, 0.0, 0.25)))
    G.append(_box("fallen_scooter_stem", (47.55, 23.45, -0.02), (0.03, 0.40, 0.03), rgba=(0.8, 0.15, 0.1, 1), euler=(0.0, 0.0, 1.35)))
    # a hedge right beside the main sidewalk: a classic blind spot
    G.append(_box("hedge", (22.0, -1.98, 0.6), (1.2, 0.32, 0.6), mat="leaf"))
    # crosswalk stripes + lane line
    for i in range(6):
        y = 24 - 1.25 + i * 0.5
        G.append(_box(f"zebra_{i}", (43.5, y, ROAD_Z + 0.001), (3.5, 0.15, 0.001), mat="white_paint", contact=False))
    for k in range(-20, 30):
        y = k * 3.0
        if 21.5 < y < 26.5:
            continue
        G.append(_box(f"lane_{k}", (43.5, y, ROAD_Z + 0.001), (0.06, 0.8, 0.001), mat="yellow_paint", contact=False))

    # ---------------- sidewalk lips / heaved slabs (real collision)
    bumps = [(6.0, -1.2, 6.0, 1.2, 0.010), (20.0, -1.2, 20.0, 1.2, 0.015), (33.0, 22.8, 33.0, 25.2, 0.020)]
    for i, (x0, y0, x1, y1, h) in enumerate(bumps):
        if x0 == x1:
            G.append(_box(f"lip_{i}", (x0 + 0.3, (y0 + y1) / 2, SW_Z + h / 2), (0.3, abs(y1 - y0) / 2, h / 2), mat="concrete"))

    # ---------------- static obstacles
    G.append(f'<geom name="trashcan" type="cylinder" pos="12 0.62 {SW_Z + 0.45}" size="0.28 0.45" material="metal_dark"/>')
    G.append(f'<geom name="trashcan_lid" type="cylinder" pos="12 0.62 {SW_Z + 0.92}" size="0.30 0.03" rgba="0.2 0.45 0.25 1"/>')
    G.append(_box("scooter_deck", (29.55, 3.3, SW_Z + 0.06), (0.10, 0.55, 0.06), rgba=(0.1, 0.1, 0.1, 1)))
    G.append(_box("scooter_stem", (29.55, 2.75, SW_Z + 0.12), (0.03, 0.03, 0.12), rgba=(0.8, 0.15, 0.1, 1)))
    # benches, lamps, trees, bike rack (off the walking surface)
    for i, (x, y, yaw) in enumerate([(8, 1.75, 0), (17.5, -1.75, 0), (27.2, 12.6, np.pi / 2), (54, 25.7, 0)]):
        z = PLAZA_Z if 9.6 < y < 15.6 and 25 < x < 35 else 0.0
        G.append(_box(f"bench_{i}", (x, y, z + 0.45), (0.9, 0.22, 0.03), mat="wood", euler=(0, 0, yaw)))
        G.append(_box(f"benchb_{i}", (x, y, z + 0.22), (0.8, 0.18, 0.22), mat="metal_dark", euler=(0, 0, yaw)))
    lamps = [(x, s * 1.6) for x in (4, 16, 28) for s in (1, -1)] + [(31.6, y) for y in (4, 21)] + [(x, 25.6) for x in (36, 52)]
    for i, (x, y) in enumerate(lamps):
        G.append(f'<geom name="lamp_{i}" type="cylinder" pos="{x} {y} 1.9" size="0.05 1.9" material="metal_dark"/>')
        G.append(f'<geom name="lampl_{i}" type="sphere" pos="{x} {y} 3.85" size="0.18" material="lamp_glow" contype="0" conaffinity="0"/>')
    trees = [(6, 4.5), (10, -4), (19, 4.2), (24, -4.5), (34.5, 8), (25.5, 17), (36, 28.5), (50, 30.5), (52, 20), (64, 20), (-1, 6), (2, -6), (11, 9), (19, 12)]
    for i, (x, y) in enumerate(trees):
        h = rng.uniform(2.2, 3.2)
        G.append(f'<geom name="trunk_{i}" type="cylinder" pos="{x} {y} {h/2}" size="0.14 {h/2}" material="trunk"/>')
        G.append(f'<geom name="canopy_{i}" type="ellipsoid" pos="{x} {y} {h + 1.0}" size="{rng.uniform(1.3,1.9):.2f} {rng.uniform(1.3,1.9):.2f} 1.4" material="leaf"/>')
    # buildings
    def building(nm, x0, x1, y0, y1, h, mat, door=None):
        G.append(_box(nm, ((x0 + x1) / 2, (y0 + y1) / 2, h / 2), ((x1 - x0) / 2, (y1 - y0) / 2, h / 2), mat=mat))
        # windows on the face nearest the route
        face_y = y0 if y0 > 0 else None
        for k in range(int((x1 - x0) // 3)):
            for fl in range(int(h // 3.5)):
                wx = x0 + 1.5 + 3 * k
                wz = 1.8 + fl * 3.5
                if door and abs(wx - door[0]) < 1.6 and fl == 0:
                    continue
                if y0 > 0:
                    G.append(_box(f"{nm}_w{k}_{fl}", (wx, y0 - 0.02, wz), (0.6, 0.03, 0.8), mat="window", contact=False))
        if door:
            G.append(_box(f"{nm}_door", (door[0], door[1] - 0.03, 1.2), (0.9, 0.04, 1.2), rgba=(0.12, 0.2, 0.3, 1), contact=False))
            G.append(_box(f"{nm}_sign", (door[0], door[1] - 0.05, 2.75), (1.4, 0.03, 0.25), mat="sign_glow", contact=False))
    building("dining_hall", -16, -3.2, -9, 9, 8, "brick")
    G.append(_box("dining_door", (-3.19, 0, 1.2), (0.04, 1.0, 1.2), rgba=(0.15, 0.2, 0.3, 1), contact=False))
    G.append(_box("dining_awning", (-2.6, 0, 2.6), (0.6, 1.6, 0.05), rgba=(0.75, 0.15, 0.12, 1), contact=False))
    building("dorm", 53, 70, 26.4, 40, 12, "brick2", door=(60, 26.4))
    building("library", 20, 34, 30, 44, 10, "brick", door=None)
    building("lab", 36, 46, -22, -8, 9, "brick2")
    building("hall_far", -30, -18, 14, 30, 11, "brick")

    # ---------------- agents (mocap)
    agents = [
        Agent("ped_a", "ped", [(27, 0.40), (-2, 0.40)], 1.25, trigger=lambda p, t: t > 2.0, color=(0.75, 0.2, 0.2)),
        Agent("ped_b", "ped", [(30.35, 12.3)], 0.0, color=(0.2, 0.45, 0.7), idle_yaw=-np.pi / 2),
        Agent("ped_g1", "ped", [(34.0, 24.75)], 0.0, color=(0.55, 0.3, 0.65), idle_yaw=0.3),
        Agent("ped_g2", "ped", [(34.7, 24.95)], 0.0, color=(0.9, 0.65, 0.2), idle_yaw=np.pi + 0.3),
        Agent("ped_c", "ped", [(52.4, 19.5), (52.4, 29.0)], 1.15, trigger=lambda p, t: p[1] > 22 and p[0] > 48.9, color=(0.2, 0.6, 0.35)),
        Agent("car_n", "car", [(45.2, -12), (45.2, 70)], 6.5, trigger=lambda p, t: p[1] > 22 and p[0] > 36.0, color=(0.75, 0.1, 0.12), z=ROAD_Z),
        Agent("car_s", "car", [(41.8, 60), (41.8, -20)], 6.5, trigger=lambda p, t: p[1] > 22 and p[0] > 37.2, color=(0.12, 0.25, 0.6), z=ROAD_Z),
        Agent("ped_cust", "ped", [(60.7, 26.15), (60.35, 24.95)], 0.9, trigger=None, color=(0.95, 0.45, 0.6)),
        Agent("ped_bg", "ped", [(-10, -5), (10, -5)], 1.3, trigger=lambda p, t: True, color=(0.4, 0.4, 0.4), loop=True),
        # a kid darts out from behind the hedge
        Agent("child", "ped", [(22.9, -2.75), (22.9, 3.2)], 2.3, trigger=lambda p, t: 19.8 < p[0] < 26 and abs(p[1]) < 2, color=(0.95, 0.75, 0.15), scale=0.62),
        # a bike comes up fast from behind, then turns off up the branch path
        Agent("bike_a", "bike", [(-7.0, 0.78), (14.2, 0.78), (15.0, 1.6), (15.0, 24.0)], 5.2, trigger=lambda p, t: p[0] > 2.5, color=(0.15, 0.55, 0.85)),
    ]
    for ag in agents:
        c = " ".join(f"{x:.2f}" for x in ag.color)
        k = ag.scale
        if ag.kind == "ped":
            B.append(f"""
    <body name="{ag.name}" mocap="true" pos="{ag.path[0][0]} {ag.path[0][1]} 0">
      <geom type="capsule" fromto="0 0 {0.95*k:.3f} 0 0 {1.45*k:.3f}" size="{0.2*k:.3f}" rgba="{c} 1"/>
      <geom type="sphere" pos="0 0 {1.66*k:.3f}" size="{0.11*max(k,0.8):.3f}" material="skin"/>
      <geom type="sphere" pos="0.01 0 {1.71*k:.3f}" size="{0.105*max(k,0.8):.3f}" rgba="0.2 0.15 0.1 1" contype="0" conaffinity="0"/>
    </body>
    <body name="{ag.name}_l" mocap="true" pos="{ag.path[0][0]} {ag.path[0][1]} 0">
      <geom type="capsule" fromto="0 {0.09*k:.3f} 0.05 0 {0.09*k:.3f} {0.9*k:.3f}" size="{0.075*k:.3f}" rgba="0.2 0.22 0.3 1"/>
    </body>
    <body name="{ag.name}_r" mocap="true" pos="{ag.path[0][0]} {ag.path[0][1]} 0">
      <geom type="capsule" fromto="0 {-0.09*k:.3f} 0.05 0 {-0.09*k:.3f} {0.9*k:.3f}" size="{0.075*k:.3f}" rgba="0.2 0.22 0.3 1"/>
    </body>""")
        elif ag.kind == "bike":
            B.append(f"""
    <body name="{ag.name}" mocap="true" pos="{ag.path[0][0]} {ag.path[0][1]} 0">
      <geom type="cylinder" pos="0.52 0 0.34" size="0.34 0.02" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="cylinder" pos="-0.52 0 0.34" size="0.34 0.02" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="capsule" fromto="-0.5 0 0.36 0.45 0 0.62" size="0.025" rgba="{c} 1"/>
      <geom type="capsule" fromto="-0.15 0 0.62 -0.15 0 1.25" size="0.17" rgba="0.25 0.25 0.3 1"/>
      <geom type="sphere" pos="-0.05 0 1.45" size="0.12" material="skin"/>
      <geom type="sphere" pos="-0.04 0 1.5" size="0.125" rgba="{c} 1" contype="0" conaffinity="0"/>
    </body>""")
        else:
            B.append(f"""
    <body name="{ag.name}" mocap="true" pos="{ag.path[0][0]} {ag.path[0][1]} {ag.z}">
      <geom type="box" pos="0 0 0.55" size="2.25 0.9 0.4" rgba="{c} 1"/>
      <geom type="box" pos="-0.3 0 1.15" size="1.3 0.82 0.3" rgba="0.2 0.25 0.3 1"/>
      <geom type="cylinder" pos="1.4 0.9 0.33" size="0.33 0.1" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="cylinder" pos="-1.4 0.9 0.33" size="0.33 0.1" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="cylinder" pos="1.4 -0.9 0.33" size="0.33 0.1" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="cylinder" pos="-1.4 -0.9 0.33" size="0.33 0.1" zaxis="0 1 0" rgba="0.05 0.05 0.05 1"/>
      <geom type="box" pos="2.26 0.6 0.65" size="0.01 0.15 0.06" material="headlamp" contype="0" conaffinity="0"/>
      <geom type="box" pos="2.26 -0.6 0.65" size="0.01 0.15 0.06" material="headlamp" contype="0" conaffinity="0"/>
    </body>""")

    route = [(-0.8, 0.0, "start"), (30.0, 0.0, "corner"), (30.0, 24.0, "corner"),
             (38.3, 24.0, "xwalk_in"), (48.7, 24.0, "xwalk_out"), (60.0, 24.0, "deliver")]
    cmap = CampusMap(route=route, sidewalks=sidewalks, road_poly=(40.0, 47.0, -60, 80),
                     crosswalk=(38.3, 48.7, 24.0), delivery=(60.0, 24.0), door=(60.0, 26.4),
                     bumps=bumps, agents=agents,
                     slow_zones=[(28.6, 31.4, 5.6, 10.2, 1.0), (28.6, 31.4, 15.0, 19.6, 1.0), (37.5, 40.2, 22.6, 25.4, 0.9), (46.8, 49.5, 22.6, 25.4, 0.9)],
                     curbs=[(47.0, 21.0, 23.3, SW_Z - ROAD_Z), (47.0, 24.7, 26.4, SW_Z - ROAD_Z)])

    xml = f"""
<mujoco model="orb_campus">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="3"/>
  <size memory="64M"/>
  <statistic extent="40" center="30 12 0"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="40"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <map znear="0.001" zfar="12" haze="0.25"/>
    <rgba haze="0.85 0.9 0.97 1"/>
  </visual>
  <default><geom friction="0.9 0.012 0.0035"/></default>
  <asset>
    {''.join(A)}
    {robot_xml['assets']}
  </asset>
  <worldbody>
    <light name="sun" pos="10 -10 30" dir="-0.35 0.45 -1" directional="true" castshadow="true" diffuse="0.75 0.73 0.68" specular="0.2 0.2 0.2"/>
    {''.join(G)}
    {''.join(B)}
    {robot_xml['body']}
  </worldbody>
  <contact>{robot_xml['contact']}</contact>
  <tendon>{robot_xml['tendon']}</tendon>
  <equality>{robot_xml['equality']}</equality>
  <actuator>{robot_xml['actuators']}</actuator>
  <sensor>{robot_xml['sensors']}</sensor>
</mujoco>"""
    return xml, cmap
