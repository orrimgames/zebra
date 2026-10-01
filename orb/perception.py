"""Onboard perception, driven only by the robot's simulated cameras.

Each 10 Hz frame:
  1. render RGB + depth from the stereo pair in the caps (160x120 rectified
     crops of the fisheyes); depth gets stereo-like noise (sigma ~ z^2) and
     dropouts, so the pipeline sees what a real stereo matcher would give.
  2. classify every pixel by colour: walkable pavement / grass / asphalt /
     paint / unknown.
  3. back-project to 3-D with the known camera extrinsics and drop the points
     into a robot-centred bird's-eye grid (BEV, 10 cm cells).
  4. per cell: height spread and height vs. a robust local ground plane ->
     obstacle (anything >= 9 cm proud: people, bins, scooters, curbs going up)
     and drop-off (>= 7 cm below: curb edges, plaza edges).
Outputs the BEV layers the planner uses, plus overlays for the video.
"""
import numpy as np
import mujoco
from scipy.ndimage import minimum_filter

WALK, GRASS, ASPHALT, PAINT, UNKNOWN = 1, 2, 3, 4, 0


class BEV:
    def __init__(self, x0=-0.6, x1=9.0, y0=-3.0, y1=3.0, res=0.1):
        self.x0, self.x1, self.y0, self.y1, self.res = x0, x1, y0, y1, res
        self.nx = int(round((x1 - x0) / res)); self.ny = int(round((y1 - y0) / res))
        self.clear()

    def clear(self):
        s = (self.nx, self.ny)
        self.count = np.zeros(s, np.int32)
        self.zmin = np.full(s, np.inf); self.zmax = np.full(s, -np.inf)
        self.votes = np.zeros(s + (5,), np.int32)
        self.obst = np.zeros(s, bool); self.drop = np.zeros(s, bool)
        self.label = np.zeros(s, np.int8)
        self.seen = np.zeros(s, bool)

    def idx(self, x, y):
        return ((x - self.x0) / self.res).astype(int), ((y - self.y0) / self.res).astype(int)

    def cell_center(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res


class Perception:
    def __init__(self, m, cams=("cam_left", "cam_right"), W=160, H=120, seed=0):
        self.m = m
        self.W, self.H = W, H
        self.r = mujoco.Renderer(m, H, W)
        self.cams = list(cams)
        self.cam_ids = {c: m.camera(c).id for c in ("cam_left", "cam_right", "cam_left_side", "cam_right_side")}
        self.rng = np.random.default_rng(seed)
        fovy = np.radians(m.cam_fovy[self.cam_ids["cam_left"]])
        self.f = (H / 2) / np.tan(fovy / 2)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self.ray = np.stack([(u - W / 2) / self.f, -(v - H / 2) / self.f, -np.ones_like(u)], -1)  # cam frame, per unit depth
        self.bev = BEV()
        self.last = {}
        self.ground = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ pixels
    @staticmethod
    def classify(rgb):
        f = rgb.astype(np.float32) / 255.0
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        mx, mn = f.max(-1), f.min(-1)
        sat = (mx - mn) / (mx + 1e-6)
        lab = np.full(r.shape, UNKNOWN, np.int8)
        grass = (g > r + 0.05) & (g > b + 0.04)
        paint = (mx > 0.93) & (sat < 0.08)
        asphalt = (mx < 0.36) & (sat < 0.22) & ~grass
        walk = (mx >= 0.36) & (sat < 0.30) & ~grass & ~paint & (r >= b - 0.02)
        lab[walk] = WALK; lab[grass] = GRASS; lab[asphalt] = ASPHALT; lab[paint] = PAINT
        return lab

    def _update(self, d, cam):
        self.r.update_scene(d, cam)
        sc = self.r.scene
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    def _render(self, d, cam):
        self._update(d, cam)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self._update(d, cam)
        dep = self.r.render().copy()
        self.r.disable_depth_rendering()
        return rgb, dep

    # ------------------------------------------------------------------ frame
    def step(self, d, robot_pos, heading, cams=None):
        cams = cams or self.cams
        c, s = np.cos(heading), np.sin(heading)
        Rw2r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
        ground_z = robot_pos[2] - 0.325
        pts, labs = [], []
        self.last = {}
        for cam in cams:
            rgb, dep = self._render(d, cam)
            # stereo-like noise + dropouts
            z = dep.astype(np.float64)
            noise = self.rng.normal(0, 1, z.shape) * (0.004 * z ** 2 + 0.005)
            zn = z + noise
            valid = (z > 0.15) & (z < 10.0) & (self.rng.random(z.shape) > 0.03)
            cid = self.cam_ids[cam]
            Rc = d.cam_xmat[cid].reshape(3, 3); pc = d.cam_xpos[cid]
            P = (self.ray * zn[..., None]) @ Rc.T + pc          # world points
            lab = self.classify(rgb)
            pts.append(P[valid]); labs.append(lab[valid])
            self.last[cam] = dict(rgb=rgb, depth=z, label=lab, world=P[valid])
        P = np.concatenate(pts); L = np.concatenate(labs)
        # to robot frame (origin at contact point, x forward along heading)
        Pr = (P - np.array([robot_pos[0], robot_pos[1], ground_z])) @ Rw2r.T
        self._accumulate(Pr, L)
        return self.bev

    def _accumulate(self, Pr, L):
        b = self.bev; b.clear()
        i, j = b.idx(Pr[:, 0], Pr[:, 1])
        ok = (i >= 0) & (i < b.nx) & (j >= 0) & (j < b.ny) & (Pr[:, 2] < 2.2)
        i, j, z, L = i[ok], j[ok], Pr[ok, 2], L[ok]
        flat = i * b.ny + j
        np.add.at(b.count.reshape(-1), flat, 1)
        np.minimum.at(b.zmin.reshape(-1), flat, z)
        np.maximum.at(b.zmax.reshape(-1), flat, z)
        low = z < b.zmin.reshape(-1)[flat] + 0.04
        np.add.at(b.votes.reshape(-1, 5), (flat[low], L[low]), 1)
        b.seen = b.count >= 1
        b.label = np.where(b.seen, b.votes.argmax(-1), UNKNOWN).astype(np.int8)
        zmin = np.where(b.seen, b.zmin, np.inf)
        # local ground = lowest surface within +/-0.25 m (handles ramps; a
        # single global plane cannot)
        gloc = minimum_filter(zmin, size=5, mode="nearest")
        spread = np.where(b.seen, b.zmax - b.zmin, 0)
        b.obst = b.seen & ((spread > 0.10) | (b.zmax - gloc > 0.10))
        # step edges: neighbouring surfaces differ by > 7 cm over 10 cm
        # (curbs 12 cm, plaza edge 30 cm) -> not traversable; 1:12 ramps
        # (0.8 cm per cell) and 1-2 cm sidewalk lips are fine.
        st = np.zeros_like(b.seen)
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = zmin[max(di, 0):b.nx + min(di, 0) or None, max(dj, 0):b.ny + min(dj, 0) or None]
            c = zmin[max(-di, 0):b.nx + min(-di, 0) or None, max(-dj, 0):b.ny + min(-dj, 0) or None]
            e = np.isfinite(a) & np.isfinite(c) & (np.abs(a - c) > 0.07)
            st[max(di, 0):b.nx + min(di, 0) or None, max(dj, 0):b.ny + min(dj, 0) or None] |= e
            st[max(-di, 0):b.nx + min(-di, 0) or None, max(-dj, 0):b.ny + min(-dj, 0) or None] |= e
        b.drop = st & ~b.obst
        o = b.obst.copy()
        o[1:, :] |= b.obst[:-1, :]; o[:-1, :] |= b.obst[1:, :]; o[:, 1:] |= b.obst[:, :-1]; o[:, :-1] |= b.obst[:, 1:]
        b.obst = o
        # fill label holes from neighbours (sparse far rows) for edge finding
        lab = b.label.copy()
        for _ in range(2):
            hole = lab == UNKNOWN
            up = np.roll(lab, 1, 0); dn = np.roll(lab, -1, 0)
            fill = np.where(up != UNKNOWN, up, dn)
            lab = np.where(hole, fill, lab)
        b.label_filled = lab

    # ------------------------------------------------------------------ viz
    def bev_image(self, scale=4, path_pts=None, corridor=None):
        b = self.bev
        img = np.full((b.nx, b.ny, 3), 30, np.uint8)
        col = {WALK: (185, 185, 180), GRASS: (60, 120, 50), ASPHALT: (70, 70, 80), PAINT: (240, 240, 240)}
        for k, c in col.items():
            img[getattr(b, "label_filled", b.label) == k] = c
        img[b.drop] = (230, 150, 30)
        img[b.obst] = (220, 40, 50)
        img = img[::-1, ::-1]     # x up, y left
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)
        return img
