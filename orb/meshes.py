"""Procedural meshes (band with hatch cut-out, lid, side caps, food pod).

All meshes are visual only (the rolling contact is an exact MuJoCo sphere).
Surfaces are emitted double-sided so the renderer never shows a missing face.
"""
import numpy as np


def _sph(r, lon, lat):
    """Point on sphere. lon = angle about the axle (y) measured from +z toward +x,
    lat = angle from the rolling plane toward +y."""
    return np.array([r * np.cos(lat) * np.sin(lon), r * np.sin(lat), r * np.cos(lat) * np.cos(lon)])


def _grid(breaks, step):
    """Monotone grid through all break points with spacing <= step."""
    out = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(np.ceil((b - a) / step)))
        out.extend(np.linspace(a, b, n + 1)[:-1])
    out.append(breaks[-1])
    return np.array(out)


class MeshBuilder:
    def __init__(self):
        self.v, self.f = [], []

    def add_v(self, p):
        self.v.append(np.asarray(p, float))
        return len(self.v) - 1

    def tri(self, a, b, c, double=True):
        self.f.append((a, b, c))
        if double:
            self.f.append((a, c, b))

    def quad(self, a, b, c, d, double=True):
        self.tri(a, b, c, double)
        self.tri(a, c, d, double)

    def mjcf(self, name, scale=1.0):
        v = " ".join(f"{x*scale:.5f}" for p in self.v for x in p)
        f = " ".join(str(i) for t in self.f for i in t)
        return f'<mesh name="{name}" vertex="{v}" face="{f}" inertia="shell"/>'


def band_mesh(R, wall, band_lat, hatch_lon, hatch_lat, seam=np.radians(0.7), offset=np.zeros(3)):
    """Rolling band (|lat| <= band_lat) with a rectangular hatch hole centred at lon=0."""
    hl, ht = hatch_lon + seam, hatch_lat + seam
    lons = _grid([-np.pi, -hl, hl, np.pi], np.radians(3.0))
    lats = _grid([-band_lat, -ht, ht, band_lat], np.radians(2.5))
    mb = MeshBuilder()
    idx = {}
    for s, r in (("o", R), ("i", R - wall)):
        for i, lo in enumerate(lons):
            for j, la in enumerate(lats):
                idx[(s, i, j)] = mb.add_v(_sph(r, lo, la) - offset)
    edges = {}
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            clo, cla = 0.5 * (lons[i] + lons[i + 1]), 0.5 * (lats[j] + lats[j + 1])
            if abs(clo) < hl and abs(cla) < ht:
                continue  # hatch opening
            for s in ("o", "i"):
                mb.quad(idx[(s, i, j)], idx[(s, i + 1, j)], idx[(s, i + 1, j + 1)], idx[(s, i, j + 1)])
            for e in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                      ((i + 1, j + 1), (i, j + 1)), ((i, j + 1), (i, j))):
                k = tuple(sorted(e))
                edges[k] = edges.get(k, 0) + 1
    # close every boundary edge (rims + hatch hole) with a wall strip
    for (a, b), n in edges.items():
        if n == 1:
            mb.quad(idx[("o",) + a], idx[("o",) + b], idx[("i",) + b], idx[("i",) + a])
    return mb


def lid_mesh(R, wall, hatch_lon, hatch_lat, hinge_pt, seam=np.radians(0.35)):
    hl, ht = hatch_lon - seam, hatch_lat - seam
    lons = _grid([-hl, hl], np.radians(2.5))
    lats = _grid([-ht, ht], np.radians(2.5))
    mb = MeshBuilder()
    idx = {}
    for s, r in (("o", R), ("i", R - wall)):
        for i, lo in enumerate(lons):
            for j, la in enumerate(lats):
                idx[(s, i, j)] = mb.add_v(_sph(r, lo, la) - hinge_pt)
    ni, nj = len(lons), len(lats)
    for i in range(ni - 1):
        for j in range(nj - 1):
            for s in ("o", "i"):
                mb.quad(idx[(s, i, j)], idx[(s, i + 1, j)], idx[(s, i + 1, j + 1)], idx[(s, i, j + 1)])
    ring = [(i, 0) for i in range(ni)] + [(ni - 1, j) for j in range(nj)] + \
           [(i, nj - 1) for i in reversed(range(ni))] + [(0, j) for j in reversed(range(nj))]
    for a, b in zip(ring[:-1], ring[1:]):
        if a != b:
            mb.quad(idx[("o",) + a], idx[("o",) + b], idx[("i",) + b], idx[("i",) + a])
    return mb


def cap_mesh(R, wall, band_lat, side, gap=np.radians(0.6)):
    """Stationary side cap from latitude band_lat+gap to the pole on side +1/-1."""
    th_max = np.pi / 2 - band_lat - gap   # polar angle from the axle
    ths = np.linspace(0.0, th_max, 24)
    phs = np.linspace(0, 2 * np.pi, 97)[:-1]
    mb = MeshBuilder()
    idx = {}
    for s, r in (("o", R), ("i", R - wall)):
        pole = mb.add_v([0, side * r, 0])
        idx[(s, "pole")] = pole
        for i, th in enumerate(ths[1:], 1):
            for k, ph in enumerate(phs):
                idx[(s, i, k)] = mb.add_v([r * np.sin(th) * np.cos(ph), side * r * np.cos(th), r * np.sin(th) * np.sin(ph)])
    nk = len(phs)
    for s in ("o", "i"):
        for k in range(nk):
            mb.tri(idx[(s, "pole")], idx[(s, 1, k)], idx[(s, 1, (k + 1) % nk)])
        for i in range(1, len(ths) - 1):
            for k in range(nk):
                mb.quad(idx[(s, i, k)], idx[(s, i + 1, k)], idx[(s, i + 1, (k + 1) % nk)], idx[(s, i, (k + 1) % nk)])
    last = len(ths) - 1
    for k in range(nk):
        mb.quad(idx[("o", last, k)], idx[("o", last, (k + 1) % nk)], idx[("i", last, (k + 1) % nk)], idx[("i", last, k)])
    return mb


def pod_mesh(r, top, half_w, ins, n=40):
    """Open-top insulated drum section hanging on the axle (body frame, axle = y)."""
    def profile(rr, zt):
        a = np.arccos(np.clip(zt / rr, -1, 1))       # angle from +z where chord meets circle
        angs = np.linspace(a, 2 * np.pi - a, n)       # sweep around the bottom
        return np.stack([rr * np.sin(angs), rr * np.cos(angs)], 1)   # (x, z)
    O = profile(r, top)
    I = profile(r - ins, top)
    wo, wi = half_w, half_w - ins
    mb = MeshBuilder()
    def ring(prof, y):
        return [mb.add_v([x, y, z]) for x, z in prof]
    oL, oR, iL, iR = ring(O, -wo), ring(O, wo), ring(I, -wi), ring(I, wi)
    for k in range(n - 1):
        mb.quad(oL[k], oL[k + 1], oR[k + 1], oR[k])
        mb.quad(iL[k], iL[k + 1], iR[k + 1], iR[k])
    # end walls (fan) outer and inner
    for ringpts in (oL, oR, iL, iR):
        for k in range(1, n - 1):
            mb.tri(ringpts[0], ringpts[k], ringpts[k + 1])
    # top rim: 4 quads between outer rectangle and inner opening
    xo, xi = O[0, 0], I[0, 0]   # O[0] is the +x end of the chord
    def v(x, y):
        return mb.add_v([x, y, top])
    A, B, C, D = v(-xo, -wo), v(xo, -wo), v(xo, wo), v(-xo, wo)
    a, b, c, d = v(-xi, -wi), v(xi, -wi), v(xi, wi), v(-xi, wi)
    mb.quad(A, B, b, a); mb.quad(B, C, c, b); mb.quad(C, D, d, c); mb.quad(D, A, a, d)
    return mb


def seam_ring_mesh(R, lat0, lat1, side, n=128):
    """Thin glowing strip on the sphere between two latitudes (the cap seam)."""
    mb = MeshBuilder()
    lons = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    a = [mb.add_v(_sph(R, lo, side * lat0)) for lo in lons]
    b = [mb.add_v(_sph(R, lo, side * lat1)) for lo in lons]
    for k in range(n):
        k2 = (k + 1) % n
        mb.quad(a[k], a[k2], b[k2], b[k])
    return mb
