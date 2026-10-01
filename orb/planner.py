"""Route following, localisation, obstacle handling and the delivery state machine.

Everything here uses only what a real robot would have:
  * the campus map (sidewalk centre-lines, crosswalk and delivery points)
  * noisy odometry + gyro, noisy GNSS (metre-level drift)
  * the perception BEV from the stereo cameras
"""
import numpy as np
from scipy import ndimage
from .perception import WALK, GRASS, ASPHALT, PAINT, UNKNOWN


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Route:
    """Route through the sidewalk map, with every 90-degree corner replaced by
    an arc (a pendulum sphere turns like a bicycle, it cannot pivot)."""

    def __init__(self, pts, r_corner=2.2, ds=0.05):
        V = np.array([(p[0], p[1]) for p in pts], float)
        self.V, self.vtags = V, [p[2] for p in pts]
        out = [V[0]]
        for i in range(1, len(V) - 1):
            a, b, c = V[i - 1], V[i], V[i + 1]
            t1 = (b - a) / np.linalg.norm(b - a); t2 = (c - b) / np.linalg.norm(c - b)
            ang = wrap(np.arctan2(t2[1], t2[0]) - np.arctan2(t1[1], t1[0]))
            if abs(ang) < 1e-3:
                out.append(b); continue
            dcut = r_corner * np.tan(abs(ang) / 2)
            p1, p2 = b - t1 * dcut, b + t2 * dcut
            n1 = np.array([-t1[1], t1[0]]) * np.sign(ang)
            cen = p1 + n1 * r_corner
            a0 = np.arctan2(p1[1] - cen[1], p1[0] - cen[0])
            for k in np.linspace(0, 1, max(4, int(abs(ang) * r_corner / ds))):
                th = a0 + np.sign(ang) * abs(ang) * k
                out.append(cen + r_corner * np.array([np.cos(th), np.sin(th)]))
        out.append(V[-1])
        # resample uniformly
        P = np.array(out)
        segL = np.linalg.norm(np.diff(P, axis=0), axis=1)
        keep = np.concatenate([[True], segL > 1e-6])
        P = P[keep]
        S = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        n = int(S[-1] / ds) + 1
        ss = np.linspace(0, S[-1], n)
        self.P = np.stack([np.interp(ss, S, P[:, 0]), np.interp(ss, S, P[:, 1])], 1)
        seg = np.diff(self.P, axis=0)
        self.L = np.linalg.norm(seg, axis=1)
        self.S = np.concatenate([[0], np.cumsum(self.L)])
        self.T = seg / self.L[:, None]
        self.N = np.stack([-self.T[:, 1], self.T[:, 0]], 1)
        self.H = np.unwrap(np.arctan2(self.T[:, 1], self.T[:, 0]))
        k = np.gradient(self.H, self.S[:-1])
        self.kappa = np.abs(np.convolve(k, np.ones(5) / 5, mode="same"))
        self.total = self.S[-1]

    def s_of_tag(self, tag, nth=0):
        idx = [i for i, t in enumerate(self.vtags) if t == tag][nth]
        return self.project(self.V[idx])[0]

    def seg_at(self, s):
        return int(np.clip(np.searchsorted(self.S, s, side="right") - 1, 0, len(self.L) - 1))

    def point(self, s, offset=0.0):
        i = self.seg_at(s)
        p = self.P[i] + self.T[i] * (s - self.S[i])
        return p + self.N[i] * offset

    def heading(self, s):
        t = self.T[self.seg_at(s)]
        return np.arctan2(t[1], t[0])

    def project(self, p, s_hint=None, window=6.0):
        lo, hi = 0, len(self.L)
        if s_hint is not None:
            lo = max(0, self.seg_at(s_hint - window)); hi = min(len(self.L), self.seg_at(s_hint + window) + 1)
        A = self.P[lo:hi]; T = self.T[lo:hi]; L = self.L[lo:hi]
        q = p[None, :] - A
        t = np.clip(np.sum(q * T, 1), 0, L)
        r = q - T * t[:, None]
        dist = np.linalg.norm(r, axis=1)
        k = int(np.argmin(dist))
        i = lo + k
        return self.S[i] + t[k], float(r[k] @ self.N[i]), i

    def straight_extent(self, s):
        """(distance since last curve, distance to next curve) along the path."""
        i = self.seg_at(s)
        curvy = self.kappa > 0.05
        j = i
        while j < len(curvy) - 1 and not curvy[j]:
            j += 1
        k = i
        while k > 0 and not curvy[k]:
            k -= 1
        return s - self.S[k] if curvy[k] else s, self.S[j] - s if curvy[j] else self.total - s

    def v_curve(self, s, v_of_kappa, a_dec=0.35, horizon=5.0):
        i0 = self.seg_at(s); i1 = self.seg_at(s + horizon)
        v = 9.0
        for i in range(i0, i1 + 1):
            vk = v_of_kappa(self.kappa[i])
            v = min(v, np.sqrt(vk * vk + 2 * a_dec * max(0.0, self.S[i] - s)))
        return v


class Estimator:
    """Dead reckoning + GNSS + map-relative vision corrections."""

    def __init__(self, x, y, psi):
        self.x, self.y, self.psi = x, y, psi
        self.n_vis = 0

    def predict(self, ds, dpsi):
        self.psi = wrap(self.psi + dpsi)
        self.x += ds * np.cos(self.psi)
        self.y += ds * np.sin(self.psi)

    def gnss(self, z, k=0.03):
        self.x += k * (z[0] - self.x); self.y += k * (z[1] - self.y)

    def lateral(self, innov, normal, k=0.25):
        self.x += k * innov * normal[0]; self.y += k * innov * normal[1]
        self.n_vis += 1

    def along(self, innov, tangent, k=0.2):
        self.x += k * innov * tangent[0]; self.y += k * innov * tangent[1]
        self.n_along = getattr(self, "n_along", 0) + 1

    def heading_fix(self, dpsi, k=0.08):
        self.psi = wrap(self.psi + k * dpsi)

    @property
    def p(self):
        return np.array([self.x, self.y])


class SensorSim:
    """Noise models for odometry, gyro and GNSS (seeded)."""

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.odo_scale = 1.0 + self.rng.normal(0, 0.015)
        self.gyro_bias = self.rng.normal(0, 0.002)
        self.gb = np.array([self.rng.normal(0, 1.0), self.rng.normal(0, 1.0)])   # GNSS bias (m)
        self.tau = 60.0

    def odom(self, ds):
        return ds * self.odo_scale + self.rng.normal(0, 0.002) * np.sqrt(abs(ds) + 1e-9)

    def gyro(self, r, dt):
        return (r + self.gyro_bias + self.rng.normal(0, 0.01)) * dt

    def gnss(self, p, dt=1.0):
        a = np.exp(-dt / self.tau)
        self.gb = a * self.gb + np.sqrt(1 - a * a) * self.rng.normal(0, 1.0, 2)
        return p + self.gb + self.rng.normal(0, 0.35, 2)


class Planner:
    CRUISE = 1.4
    HALF_W = 0.325
    MARGIN = 0.18

    def __init__(self, cmap, lowlevel):
        self.map = cmap
        self.route = Route(cmap.route)
        self.ll = lowlevel
        self.state = "DRIVE"
        self.t_state = 0.0
        self.s = 0.0
        self.e = 0.0
        self.o_cmd = -0.30
        self.s_xin = self.route.s_of_tag("xwalk_in")
        self.s_xout = self.route.s_of_tag("xwalk_out")
        self.s_del = self.route.s_of_tag("deliver")
        self.road_clear_t = 0.0
        self.yield_t = 0.0
        self.free = 9.9
        self.msg = ""
        self.s_final = None
        self.events = []
        self.gaze = (0.0, 0.0)
        self.corridor = None
        self.odo_s = 0.0          # along-track odometry for the final alignment
        self.crossed = False
        self.tracks = []
        self.sidestep_t = 0.0
        self.sidestep_o = 0.0
        self.blocked_t = 0.0
        self.backup_left = 0.0
        self.backup_kind = "swing"
        self.charge_t = 0.0
        self.d_emerg = 9.9

    def set_state(self, st, t):
        if st != self.state:
            self.events.append((round(t, 2), self.state, st))
            self.state, self.t_state = st, t

    # --------------------------------------------------------------- vision -> localisation
    def vision_lateral(self, est, bev):
        s, e, i = self.route.project(est.p, self.s)
        if self.state.startswith("XWALK") or self.s_xin - 1.0 < s < self.s_xout + 1.0:
            return None
        since, until = self.route.straight_extent(s)
        if until < 3.2 or since < 1.0:     # near corners
            return None
        # plaza section has no edges to see (wide)
        dpsi = wrap(self.route.heading(s) - est.psi)
        u = np.array([np.cos(dpsi), np.sin(dpsi)]); n = np.array([-np.sin(dpsi), np.cos(dpsi)])
        ts = np.arange(-2.6, 2.6, 0.05)
        centers, xs = [], []
        lab = getattr(bev, "label_filled", bev.label)
        for xf in (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8):
            pts = xf * u[None, :] + ts[:, None] * n[None, :]
            ii, jj = bev.idx(pts[:, 0], pts[:, 1])
            ok = (ii >= 0) & (ii < bev.nx) & (jj >= 0) & (jj < bev.ny)
            L = np.full(len(ts), UNKNOWN)
            L[ok] = lab[ii[ok], jj[ok]]
            walk = (L == WALK)
            k0 = int(np.argmin(np.abs(ts + e)))          # expected centre
            if not walk[k0]:
                cand = np.nonzero(walk)[0]
                if len(cand) == 0:
                    continue
                k0 = cand[np.argmin(np.abs(cand - k0))]
            a = k0
            while a > 0 and walk[a - 1]:
                a -= 1
            b = k0
            while b < len(ts) - 1 and walk[b + 1]:
                b += 1
            if a == 0 or b == len(ts) - 1:
                continue
            if L[a - 1] not in (GRASS, ASPHALT) or L[b + 1] not in (GRASS, ASPHALT):
                continue
            w = ts[b] - ts[a] + 0.05
            if not (2.0 < w < 2.8):
                continue
            centers.append(0.5 * (ts[a] + ts[b])); xs.append(xf)
        if len(centers) < 2:
            return None
        c = float(np.median(centers))
        e_meas = -c
        slope = np.polyfit(xs, centers, 1)[0] if len(xs) >= 4 else 0.0
        return e_meas - e, self.route.N[i], -np.arctan(slope)

    def vision_along(self, est, bev):
        """Along-track fix: at an L-corner the pavement ends 1.2 m past the
        corner vertex; measure that edge straight ahead in the BEV."""
        R = self.route
        s, e, _ = R.project(est.p, self.s)
        V = R.V
        for k in range(1, len(V) - 1):
            if R.vtags[k] != "corner":
                continue
            t_in = (V[k] - V[k - 1]) / np.linalg.norm(V[k] - V[k - 1])
            end = V[k] + t_in * 1.2
            D_pred = float((end - est.p) @ t_in)
            if not (1.8 < D_pred < 5.2):
                continue
            dpsi = wrap(np.arctan2(t_in[1], t_in[0]) - est.psi)
            if abs(dpsi) > np.radians(20):
                return None
            u = np.array([np.cos(dpsi), np.sin(dpsi)]); n = np.array([-np.sin(dpsi), np.cos(dpsi)])
            lab = getattr(bev, "label_filled", bev.label)
            xs = np.arange(1.0, 5.8, 0.05)
            hits = []
            for lat in (-e - 0.4, -e, -e + 0.4):
                pts = xs[:, None] * u[None, :] + lat * n[None, :]
                ii, jj = bev.idx(pts[:, 0], pts[:, 1])
                ok = (ii >= 0) & (ii < bev.nx) & (jj >= 0) & (jj < bev.ny)
                L = np.full(len(xs), UNKNOWN); L[ok] = lab[ii[ok], jj[ok]]
                for a in range(len(xs) - 6):
                    if L[a] == WALK and np.all(L[a + 1:a + 7] == GRASS):
                        hits.append(xs[a] + 0.025); break
            if len(hits) >= 2:
                D_meas = float(np.median(hits))
                if abs(D_meas - D_pred) < 1.5:
                    return D_pred - D_meas, t_in
            return None
        return None

    # --------------------------------------------------------------- moving obstacles
    def track(self, t, est, bev):
        """Cluster obstacle cells, track centroids in the world frame, estimate velocity."""
        lab, n = ndimage.label(bev.obst)
        dets = []
        if n:
            idx = range(1, n + 1)
            sizes = ndimage.sum(np.ones_like(lab), lab, idx)
            coms = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
            c, s_ = np.cos(est.psi), np.sin(est.psi)
            for sz, (i, j) in zip(sizes, coms):
                if sz < 3 or sz > 400:
                    continue
                x, y = bev.x0 + (i + 0.5) * bev.res, bev.y0 + (j + 0.5) * bev.res
                if x > 5.5:
                    continue
                dets.append(np.array([est.x + c * x - s_ * y, est.y + s_ * x + c * y]))
        new = []
        for dpos in dets:
            best, bd = None, 0.8
            for tr in self.tracks:
                dd = np.linalg.norm(tr["p"] - dpos)
                if dd < bd and not tr.get("_used"):
                    best, bd = tr, dd
            if best is None:
                new.append(dict(p=dpos, v=np.zeros(2), t=t, age=1))
            else:
                dtt = max(t - best["t"], 1e-3)
                best["v"] = 0.6 * best["v"] + 0.4 * (dpos - best["p"]) / dtt
                best["p"], best["t"], best["age"], best["_used"] = dpos, t, best["age"] + 1, True
        self.tracks = [tr for tr in self.tracks if t - tr["t"] < 0.6] + new
        for tr in self.tracks:
            tr.pop("_used", None)

    def predict_conflict(self, est, st, horizon=3.0):
        """Earliest predicted conflict with a moving track along our planned path."""
        R = self.route
        v_plan = max(abs(st["v"]), 0.7)
        worst = None
        for tr in self.tracks:
            sp = np.linalg.norm(tr["v"])
            if tr["age"] < 4 or sp < 0.45 or sp > 3.0:
                continue
            for tau in np.arange(0.0, horizon, 0.2):
                pr = R.point(min(self.s + v_plan * tau, R.total), self.o_cmd)
                pp = tr["p"] + tr["v"] * tau
                if np.linalg.norm(pr - pp) < 0.95:
                    hd = R.heading(self.s)
                    rel = wrap(np.arctan2(tr["v"][1], tr["v"][0]) - hd)
                    q = tr["p"] - R.point(self.s)
                    lat = float(q @ np.array([-np.sin(hd), np.cos(hd)]))
                    if worst is None or tau < worst[0]:
                        worst = (tau, rel, lat, tr)
                    break
        return worst

    # --------------------------------------------------------------- corridor check
    def corridor_free(self, est, bev, o, s0, length, allow_road=False):
        lab = getattr(bev, "label_filled", bev.label)
        c, sn = np.cos(-est.psi), np.sin(-est.psi)
        hw = self.HALF_W + self.MARGIN
        for ds in np.arange(0.5, length, 0.1):
            p = self.route.point(s0 + ds, o)
            q = p - est.p
            x, y = c * q[0] - sn * q[1], sn * q[0] + c * q[1]
            dist = np.hypot(x, y)
            # robot-frame lateral samples across the footprint
            h = self.route.heading(s0 + ds) - est.psi
            nx_, ny_ = -np.sin(h), np.cos(h)
            for t in (-hw, -hw / 2, 0, hw / 2, hw):
                xi, yi = x + t * nx_, y + t * ny_
                i, j = int((xi - bev.x0) / bev.res), int((yi - bev.y0) / bev.res)
                if not (0 <= i < bev.nx and 0 <= j < bev.ny):
                    continue
                if bev.obst[i, j] or bev.drop[i, j]:
                    return dist
                L = lab[i, j]
                if L == GRASS or (L == ASPHALT and not allow_road):
                    return dist
        return length

    def straight_free(self, bev, allow_road, length=2.6):
        hw = self.HALF_W + 0.12
        for x in np.arange(0.45, length, 0.1):
            for y in np.linspace(-hw, hw, 7):
                i, j = int((x - bev.x0) / bev.res), int((y - bev.y0) / bev.res)
                if 0 <= i < bev.nx and 0 <= j < bev.ny and (bev.obst[i, j] or bev.drop[i, j]):
                    return x
        return length

    # --------------------------------------------------------------- main tick (10 Hz)
    def tick(self, t, est, bev, st, road_objects, dt):
        R = self.route
        s, e, _ = R.project(est.p, self.s)
        self.s, self.e = s, e
        v_lim = self.CRUISE
        yaw_cmd = 0.0
        # asphalt is only ever drivable in XWALK_CROSS, which can only be
        # entered from XWALK_WAIT after the road has been seen clear
        allow_road = self.state == "XWALK_CROSS"
        in_plaza = 9.0 < est.p[1] < 16.0 and abs(est.p[0] - 30) < 3
        o_pref = -0.30 if not (allow_road or in_plaza) else -0.1

        # ------------------------------ state machine
        if self.state == "DRIVE":
            if self.s_xin - 8.0 < s < self.s_xout and not self.crossed:
                # braking profile to the stop line (0.4 m/s^2)
                v_lim = min(v_lim, np.sqrt(2 * 0.4 * max(0.0, self.s_xin - 0.1 - s)))
                if s > self.s_xin - 0.45 and abs(st["v"]) < 0.08:
                    self.set_state("XWALK_WAIT", t)
            if s > self.s_del - 4.0:
                self.set_state("APPROACH", t)
        elif self.state == "XWALK_WAIT":
            v_lim = 0.0
            self.msg = "checking road for cars"
            if road_objects == 0:
                self.road_clear_t += dt
            else:
                self.road_clear_t = 0.0
            if self.road_clear_t > 1.5 and t - self.t_state > 1.0:
                self.set_state("XWALK_CROSS", t)
                self.crossed = True
        elif self.state == "XWALK_CROSS":
            self.msg = "crossing"
            if s > self.s_xout + 0.8:
                self.set_state("DRIVE", t)
        elif self.state == "APPROACH":
            v_lim = min(v_lim, 0.25 + 0.3 * max(0.0, self.s_del - s))
            if s > self.s_del - 1.3:
                # choose a stop point that leaves the hatch exactly on top
                a = st["hatch_ang"]
                Rr = self.ll.P.geo.R
                base = self.odo_s
                cands = [base + Rr * (-a + 2 * np.pi * k) for k in (-1, 0, 1, 2)]
                target_s_along = base + (self.s_del - s)
                self.s_final = min(cands, key=lambda c: abs(c - target_s_along) + (5 if c < base - 0.05 else 0))
                self.set_state("ALIGN", t)
        elif self.state == "ALIGN":
            rem = self.s_final - self.odo_s
            v_lim = float(np.clip(0.9 * rem, -0.3, 0.35))
            self.msg = f"hatch alignment {np.degrees(st['hatch_ang']):+.0f} deg"
            if abs(st["hatch_ang"]) < np.radians(4) and abs(st["v"]) < 0.04:
                self.set_state("PARK", t)
            elif t - self.t_state > 12:
                self.set_state("PARK", t)
        # PARK / OPEN / HANDOFF / CLOSE / DONE handled by the sim (actuators)
        if self.state in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE"):
            return 0.0, 0.0, dict(v_lim=0.0)

        # ------------------------------ corners: slow to the speed whose minimum
        # turn radius (set by the lean limit) fits the arc
        v_lim = min(v_lim, R.v_curve(s, self.ll.v_for_curvature))
        # ------------------------------ moving people: predict and yield / sidestep
        conflict = None
        if self.state in ("DRIVE", "XWALK_CROSS", "APPROACH"):
            self.track(t, est, bev)
            conflict = self.predict_conflict(est, st)
            if conflict is not None:
                tau, rel, lat, tr = conflict
                q = tr["p"] - est.p
                self.look_at = float(wrap(np.arctan2(q[1], q[0]) - est.psi))
                if abs(rel) > np.radians(140) and lat > -0.2 and not allow_road:
                    # oncoming along the sidewalk: keep right and ease off
                    self.sidestep_t, self.sidestep_o = 2.5, max(-0.85, min(o_pref, lat - 1.0))
                    v_lim = min(v_lim, 0.7)
                    self.msg = "keeping right for pedestrian"
                elif tau < 2.6:
                    v_lim = 0.0
                    self.msg = "yielding to pedestrian"
                else:
                    v_lim = min(v_lim, 0.5)
            else:
                self.look_at = None
        if self.sidestep_t > 0:
            self.sidestep_t -= dt
            o_pref = self.sidestep_o

        # ------------------------------ obstacle-aware lateral offset
        look = 4.5
        best, best_sc, best_free = self.o_cmd, -1e9, 0.0
        if self.state in ("DRIVE", "XWALK_CROSS", "APPROACH"):
            lim = 0.78 if not in_plaza else 1.6
            for o in np.arange(-lim, lim + 1e-6, 0.1):
                f = self.corridor_free(est, bev, o + 0.0, s, look, allow_road)
                sc = min(f, look) * 1.0 - 0.9 * abs(o - o_pref) - 0.4 * abs(o - self.o_cmd)
                if f < 1.6:
                    sc -= 5
                if sc > best_sc:
                    best, best_sc, best_free = o, sc, f
            self.free = best_free
            self.o_cmd += np.clip(best - self.o_cmd, -1.0 * dt, 1.0 * dt)
            # a ball can't sidestep: an S-curve shifting d metres needs about
            # 2*sqrt(R_min(v)*d) of run, so slow down until the shift fits
            d_lat = abs(best - e)
            if d_lat > 0.12:
                room = max(0.3, min(best_free, look) - 0.7)
                R_need = (room / 2.0) ** 2 / d_lat
                v_lim = min(v_lim, self.ll.v_for_curvature(1.0 / max(R_need, 1e-3), margin=1.0))
            if best_free < 1.25:
                v_lim = 0.0
                self.yield_t += dt
                if conflict is None:
                    self.msg = "yielding"
            else:
                self.yield_t = 0.0
                v_lim = min(v_lim, 0.45 * (best_free - 0.9) + 0.15)
                if self.state == "DRIVE" and conflict is None:
                    self.msg = "cruise" if abs(self.o_cmd - o_pref) < 0.15 else "passing obstacle"
        elif self.state == "ALIGN":
            self.o_cmd += np.clip(-0.3 - self.o_cmd, -0.3 * dt, 0.3 * dt)

        # ------------------------------ safety: straight-ahead corridor along the
        # robot's ACTUAL heading (the planned offset path can lag reality)
        if self.state in ("DRIVE", "XWALK_CROSS", "APPROACH"):
            d_e = self.straight_free(bev, allow_road)
            self.d_emerg = d_e
            v_lim = min(v_lim, max(0.0, 0.65 * (d_e - 0.6)))
            if abs(self.o_cmd - o_pref) > 0.2:
                v_lim = min(v_lim, 0.8)
            # not moving for a while:
            #  * something in the way that isn't leaving -> back up, swing toward the free side
            #  * nothing in the way (wheel-less ball stalled on a lip / heaved slab:
            #    static climb limit is ~8 mm) -> back up and take it with a run-up
            if abs(st["v"]) < 0.04 and self.state in ("DRIVE", "XWALK_CROSS") and self.backup_left <= 0 and self.charge_t <= 0:
                self.blocked_t += dt
            else:
                self.blocked_t = 0.0
            if self.blocked_t > (4.0 if v_lim < 0.05 else 1.5):
                self.blocked_t = 0.0
                if v_lim < 0.05:
                    self.backup_left, self.backup_kind = 1.4, "swing"
                else:
                    self.backup_left, self.backup_kind = 1.0, "charge"
                self.events.append((round(t, 2), "stalled", "BACKUP_" + self.backup_kind.upper()))
            if self.charge_t > 0:
                self.charge_t -= dt
                if d_e > 1.4:
                    v_lim = max(v_lim, 0.95)
                    self.msg = "run-up over sidewalk lip"
        if self.backup_left > 0:
            self.backup_left -= abs(st["v"]) * dt
            if self.backup_kind == "swing":
                self.msg = "backing up to go around"
                side = np.sign(self.o_cmd - self.e) or 1.0
                return -0.3, float(0.2 * side), dict(alpha=0.0)
            self.msg = "backing up for a run-up"
            if self.backup_left <= 0:
                self.charge_t = 3.0
            return -0.35, 0.0, dict(alpha=0.0)   # reverse while yawing toward the free side

        # ------------------------------ pure pursuit on the offset path
        v_ref = st["v"]
        Ld = float(np.clip(0.9 + 0.7 * abs(v_ref), 1.0, 2.0))
        target = R.point(min(s + Ld, R.total), self.o_cmd)
        q = target - est.p
        alpha = wrap(np.arctan2(q[1], q[0]) - est.psi)
        vv = max(abs(v_ref), 0.3)
        yaw_cmd = 2.0 * vv * np.sin(alpha) / Ld
        if self.state == "ALIGN":
            yaw_cmd = 0.5 * wrap(R.heading(s) - est.psi)
        # can't turn sharply: slow down when heading error is large
        if abs(alpha) > np.radians(25) and self.state != "ALIGN":
            v_lim = min(v_lim, 0.45)
        mx = self.ll.max_yaw_rate(max(abs(v_ref), 0.3))
        yaw_cmd = float(np.clip(yaw_cmd, -mx, mx))
        g = self.look_at if getattr(self, "look_at", None) is not None else alpha
        self.gaze = (float(np.clip(g, -0.8, 0.8)), 0.0)
        return float(v_lim), yaw_cmd, dict(target=target, alpha=alpha)
