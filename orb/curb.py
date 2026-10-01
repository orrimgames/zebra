"""Leg-assisted curb / step climbing (up and down), heights 3-15 cm.

A pendulum ball alone can only roll over ~1.5 cm: lifting its own centre
over a curb edge needs ~3x the pendulum moment that fits inside. And legs
that only push straight down can't help much either: with free wheels the
edge has to carry all the horizontal load, which friction can't do while
the edge meets the band steeper than ~40 deg (a 12 cm curb meets it at 51).

So ORB turns into a small 4-wheeled cart for a few seconds (rear leg wheels
are driven by two tiny gearmotors):

UP    roll up to ~30 cm from the curb -> all legs lift the ball until its
      bottom is ~4.5 cm below the curb top -> drive forward on the leg wheels
      until the band touches the edge, now at a shallow ~30 deg -> front legs
      tuck up above the curb -> band + rear wheels roll it up while the rear
      legs follow the arc -> centre past the edge -> legs retract.
      (Low steps, h < 5 cm: no lift needed, rear legs push and the band rolls.)
DOWN  roll to the edge -> front legs reach down to the lower level, rear legs
      stay on the upper one, ball lifted 1.5 cm -> cart forward until the rear
      wheels are at the edge -> lower everything together -> band lands
      -> legs retract.

Inputs are what the robot has: curb height and edge distance from stereo +
ToF (here truth + noise), odometry along the approach axis, leg encoders.
Everything happens at <= 0.12 m/s.
"""
import numpy as np
import mujoco

FRONT, REAR = [0, 1], [2, 3]


class CliffSensor:
    """Two down-looking ToF 'cliff' sensors in the cap fronts (next to the
    forward ToFs), aimed 40 deg below horizontal: footprint ~0.6 m ahead of
    the centre. Returns (footprint ahead of centre along the heading,
    floor height there relative to the ball's own contact plane)."""
    ALPHA = np.radians(40.0)
    RANGE = 1.3

    def __init__(self, m, rb, noise=0.004, seed=0):
        self.m, self.rb = m, rb
        self.sites = [m.site(f"tof_{k}").id for k in (0, 1)]
        self.group = np.array([1, 0, 1, 0, 1, 1], np.uint8)
        self.gid = np.zeros(1, np.int32)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def read(self, d):
        Rb = d.xmat[self.rb.bid_body].reshape(3, 3)
        fwd = Rb[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        v = np.cos(self.ALPHA) * fwd - np.sin(self.ALPHA) * np.array([0, 0, 1.0])
        c = d.xpos[self.rb.bid_body]
        z_contact = d.xpos[self.rb.bid_shell][2] - self.rb.P.geo.R
        out = []
        for sid in self.sites:
            p = d.site_xpos[sid]
            dist = mujoco.mj_ray(self.m, d, p, v, self.group, 1, self.rb.bid_shell, self.gid)
            if dist < 0 or dist > self.RANGE or self.m.geom_bodyid[self.gid[0]] in self.rb.body_ids:
                dist = self.RANGE
            dist += self.rng.normal(0, self.noise)
            hit = p + dist * v
            out.append((float(np.dot(hit[:2] - c[:2], fwd[:2])), float(hit[2] - z_contact)))
        k = int(np.argmax([o[1] for o in out]))          # the sensor still seeing the higher floor
        return float(out[k][0]), float(out[k][1])         # (so a drop counts only once both see it)


class CurbClimber:
    TH_MAX = dict(kneel=25, settle=25, handoff=25, approach=40, lift=25, cart=25, climb=35, push=35, mount=30, legs=20, lower=25, land=22, backoff=25)

    def __init__(self, P, legs, h, x_edge, up=True, ll=None):
        self.P, self.legs, self.ll = P, legs, ll
        self.h, self.xe, self.up = float(h), float(x_edge), up
        g = P.geo
        self.R, self.ex = g.R, g.leg_exit_depth()
        self.state = None
        self.t_state = 0.0
        self.best_x, self.no_prog_t = -1e9, 0.0
        self.events = []
        self.result = None
        self.Z = None
        self.stall_t = 0.0
        self.low_step = up and h < 0.05
        self.set("approach", 0.0)

    # ------------------------------------------------------------ helpers
    def q_touch(self, Zc, Hg):
        """Leg extension putting the wheel on ground at height Hg when the centre is at Zc."""
        return Zc - Hg - self.ex + 0.002

    def set(self, st, t):
        if self.ll is not None:
            self.ll.th_max = np.radians(self.TH_MAX.get(st, 40))
            if st != self.state:
                self.ll.soft_reset(0.0)
            self.ll.on_legs = ("4legs" if st in ("lift", "cart", "legs") else
                               st not in ("approach", "done", "abort"))
        if st != self.state:
            if self.state is not None:
                self.events.append((round(t, 2), self.state, st))
            self.state, self.t_state = st, t

    MAX_UP, MAX_DOWN = 0.17, 0.155

    def step(self, t, s, x):
        """Returns a desired speed (float) or {'hold': th} / {'frame': 1}.
        Sets leg targets + wheel speed. self.result -> 'done' / 'abort' / 'too_high'."""
        if self.h > (self.MAX_UP if self.up else self.MAX_DOWN):
            self.result = "too_high"
            return 0.0
        if x > self.best_x + 0.01:
            self.best_x, self.no_prog_t = x, t
        if self.state not in ("approach", "legs", "lift", "backoff") and t - self.no_prog_t > 10.0:
            self.set("backoff", t)
        if self.state == "backoff":
            self.legs.set(0.0); self.legs.wheel_v = 0.0
            if max(s["legs"]) < 0.02:
                self.set("abort", t); self.result = "abort"
            return 0.0
        return self._up(t, s, x) if self.up else self._down(t, s, x)

    # ------------------------------------------------------------ UP
    def _up(self, t, s, x):
        R, h, xe, legs = self.R, self.h, self.xe, self.legs
        q = s["legs"]
        W = self.P.total_mass() * 9.81
        Z1 = R + max(0.0, h - 0.06)                      # centre height while on the legs
        dz = h - (Z1 - R)                                # edge height above the band bottom
        d_touch = np.sqrt(max(0.0, R * R - (R - dz) ** 2))
        if self.state == "approach":
            legs.set(0.0); legs.wheel_v = 0.0
            if self.low_step:
                x_stop = xe - d_touch + 0.01
                if x > x_stop - 0.01 or (abs(s["v"]) < 0.02 and x > x_stop - 0.06 and t - self.t_state > 2):
                    self.set("push", t)
                return float(np.clip(0.8 * (x_stop - x), 0.06, 0.2))
            x_stop = xe - 0.30
            # bumped into the curb earlier than the camera estimate said: trust the contact
            # (body progress, not band speed: on a wet curb face the band can spin in place)
            self.hist = getattr(self, "hist", []) + [(t, x)]
            while self.hist and self.hist[0][0] < t - 0.6:
                self.hist.pop(0)
            pushing = (abs(s["v"]) < 0.02 or x - self.hist[0][1] < 0.008) and t - self.t_state > 1.0
            self.stall_t = self.stall_t + 0.004 if pushing else 0.0
            if self.stall_t > 0.6:
                s_touch = np.sqrt(max(0.0, R * R - (R - h) ** 2))
                self.xe = xe = x + s_touch
                self.events.append((round(t, 2), "edge by contact", round(xe, 3)))
                self.stall_t = 0.0
                self.set("lift", t)
                return 0.0
            if x > x_stop - 0.01 and abs(s["v"]) < 0.04:
                self.set("lift", t)
            return float(np.clip(0.8 * (x_stop - x), 0.06, 0.2))
        if self.state == "lift":
            legs.set(self.q_touch(Z1, 0.0) + 0.010); legs.wheel_v = 0.0
            if min(q) > self.q_touch(Z1, 0.0) - 0.004:
                self.set("cart", t)
            return {"hold": 0.0}
        if self.state == "cart":
            legs.set(self.q_touch(Z1, 0.0) + 0.010)
            legs.wheel_v = 0.08
            # contact = the driven wheels stall against the band touching the edge
            stalled = s.get("wheel_v", legs.wheel_v) < 0.3 * legs.wheel_v or abs(s["v"]) < 0.01
            self.stall_t = self.stall_t + 0.004 if (stalled and t - self.t_state > 0.8) else 0.0
            if self.stall_t > 0.25:
                self.xe = xe = x + d_touch                     # contact: now we KNOW where the edge is
                self.events.append((round(t, 2), "edge by contact", round(xe, 3)))
            if x > xe - d_touch + 0.40 or t - self.t_state > 14.0:
                self.set("backoff", t)                       # never found the edge: don't guess
                return {"hold": 0.0}
            if self.stall_t > 0.25:
                self.set("handoff", t); self.q_hand = q.copy()
            return {"hold": 0.0}
        if self.state == "handoff":
            # unload the front legs onto the band/edge contact before tucking them
            ramp = max(0.0, 1.0 - (t - self.t_state) / 1.2)
            tgt = self.q_hand.copy()
            tgt[FRONT] = self.q_hand[FRONT] - (1 - ramp) * 0.03
            tgt[REAR] = self.q_touch(Z1, 0.0) + 0.012
            legs.set(tgt); legs.wheel_v = 0.02
            if ramp <= 0.0:
                self.set("climb", t)
            return {"frame": 1}
        if self.state in ("climb", "push"):
            d = xe - x
            Zm = float(np.mean(q[REAR])) + self.ex - 0.002     # centre height measured by the rear legs
            Zc = h + np.sqrt(max(0.0, R * R - max(0.0, d - 0.02) ** 2)) if d > 0 else h + R
            Zc = max(Zc, Z1)
            qf = max(0.0, self.q_touch(min(Zc, h + R), h + 0.03))   # front wheels kept 3 cm over the curb top
            tgt = np.zeros(4)
            tgt[FRONT] = qf
            if self.state == "push":
                F = np.full(4, np.nan); F[REAR] = 0.30 * W
                qm = np.full(4, self.P.geo.leg_stroke); qm[FRONT] = qf
                legs.set_force(F, q_max=qm)
            else:
                # unload the springs near the crest. The crest is judged from the rear
                # legs' own length (screw encoder + spring deflection = centre height),
                # not from the edge estimate, so a few cm of edge error can't fire a catapult.
                gap = h + R - Zm
                pre = 0.010 + 0.030 * float(np.clip(min((d - 0.06) / 0.10, (gap - 0.003) / 0.03), 0.0, 1.0))
                tgt[REAR] = self.q_touch(Zc, 0.0) + pre
                legs.set(tgt)
            near = x > xe - 0.04 or Zm > h + R - 0.012   # cresting: ease off so it doesn't surge over
            v = 0.06 if near else 0.09
            legs.wheel_v = 0.75 * v                      # the band leads, the wheels assist (no fighting)
            if near and self.ll is not None:
                self.ll.th_max = np.radians(20)
            # running away over the crest: band AND leg-wheel odometry both fast for 60 ms
            fast = min(s["v"], s.get("wheel_v", 0.0)) > 0.15
            self.surge_t = getattr(self, "surge_t", 0.0) + 0.004 if fast else 0.0
            surge = self.surge_t > 0.04
            if surge:
                self.surged = True
            if x > xe + 0.03 or surge:                    # centre past the edge (or rolling away): speed loop takes over
                self.events.append((round(t, 2), "crest", dict(x=round(x - xe, 3), Zm=round(Zm - h - R, 3), wheel_v=round(s.get("wheel_v", 0.0), 2))))
                self.set("mount", t)
            if self.state == "climb":
                # gentle constant band torque (pendulum held slightly forward): the legs
                # and wheels set the pace; a speed loop here would brake against the legs
                # traction control: if the band spins on the edge (wet), back the torque off,
                # otherwise it bites at the crest with stored speed and launches the ball
                slip = float(np.clip((0.25 - s["v"]) / 0.12, 0.0, 1.0))
                return {"hold": np.radians((6.0 if near else 14.0) * slip)}
            return v
        if self.state == "mount":
            legs.set(0.0); legs.wheel_v = 0.0
            if getattr(self, "surged", False):          # came over too fast: stop first, then carry on
                if abs(s["v"]) < 0.03:
                    self.surged = False
                    self.xe = xe = min(xe, x - 0.03)          # we're on top: the edge is behind us
                return 0.0
            if max(q) < 0.02 and x > xe + 0.3:
                self.set("done", t); self.result = "done"
            return 0.06
        return 0.0

    # ------------------------------------------------------------ DOWN
    def _down(self, t, s, x):
        R, h, xe, legs = self.R, self.h, self.xe, self.legs
        q = s["legs"]
        W = self.P.total_mass() * 9.81
        Ztop = h + R + 0.015
        if self.state == "approach":
            legs.set(0.0); legs.wheel_v = 0.0
            # the cliff ToFs see the drop ~0.6 m ahead: that, not the camera, places the edge
            c = s.get("cliff")
            if c is not None and not getattr(self, "edge_seen", False):
                fx, hz = c
                if hz < -0.5 * h:
                    self.drop_n = getattr(self, "drop_n", 0) + 1
                else:
                    self.drop_n = 0; self.last_flat = x + fx
                if self.drop_n >= 3 and getattr(self, "last_flat", None) is not None and abs(self.last_flat - xe) < 0.30:
                    self.xe = xe = self.last_flat + 0.004
                    self.edge_seen = True
                    self.events.append((round(t, 2), "edge by cliff ToF", round(xe, 3)))
            x_stop = xe - 0.04
            if x > x_stop - 0.01 and abs(s["v"]) < 0.04:
                self.set("legs", t)
            return float(np.clip(0.8 * (x_stop - x), 0.0, 0.2))
        if self.state == "legs":
            # front legs first (down to the lower level), then the rear ones;
            # the speed loop holds the band still so the push can't roll it off
            tgt = np.zeros(4)
            tgt[FRONT] = self.q_touch(Ztop, 0.0) + 0.008
            front_ok = min(q[FRONT]) > tgt[0] - 0.012
            tgt[REAR] = self.q_touch(Ztop, h) + 0.008 if front_ok else 0.0
            legs.set(tgt); legs.wheel_v = 0.0
            if front_ok and min(q[REAR]) > tgt[2] - 0.012:
                self.set("cart", t)
            return 0.0
        if self.state == "cart":
            legs.wheel_v = float(np.clip(1.0 * (xe + 0.09 - x), 0.025, 0.10))
            if x > xe + 0.085:                           # rear wheels stay ~3.5 cm short of the edge
                self.set("kneel", t); self.Z = Ztop
            return {"hold": 0.0}
        if self.state == "kneel":
            # all four legs come down together (rear wheels stay on the upper
            # level) until the band rests on the edge corner
            legs.wheel_v = 0.0
            d = max(0.0, x - xe)
            Zc = h + np.sqrt(max(0.0, R * R - d * d))
            self.Z = max(Zc - 0.03, self.Z - 0.02 * 0.004)
            tgt = np.zeros(4)
            # pitch feedback on the leg split: the height estimate is never perfect
            self.lvl = getattr(self, "lvl", 0.0) + 0.004 * 1.5 * np.tan(s["pod_pitch"])
            self.lvl = float(np.clip(self.lvl, -0.03, 0.03))
            tgt[FRONT] = self.q_touch(self.Z, 0.0) + 0.006 - self.lvl
            tgt[REAR] = max(0.0, self.q_touch(self.Z, h) + 0.006 + self.lvl)
            legs.set(tgt)
            Ftot = float(np.sum(s.get("legF", np.full(4, W / 4))))
            unloaded = Ftot < 0.62 * W and self.Z < Ztop - 0.006
            self.stall_t = self.stall_t + 0.004 if unloaded else 0.0
            if self.stall_t > 0.12 or self.Z <= Zc - 0.03:
                self.set("settle", t); self.q_f0 = tgt[FRONT].copy(); self.q_r0 = float(tgt[REAR][0])
            return {"hold": 0.0}
        if self.state == "settle":
            # rear legs up: now on a stable triangle (two front wheels + band on the corner)
            legs.wheel_v = 0.0
            tgt = np.zeros(4); tgt[FRONT] = self.q_f0
            tau = t - self.t_state
            tgt[REAR] = max(0.0, self.q_r0 - (0.02 * tau if tau < 1.0 else 0.02 + 0.12 * (tau - 1.0)))   # unload slowly, then tuck
            legs.set(tgt)
            if max(q[REAR]) < 0.01:
                self.set("lower", t)
            return {"frame": 1}
        if self.state == "lower":
            # front legs shorten slowly; the band rolls forward around the corner,
            # and the front wheels pace that roll so the band never leaves the corner
            dzr = 0.018 if self.Z > R + 0.035 else 0.008      # touch down gently
            self.Z = max(R - 0.02, self.Z - dzr * 0.004)
            zz = max(1e-3, self.Z - h)
            legs.wheel_v = float(np.clip(dzr * zz / np.sqrt(max(1e-4, R * R - zz * zz)), 0.0, 0.08)) if self.Z > R else 0.0
            tgt = np.zeros(4)
            tgt[FRONT] = max(0.0, self.q_touch(self.Z, 0.0) + 0.004)
            legs.set(tgt)
            Ftot = float(np.sum(s.get("legF", np.zeros(4))))
            if self.Z <= R - 0.015 or (self.Z < R + 0.01 and Ftot < 0.15 * W):
                self.set("land", t)
            return {"frame": 1}
        if self.state == "land":
            legs.set(0.0); legs.wheel_v = 0.0
            if max(q) < 0.02:
                self.set("done", t); self.result = "done"
            return 0.0
        return 0.0
