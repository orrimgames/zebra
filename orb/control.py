"""State extraction ("what the onboard sensors would report") and low-level control.

Sign conventions (verified in sim):
  * forward = the level body's +x axis projected on the ground
  * drive ctrl < 0 rolls the band forward; we expose u_fwd = -ctrl
  * steer (battery swing) > 0 turns LEFT (yaw rate > 0) when rolling forward
"""
import numpy as np
import mujoco


class Robot:
    """Handles to one robot inside an MjModel."""

    def __init__(self, m, P):
        self.m, self.P = m, P
        self.bid_shell = m.body("shell").id
        self.bid_body = m.body("body").id
        self.bid_yoke = m.body("yoke").id
        self.bid_batt = m.body("battery").id
        self.bid_payload = m.body("payload").id
        self.bid_pod = m.body("pod").id
        j = lambda n: m.joint(n)
        self.qa = {n: m.jnt_qposadr[j(n).id] for n in ("drive", "pod_pitch", "steer", "lid")}
        self.va = {n: m.jnt_dofadr[j(n).id] for n in ("drive", "pod_pitch", "steer", "lid")}
        self.free_q = m.jnt_qposadr[j("root").id]
        self.free_v = m.jnt_dofadr[j("root").id]
        self.act = {n: m.actuator(n).id for n in ["pod_level", "drive", "steer", "lid"] + [f"leg_{k}" for k in range(1, 5)]}
        self.hatch_site = m.site("hatch_center").id
        self.imu_site = m.site("imu").id
        self.leg_q = [m.jnt_qposadr[j(f"leg_{k}").id] for k in range(1, 5)]
        self.eq_ids = {}

    # ---------------------------------------------------------------- truth state
    def state(self, d):
        R = self.P.geo.R
        pos = d.xpos[self.bid_shell].copy()
        Rb = d.xmat[self.bid_body].reshape(3, 3)
        fwd = Rb[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.array([-fwd[1], fwd[0], 0.0])
        heading = np.arctan2(fwd[1], fwd[0])
        vel = d.qvel[self.free_v:self.free_v + 3]
        v = float(vel @ fwd)
        v_lat = float(vel @ left)
        # axle direction in world (shell y)
        ax = d.xmat[self.bid_shell].reshape(3, 3)[:, 1]
        lean = float(np.arcsin(np.clip(ax[2], -1, 1)))         # >0: left side up
        lean_rate = float(d.cvel[self.bid_body][:3] @ fwd)
        # heading rate as an attitude filter (AHRS) reports it: the rate of the
        # heading angle itself (raw world-z angular velocity of the pod picks up
        # pitch/roll coupling on ramps and must not be integrated directly)
        if getattr(self, "_h_t", None) is not None and d.time < self._h_t:   # sim was reset
            self._h_t = None
        if getattr(self, "_h_t", None) is not None and d.time > self._h_t:
            raw = ((heading - self._h_prev + np.pi) % (2 * np.pi) - np.pi) / (d.time - self._h_t)
            self._yr = 0.8 * self._yr + 0.2 * raw
        elif getattr(self, "_h_t", None) is None:
            self._yr = 0.0
        if getattr(self, "_h_t", None) is None or d.time > self._h_t:
            self._h_prev, self._h_t = heading, d.time
        yaw_rate = float(self._yr)
        # pendulum (yoke) world pitch, + = mass forward
        zy = d.xmat[self.bid_yoke].reshape(3, 3)[:, 2]
        th_y = float(np.arctan2(-(zy @ fwd), zy[2]))
        th_y_rate = float(d.cvel[self.bid_yoke][:3] @ -left)    # rotation about -left (= +y_world when facing x)
        # pod tilt (fore-aft, lateral) w.r.t. gravity
        zb = Rb[:, 2]
        pod_pitch = float(np.arctan2(-(zb @ fwd), zb[2]))
        zp = d.xmat[self.bid_pod].reshape(3, 3)[:, 2]
        pod_roll = float(np.arctan2(zp @ left, zp[2]))       # the food pod itself (has its own roll gimbal)
        pod_rate = float(d.cvel[self.bid_body][:3] @ -left)
        # hatch position on the band: angle of hatch normal from straight up,
        # measured in the rolling direction (+ = hatch has passed the top going forward)
        hs = d.site_xpos[self.hatch_site] - pos
        hs /= np.linalg.norm(hs) + 1e-9
        hatch_ang = float(np.arctan2(hs @ fwd, hs[2]))
        return dict(pos=pos, heading=heading, fwd=fwd, left=left, v=v, v_lat=v_lat, lean=lean, lean_rate=lean_rate,
                    yaw_rate=yaw_rate, th_y=th_y, th_y_rate=th_y_rate, pod_pitch=pod_pitch,
                    pod_roll=pod_roll, pod_rate=pod_rate, hatch_ang=hatch_ang)


class LowLevel:
    """Cascaded speed controller (speed -> pendulum angle -> motor torque) and
    lean-steering heading-rate controller."""

    def __init__(self, P):
        self.P = P
        g = 9.81
        self.mgl = P.pendulum_moment() * g            # Nm per sin(angle)
        self.Kp_th, self.Kd_th = 60.0, 7.0
        self.Kv, self.Ki_v = 1.0, 0.4
        self.th_max = np.radians(45)
        self.i_v = 0.0
        self.i_r = 0.0
        self.v_cmd_f = 0.0
        self.acc_lim = 0.45           # m/s^2 (food-friendly)
        self.dec_lim = 0.9
        self.steer_f = 0.0

    def reset(self):
        self.i_v = self.i_r = self.v_cmd_f = self.steer_f = 0.0
        self.dist = 0.0; self.a_f = 0.0; self.a_cmd = 0.0; self.yc = 0.0

    M_EFF = 18.0     # effective translational mass seen by the pendulum torque (fit from step tests)

    def speed(self, s, v_des, dt):
        """speed -> pendulum angle -> motor torque, with
        * acceleration feed-forward (reference is rate-limited for the food)
        * a disturbance observer that learns slope + rolling resistance in
          ~0.3 s, so ramps don't cause 30 % overspeed."""
        R = self.P.geo.R
        # jerk-limited (S-curve) reference: gentle on the food and on the
        # pendulum, which has to swing back as acceleration ends
        if not hasattr(self, "a_cmd"):
            self.a_cmd = 0.0
        speeding_up = abs(v_des) > abs(self.v_cmd_f) and np.sign(v_des) == np.sign(self.v_cmd_f or v_des)
        lim = self.acc_lim if speeding_up else self.dec_lim
        a_des = np.clip(1.6 * (v_des - self.v_cmd_f), -lim, lim)
        self.a_cmd += np.clip(a_des - self.a_cmd, -1.2 * dt, 1.2 * dt)
        self.v_cmd_f += self.a_cmd * dt
        a_ref = self.a_cmd
        # measured acceleration (filtered derivative)
        if not hasattr(self, "v_prev"):
            self.v_prev, self.a_f, self.dist = s["v"], 0.0, 0.0
        a_raw = (s["v"] - self.v_prev) / dt; self.v_prev = s["v"]
        self.a_f += min(1.0, dt / 0.08) * (a_raw - self.a_f)
        # disturbance torque = what the pendulum is producing minus what accelerates the ball
        d_raw = self.mgl * np.sin(s["th_y"]) - self.M_EFF * R * self.a_f
        self.dist += min(1.0, dt / 0.3) * (d_raw - self.dist)
        e = self.v_cmd_f - s["v"]
        self.i_v = np.clip(self.i_v + e * dt, -1.0, 1.0)
        ff = (self.M_EFF * R * a_ref + self.dist) / self.mgl
        th_d = np.arcsin(np.clip(ff, -0.95, 0.95)) + self.Kv * e + self.Ki_v * self.i_v
        th_d = np.clip(th_d, -self.th_max, self.th_max)
        u = self.mgl * np.sin(th_d) + self.Kp_th * (th_d - s["th_y"]) - self.Kd_th * s["th_y_rate"]
        u = np.clip(u, -self.P.d.motor_peak_torque, self.P.d.motor_peak_torque)
        return -u, th_d        # ctrl for 'drive' actuator

    def level(self, s):
        """Pod leveling motor (pod <-> yoke). Gravity does the heavy lifting;
        this adds damping and trims out drag/acceleration tilt."""
        tau = 9.0 * s["pod_pitch"] + 1.6 * s["pod_rate"]   # +tau about axle lowers pod_pitch
        return float(np.clip(tau, -4, 4))

    STEER_GAIN = 0.49     # rad/s yaw per rad battery swing, steady state (scenarios/calibrate_steer.py)

    @staticmethod
    def lean_cap(v):
        """Allowed lean: generous when slow (roll/yaw coupling is weak), 7 deg at speed."""
        return float(np.interp(abs(v), [0.0, 0.5, 1.0], [0.22, 0.20, 0.12]))

    def lean_max(self, v):
        return min(self.lean_cap(v), self.P.geo.steer_max / (2.2 + 3.3 * v * v))

    def max_yaw_rate(self, v):
        """Yaw rate the lean limit allows at speed v (with 10 % margin)."""
        return 0.9 * abs(v) * np.tan(self.lean_max(v)) / self.P.geo.R

    def min_turn_radius(self, v):
        return self.P.geo.R / np.tan(self.lean_max(v))

    def v_for_curvature(self, kappa, margin=0.85):
        """Fastest speed whose minimum turn radius still fits curvature kappa."""
        if kappa < 1e-3:
            return 9.0
        rad = margin / kappa
        # R/tan(steer_max/(2.2+3.3 v^2)) <= rad  -> solve numerically
        lo, hi = 0.0, 3.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self.min_turn_radius(mid) <= rad:
                lo = mid
            else:
                hi = mid
        return max(0.22, lo)

    def steer(self, s, yaw_rate_des, dt):
        """Cascaded lean steering (tuned in scenarios/tune_roll2.py).

        Geometry: rolling with the axle tilted by `lean` makes the contact
        patch's spin friction yaw the ball at  r = v*tan(lean)/R.  So the
        desired yaw rate becomes a desired lean (capped at 7 deg), the battery
        swing that holds that lean is feed-forwarded (the ratio grows with
        speed because centripetal load pushes the ball upright), and the
        roll-pendulum mode is damped with lean-rate feedback.  Two rate limits
        keep the roll mode from being pumped: the yaw command (0.4 rad/s^2) and
        the battery swing (5 rad/s)."""
        R = self.P.geo.R
        if not hasattr(self, "yc"):
            self.yc = 0.0
        self.yc += float(np.clip(yaw_rate_des - self.yc, -0.4 * dt, 0.4 * dt))
        if abs(s["v"]) < 0.05:
            self.yc *= 0.99   # nothing to steer against at standstill
        v = s["v"]
        vv = max(abs(v), 0.3) * (1.0 if v >= -0.02 else -1.0)
        cap = self.lean_cap(v)
        lam_d = float(np.clip(np.arctan(self.yc * R / vv), -cap, cap))   # + = left side down
        cff = 2.2 + 3.3 * v * v
        target = cff * lam_d + 1.0 * s["lean_rate"]
        lim = self.P.geo.steer_max
        target = float(np.clip(target, -lim, lim))
        self.steer_f += float(np.clip(target - self.steer_f, -5.0 * dt, 5.0 * dt))
        return self.steer_f
