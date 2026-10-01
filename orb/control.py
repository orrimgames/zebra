"""State extraction, the robot's sensor model, and low-level control (v2).

Sign conventions (verified in sim):
  * forward = the level body's +x axis projected on the ground
  * drive ctrl < 0 rolls the band forward; we expose u_fwd = -ctrl
  * steer (tray swing) > 0 turns LEFT (yaw rate > 0) when rolling forward

Sim-to-real: the controllers below never read simulator truth directly.
They read `SensorModel.measure()`, which adds the latency, noise, bias and
scale errors of the real sensors (IMU attitude filter, gyro, band encoder),
and they run at the robot's 250 Hz control rate, not the physics rate.
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
        self.bid_lid = m.body("lid").id
        j = lambda n: m.joint(n)
        self.qa = {n: m.jnt_qposadr[j(n).id] for n in ("drive", "pod_pitch", "steer", "lid", "pod_iso")}
        self.va = {n: m.jnt_dofadr[j(n).id] for n in ("drive", "pod_pitch", "steer", "lid", "pod_iso")}
        self.free_q = m.jnt_qposadr[j("root").id]
        self.free_v = m.jnt_dofadr[j("root").id]
        self.act = {n: m.actuator(n).id for n in ["pod_level", "drive", "steer", "lid"] + [f"leg_{k}" for k in range(1, 5)] + [f"wheel_{k}" for k in range(1, 5)]}
        self.hatch_site = m.site("hatch_center").id
        self.imu_site = m.site("imu").id
        self.leg_q = [m.jnt_qposadr[j(f"leg_{k}").id] for k in range(1, 5)]
        self.leg_v = [m.jnt_dofadr[j(f"leg_{k}").id] for k in range(1, 5)]
        self.wheel_geoms = [m.geom(f"foot_{k}").id for k in range(1, 5)]
        self.leg_xy = [(np.sign(m.body_pos[m.body(f"leg_{k}").id][0]), np.sign(m.body_pos[m.body(f"leg_{k}").id][1])) for k in range(1, 5)]
        self.shell_geom = m.geom("shell_contact").id
        self.body_ids = set(range(self.bid_shell, self.bid_shell + 17))
        self._h_t = None

    # ---------------------------------------------------------------- truth state
    def state(self, d):
        pos = d.xpos[self.bid_shell].copy()
        Rb = d.xmat[self.bid_body].reshape(3, 3)
        fwd = Rb[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.array([-fwd[1], fwd[0], 0.0])
        heading = np.arctan2(fwd[1], fwd[0])
        vel = d.qvel[self.free_v:self.free_v + 3]
        v = float(vel @ fwd)
        v_lat = float(vel @ left)
        ax = d.xmat[self.bid_shell].reshape(3, 3)[:, 1]
        lean = float(np.arcsin(np.clip(ax[2], -1, 1)))         # >0: left side up
        lean_rate = float(d.cvel[self.bid_body][:3] @ fwd)
        # heading rate as an attitude filter (AHRS) reports it
        if self._h_t is not None and d.time < self._h_t:
            self._h_t = None
        if self._h_t is not None and d.time > self._h_t:
            raw = ((heading - self._h_prev + np.pi) % (2 * np.pi) - np.pi) / (d.time - self._h_t)
            self._yr = 0.8 * self._yr + 0.2 * raw
        elif self._h_t is None:
            self._yr = 0.0
        if self._h_t is None or d.time > self._h_t:
            self._h_prev, self._h_t = heading, d.time
        zy = d.xmat[self.bid_yoke].reshape(3, 3)[:, 2]
        th_y = float(np.arctan2(-(zy @ fwd), zy[2]))
        th_y_rate = float(d.cvel[self.bid_yoke][:3] @ -left)
        zb = Rb[:, 2]
        pod_pitch = float(np.arctan2(-(zb @ fwd), zb[2]))
        zp = d.xmat[self.bid_pod].reshape(3, 3)[:, 2]
        pod_roll = float(np.arctan2(zp @ left, zp[2]))
        pod_rate = float(d.cvel[self.bid_body][:3] @ -left)
        hs = d.site_xpos[self.hatch_site] - pos
        hs /= np.linalg.norm(hs) + 1e-9
        hatch_ang = float(np.arctan2(hs @ fwd, hs[2]))
        legs = np.array([d.qpos[q] for q in self.leg_q])
        return dict(t=d.time, pos=pos, heading=heading, fwd=fwd, left=left, v=v, v_lat=v_lat, lean=lean, lean_rate=lean_rate,
                    yaw_rate=float(self._yr), th_y=th_y, th_y_rate=th_y_rate, pod_pitch=pod_pitch,
                    pod_roll=pod_roll, pod_rate=pod_rate, hatch_ang=hatch_ang, legs=legs,
                    steer=float(d.qpos[self.qa["steer"]]))

    def wheel_loads(self, d):
        """Normal force on each leg wheel and on the shell (N)."""
        f = np.zeros(6)
        out = np.zeros(4); shell = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g = (c.geom1, c.geom2)
            for k, wg in enumerate(self.wheel_geoms):
                if wg in g:
                    mujoco.mj_contactForce(self.m, d, i, f); out[k] += abs(f[0])
            if self.shell_geom in g:
                mujoco.mj_contactForce(self.m, d, i, f); shell += abs(f[0])
        return out, shell


class SensorModel:
    """What the robot's own sensors report: delayed, noisy, biased truth.

    Defaults are datasheet-class numbers for a BMI088-class IMU with a
    complementary attitude filter, a 14-bit magnetic encoder on the band
    drive, and a TPU tread whose rolling radius drifts with wear/pressure."""

    def __init__(self, P, seed=0, scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.delay_n = max(1, int(round(P.sensor_delay / P.control_dt)))
        self.buf = []
        s = scale
        self.ang_noise = np.radians(0.15) * s        # attitude filter noise
        self.ang_bias = self.rng.normal(0, np.radians(0.4) * s, 2)   # pitch/roll mount + filter bias
        self.rate_noise = 0.01 * s                  # rad/s
        self.v_scale = 1.0 + self.rng.normal(0, 0.012 * s)            # tread radius error
        self.v_noise = 0.01 * s

    def measure(self, st):
        n = self.rng.normal
        m = dict(st)
        m["th_y"] = st["th_y"] + self.ang_bias[0] + n(0, self.ang_noise)
        m["pod_pitch"] = st["pod_pitch"] + self.ang_bias[0] + n(0, self.ang_noise)
        m["lean"] = st["lean"] + self.ang_bias[1] + n(0, self.ang_noise)
        m["th_y_rate"] = st["th_y_rate"] + n(0, self.rate_noise)
        m["pod_rate"] = st["pod_rate"] + n(0, self.rate_noise)
        m["lean_rate"] = st["lean_rate"] + n(0, self.rate_noise)
        m["yaw_rate"] = st["yaw_rate"] + n(0, self.rate_noise)
        m["v"] = st["v"] * self.v_scale + n(0, self.v_noise)
        self.buf.append(m)
        if len(self.buf) > self.delay_n:
            return self.buf.pop(0)
        return self.buf[0]


class LowLevel:
    """Cascaded speed controller (speed -> pendulum angle -> motor torque),
    lean-steering heading-rate controller with integral wind rejection,
    pod leveling."""

    M_EFF = 18.5         # effective translational mass seen by the pendulum torque (fit: scenarios/fit_drive.py)
    # steady-state tray swing per unit lean, cff(v) = A + B v^2 (fit: scenarios/calibrate_steer.py)
    CFF_A, CFF_B = 1.68, 1.16

    def __init__(self, P):
        self.P = P
        g = 9.81
        self.mgl = P.pendulum_moment() * g            # Nm per sin(angle)
        self.Kp_th, self.Kd_th = 95.0, 18.0
        self.Kv, self.Ki_v = 1.0, 0.4
        self.th_max = np.radians(60)
        self.acc_lim = 0.6            # m/s^2 normal (food-friendly)
        self.dec_lim = 1.0
        self.dec_emerg = 2.6          # emergency stop
        self.jerk = 1.6
        self.jerk_emerg = 12.0
        self.steer_Kp, self.steer_Ki, self.steer_Kd = 0.4, 4.0, 1.0
        self.yaw_slew = 1.0
        self.lvl_kp, self.lvl_kd = 40.0, 5.0
        self.emergency = False
        self.frame_vref = 0.0
        self.pend_fade = 15.0         # deg; steering fades while the pendulum is swung back to brake
        self.on_legs = False          # legs carrying load: steering centred, caps held stiffly level
        self.reset()

    def reset(self):
        self.i_v = 0.0
        self.v_cmd_f = 0.0
        self.steer_f = 0.0
        self.dist = 0.0; self.a_f = 0.0; self.a_cmd = 0.0; self.yc = 0.0
        self.v_prev = None
        self.i_l = 0.0

    # ------------------------------------------------------------------ speed
    def speed(self, s, v_des, dt):
        """speed -> pendulum angle -> motor torque, with a jerk-limited
        reference, acceleration feed-forward and a disturbance observer that
        learns slope + rolling resistance + headwind in ~0.3 s."""
        R = self.P.geo.R
        emerg = self.emergency
        speeding_up = abs(v_des) > abs(self.v_cmd_f) and np.sign(v_des) == np.sign(self.v_cmd_f or v_des)
        lim = self.acc_lim if speeding_up else (self.dec_emerg if emerg else self.dec_lim)
        jerk = self.jerk_emerg if emerg else self.jerk
        a_des = np.clip(2.0 * (v_des - self.v_cmd_f), -lim, lim)
        self.a_cmd += np.clip(a_des - self.a_cmd, -jerk * dt, jerk * dt)
        self.v_cmd_f += self.a_cmd * dt
        if emerg and abs(v_des) < 1e-3 and abs(self.v_cmd_f) < 0.05:
            self.v_cmd_f = 0.0
        a_ref = self.a_cmd
        if self.v_prev is None:
            self.v_prev = s["v"]
        a_raw = (s["v"] - self.v_prev) / dt; self.v_prev = s["v"]
        self.a_f += min(1.0, dt / 0.08) * (a_raw - self.a_f)
        d_raw = self.mgl * np.sin(s["th_y"]) - self.M_EFF * R * self.a_f
        self.dist += min(1.0, dt / 0.3) * (d_raw - self.dist)
        e = self.v_cmd_f - s["v"]
        self.i_v = np.clip(self.i_v + e * dt, -1.0, 1.0)
        ff = (self.M_EFF * R * a_ref + self.dist) / self.mgl
        th_d = np.arcsin(np.clip(ff, -0.95, 0.95)) + self.Kv * e + self.Ki_v * self.i_v
        th_d = np.clip(th_d, -self.th_max, self.th_max)
        u = self.mgl * np.sin(th_d) + self.Kp_th * (th_d - s["th_y"]) - self.Kd_th * s["th_y_rate"]
        u = np.clip(u, -self.P.d.motor_peak_torque, self.P.d.motor_peak_torque)
        self.th_d = th_d
        return -u, th_d        # ctrl for 'drive' actuator

    def soft_reset(self, v_now=0.0):
        """Forget learned disturbances (slope/push) at a mode change."""
        self.dist = self.mgl * np.sin(np.radians(2.8))     # flat-ground rolling resistance
        self.i_v = 0.0; self.a_cmd = 0.0; self.v_cmd_f = v_now

    def hold_frame(self, s):
        """Leg modes: the leveling worm gear is locked, so caps + pendulum are
        one rigid frame and the DRIVE motor keeps that frame level, reacting
        the legs' pitching moment into the band's contact (ground or curb edge)."""
        # + a speed term: with the band on a corner or on flat ground, a pure pitch loop
        #   would happily roll the ball away; this keeps it crawling at <= v_ref
        u = self.mgl * np.sin(s["th_y"]) + 320.0 * (0.0 - s["pod_pitch"]) - 28.0 * s["pod_rate"] + 60.0 * (self.frame_vref - s["v"])
        self.v_cmd_f = s["v"]; self.a_cmd = 0.0; self.i_v = 0.0; self.v_prev = s["v"]
        return -float(np.clip(u, -self.P.d.motor_peak_torque, self.P.d.motor_peak_torque))

    def hold(self, s, th_ref=0.0):
        """Pendulum position hold (used while the ball rides on its legs)."""
        u = self.mgl * np.sin(th_ref) + self.Kp_th * (th_ref - s["th_y"]) - self.Kd_th * s["th_y_rate"]
        self.v_cmd_f = s["v"]; self.a_cmd = 0.0; self.i_v = 0.0; self.v_prev = s["v"]
        return -float(np.clip(u, -self.P.d.motor_peak_torque, self.P.d.motor_peak_torque))

    def level(self, s):
        if self.on_legs == "4legs":            # four wheels on the ground already fix the caps' pitch...
            e = s["pod_pitch"]
            db = 0.05                            # ...unless a wheel lifts: then catch it before it tips
            if abs(e) < db:
                return 0.0
            tau = 220.0 * (e - np.sign(e) * db) + 18.0 * s["pod_rate"]
            return float(np.clip(tau, -self.P.d.level_torque, self.P.d.level_torque))
        kp, kd = (220.0, 18.0) if self.on_legs else (self.lvl_kp, self.lvl_kd)
        tau = kp * s["pod_pitch"] + kd * s["pod_rate"]
        return float(np.clip(tau, -self.P.d.level_torque, self.P.d.level_torque))

    # ------------------------------------------------------------------ steering
    def cff(self, v):
        return self.CFF_A + self.CFF_B * v * v

    @staticmethod
    def lean_cap(v):
        """Allowed lean: generous when slow, ~7 deg at speed (roll-mode margin)."""
        return float(np.interp(abs(v), [0.0, 0.5, 1.0, 2.0], [0.24, 0.22, 0.15, 0.12]))

    def lean_max(self, v):
        usable = 0.75 * self.P.geo.steer_max           # 25 % of the swing kept for gusts
        return min(self.lean_cap(v), usable / self.cff(v))

    def max_yaw_rate(self, v):
        return 0.9 * abs(v) * np.tan(self.lean_max(v)) / self.P.geo.R

    def min_turn_radius(self, v):
        """Radius the planner may ask for at speed v (matches the stress test: tray
        swing for gusts kept in reserve, measured lag of the lean loop included)."""
        return (self.P.geo.R / np.tan(self.lean_max(v))) / 0.9 * (1.0 + 0.08 * abs(v))

    def v_for_curvature(self, kappa, margin=0.85):
        if kappa < 1e-3:
            return 9.0
        rad = margin / kappa
        lo, hi = 0.0, 3.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self.min_turn_radius(mid) <= rad:
                lo = mid
            else:
                hi = mid
        return max(0.22, lo)

    def steer(self, s, yaw_rate_des, dt):
        """Cascaded lean steering.

        A leaning ball turns at r = v*tan(lean)/R, so the yaw-rate command
        becomes a lean command; the tray swing that holds that lean is
        feed-forwarded (cff grows with v^2: gyroscopic + centripetal load
        push the ball upright), lean error gets PI feedback (the integral is
        what cancels a steady crosswind or a cross-slope), and the roll
        pendulum mode is damped with lean-rate feedback."""
        R = self.P.geo.R
        if self.on_legs:
            self.i_l = 0.0; self.yc = 0.0
            self.steer_f += float(np.clip(0.0 - self.steer_f, -self.P.d.steer_rate * dt, self.P.d.steer_rate * dt))
            return self.steer_f
        self.yc += float(np.clip(yaw_rate_des - self.yc, -self.yaw_slew * dt, self.yaw_slew * dt))
        v = s["v"]
        if abs(v) < 0.05:
            self.yc *= 0.99
        vv = max(abs(v), 0.3) * (1.0 if v >= -0.02 else -1.0)
        cap = self.lean_cap(v)
        lam_d = float(np.clip(np.arctan(self.yc * R / vv), -cap, cap))   # + = left side down
        lam = -s["lean"]
        e_l = lam_d - lam
        lim = self.P.geo.steer_max
        # gain schedule: lean feedback fades out at walking pace and below, where
        # lean barely turns the ball and the roll mode is least damped
        sched = float(np.clip((abs(v) - 0.2) / 0.6, 0.0, 1.0))
        # pendulum swung far (hard braking / steep hill): the tray's swing axis is
        # tilted and swinging it mostly yaws the ball, so steer gently and hold the integral
        braking = s["th_y"] * v < 0                      # pendulum swung back against the motion
        pend = float(np.clip(1.0 - (abs(s["th_y"]) - np.radians(self.pend_fade)) / np.radians(15), 0.0, 1.0)) if braking else 1.0
        sched *= pend
        if abs(v) > 0.15:
            self.i_l = float(np.clip(self.i_l + sched * self.steer_Ki * e_l * dt, -0.7 * lim, 0.7 * lim))
        self.i_l *= (1.0 - (1.0 - sched) * 0.5 * dt)          # leak when slow
        target = pend * (self.cff(v) * lam_d + self.i_l + self.steer_Kd * s["lean_rate"]) + sched * self.steer_Kp * e_l
        target = float(np.clip(target, -lim, lim))
        self.steer_f += float(np.clip(target - self.steer_f, -self.P.d.steer_rate * dt, self.P.d.steer_rate * dt))
        return self.steer_f

    def crosswind_estimate(self):
        """Side force (N, + pushes the ball to its left) implied by the tray
        offset the lean integral needed: the robot's own anemometer."""
        P = self.P
        return -P.m.battery * 9.81 * P.geo.tray_r * np.sin(self.i_l) / P.geo.R


class Legs:
    """Series-elastic lead-screw legs. Position mode: rate-limited screw
    target. Force mode: screw = measured leg position + F/k (the spring
    deflection IS the force sensor), still limited by the screw speed."""

    def __init__(self, P, rb):
        self.P, self.rb = P, rb
        self.target = np.zeros(4)
        self.cmd = np.zeros(4)
        self.force = None
        self.wheel_v = 0.0          # rear wheel surface speed command (m/s)

    def set(self, q):
        self.force = None
        self.target = np.clip(np.broadcast_to(np.asarray(q, float), (4,)).copy(), 0.0, self.P.geo.leg_stroke)

    def set_force(self, F, q_max=None):
        """F: per-leg force (N) or None for position-held legs given by q_max."""
        self.force = np.broadcast_to(np.asarray(F, float), (4,)).copy()
        self.q_max = np.full(4, self.P.geo.leg_stroke) if q_max is None else np.broadcast_to(np.asarray(q_max, float), (4,)).copy()

    def step(self, d, dt):
        r = self.P.geo.leg_speed * dt
        if self.force is not None:
            q = np.array([d.qpos[a] for a in self.rb.leg_q])
            tgt = np.where(np.isnan(self.force), self.q_max, q + np.nan_to_num(self.force) / self.P.d.leg_k)
            self.target = np.clip(np.minimum(tgt, self.q_max), 0.0, self.P.geo.leg_stroke + 0.01)
        self.cmd += np.clip(self.target - self.cmd, -r, r)
        for k in range(4):
            d.ctrl[self.rb.act[f"leg_{k+1}"]] = self.cmd[k]
        tgt = np.broadcast_to(np.asarray(self.wheel_v, float), (4,))
        if not hasattr(self, "wv"):
            self.wv = np.zeros(4)
        self.wv += np.clip(tgt - self.wv, -0.25 * dt, 0.25 * dt)          # gentle wheel ramps
        w = self.wv / self.P.geo.leg_wheel_r
        for k in range(4):
            d.ctrl[self.rb.act[f"wheel_{k+1}"]] = w[k]

    def park(self, d, W, t_since, hold=0.20):
        """Deploy the legs on a ball that is already stopped and braked.

        Each leg runs out in position mode (fast, then slow from 6 cm), and
        stops the moment its series spring shows a contact load above what the
        leg's own friction gives in free air. Touch-down loads are tiny, so
        the braked ball (a weeble: CG below centre, sitting in equilibrium) is
        not tipped by uneven feet on a slope. When all four are down, all
        four springs are preloaded by the same amount (hold*W each), so the
        stance carries most of the weight. Returns True when done."""
        dt = 0.004
        if t_since < 0.01 or not hasattr(self, "pk_down"):
            self.pk_down = np.zeros(4, bool); self.pk_t = np.zeros(4); self.pk_done = False
            self.pk_tgt = np.array([d.qpos[a] for a in self.rb.leg_q])
            self.pk_pre = 0.0
        if self.pk_done:
            return True
        q = np.array([d.qpos[a] for a in self.rb.leg_q])
        F = self.forces(d)
        v_find = 0.05
        F_free = (350.0 + 60.0) * v_find + 14.0             # leg drag in free air + margin
        for k in range(4):
            if self.pk_down[k]:
                continue
            fast = q[k] < 0.06
            self.pk_tgt[k] = min(self.P.geo.leg_stroke, self.cmd[k] + (self.P.geo.leg_speed if fast else v_find) * dt * 1.02)
            contact = q[k] > 0.075 and F[k] > F_free             # (past the fast-to-slow lag transient)
            self.pk_t[k] = self.pk_t[k] + dt if contact else 0.0
            if self.pk_t[k] > 0.02 or self.pk_tgt[k] >= self.P.geo.leg_stroke:
                self.pk_down[k] = True
                self.pk_tgt[k] = q[k] + 15.0 / self.P.d.leg_k    # keep a light touch load
        if self.pk_down.all():
            # equal preload ramp on all four
            self.pk_pre = min(hold * W / self.P.d.leg_k, self.pk_pre + 0.02 * dt)
            tgt = self.pk_tgt + self.pk_pre
            if self.pk_pre >= hold * W / self.P.d.leg_k - 1e-9:
                self.pk_done = True
        else:
            tgt = self.pk_tgt
        self.force = None
        self.target = np.clip(tgt, 0.0, self.P.geo.leg_stroke)
        if self.pk_done:
            self.set(self.target)
        return self.pk_done

    def lock_wheels(self, m, on):
        """Worm-gear wheel motors: parked = self-locking (stiff, ~3 Nm holding)."""
        for k in range(1, 5):
            a = self.rb.act[f"wheel_{k}"]
            kv = 60.0 if on else 3.0
            m.actuator_gainprm[a][0] = kv; m.actuator_biasprm[a][2] = -kv
            hold = 3.0 if on else self.P.d.wheel_torque
            m.actuator_forcerange[a] = [-hold, hold]
        if on:
            self.wheel_v = 0.0

    def forces(self, d):
        """Per-leg force from the series-spring deflection (what the robot measures)."""
        q = np.array([d.qpos[a] for a in self.rb.leg_q])
        return np.maximum(0.0, self.P.d.leg_k * (self.cmd - q))

    def ground_q(self, s=None):
        """Leg extension at which the wheel just touches flat ground."""
        g = self.P.geo
        return g.R - g.leg_exit_depth() + 0.002
