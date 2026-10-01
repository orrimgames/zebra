"""Closed-loop campus delivery simulation + video (v2).

    python run_delivery.py            # renders out/delivery.mp4 + metrics

The robot runs as it would for real: 250 Hz motor control on delayed/noisy
sensors, 10 Hz camera perception + planning, an independent safety MCU on
its own ToF sensors, radar for crossing roads, and wind that blows on it.
"""
import json, time, os
import numpy as np
import mujoco
import cv2

from .params import Params
from . import model, world
from .control import Robot, LowLevel, SensorModel, Legs
from .perception import Perception
from .planner import Planner, Estimator, SensorSim, wrap
from .curb import CurbClimber, CliffSensor
from .wind import Wind, apply as apply_wind
from .safety import SafetyMCU
from . import meshes

CYAN, AMBER, GREEN, LILAC, RED = (0.35, 0.95, 1.0), (1.0, 0.72, 0.2), (0.45, 1.0, 0.55), (0.75, 0.62, 1.0), (1.0, 0.25, 0.2)
STATE_COLORS = {
    "DRIVE": CYAN, "XWALK_CROSS": CYAN, "APPROACH": CYAN, "YIELD": AMBER, "XWALK_WAIT": AMBER, "ALIGN": (0.6, 0.8, 1.0),
    "PARK": GREEN, "OPEN": GREEN, "HANDOFF": GREEN, "CLOSE": GREEN, "DONE": GREEN,
    "CURB_ALIGN": LILAC, "CURB": LILAC, "CURB_EXIT": LILAC, "BRACE": AMBER, "EMERG": RED,
}


class Eyes:
    """Animated LED eyes (drawn on the level body, so they always face forward)."""

    def __init__(self, m, P):
        self.m, self.R = m, P.geo.R
        self.g = [m.geom("eye_left").id, m.geom("eye_right").id]
        self.mat_eye = m.material("orb_eye").id
        self.mat_acc = m.material("orb_accent").id
        self.next_blink = 2.5
        self.gz = np.zeros(2)

    def update(self, t, gaze, color, happy=False, alarm=False):
        self.gz += 0.15 * (np.array(gaze) - self.gz)
        blink = 0.0 <= (t - self.next_blink) < 0.14 and not alarm
        if t - self.next_blink > 0.14:
            self.next_blink = t + 2.5 + 3.0 * ((t * 7.31) % 1.0)
        for side, gid in ((+1, self.g[0]), (-1, self.g[1])):
            lat = side * np.radians(10.5) + self.gz[0] * np.radians(9)
            lon = np.radians(90 - 11) - self.gz[1] * np.radians(6) - (np.radians(2.0) if happy else 0)
            p = meshes._sph(self.R + 0.0015, lon, lat)
            n = p / np.linalg.norm(p)
            q = np.zeros(4); mujoco.mju_quatZ2Vec(q, n)
            self.m.geom_pos[gid] = p; self.m.geom_quat[gid] = q
            h = 0.036 * (0.12 if blink else (0.42 if happy else (1.25 if alarm else 1.0)))
            self.m.geom_size[gid] = [h, 0.026 if not happy else 0.030, 0.0015]
        self.m.mat_rgba[self.mat_eye][:3] = color
        self.m.mat_rgba[self.mat_acc][:3] = color
        pulse = 0.65 + 0.35 * np.sin(t * (14.0 if alarm else 3.0)) if color != CYAN else 0.85
        self.m.mat_emission[self.mat_acc] = pulse


class Director:
    """Chase camera that frames the action."""

    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.lookat = None
        self.az = 200.0; self.el = -16.0; self.dist = 3.4

    def update(self, t, pos, heading, state, dt):
        want_az = np.degrees(heading) + 180 - 25
        want_el, want_d = -16.0, 3.6
        la = pos + np.array([0.6 * np.cos(heading), 0.6 * np.sin(heading), 0.0])
        if state in ("XWALK_WAIT",):
            want_az, want_el, want_d = np.degrees(heading) + 180 + 55, -22, 9.0
            la = pos + np.array([3.5, 0, 0])
        elif state in ("CURB_ALIGN", "CURB"):
            want_az, want_el, want_d = np.degrees(heading) - 70, -10, 2.6
            la = pos + np.array([0.25, 0, -0.05])
        elif state in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE"):
            want_az, want_el, want_d = 70 + 10 * np.sin(t * 0.3), -26, 3.1
            la = pos + np.array([0.15, 0.55, 0.15])
        elif state in ("YIELD", "EMERG"):
            want_d, want_el = 4.6, -30
        if self.lookat is None:
            self.lookat = la.copy(); self.az = want_az
        k = min(1.0, dt / 0.6)
        self.lookat += k * (la - self.lookat)
        self.az += k * wrap(np.radians(want_az - self.az)) * 180 / np.pi
        self.el += k * (want_el - self.el)
        self.dist += k * (want_d - self.dist)
        c = self.cam
        c.lookat[:] = self.lookat; c.azimuth = self.az; c.elevation = self.el; c.distance = self.dist
        return c


def default_wind(seed=4):
    """A breezy Kansas afternoon: 6 m/s from the south with gusts to ~14 m/s."""
    return Wind(6.0, np.pi / 2, turb=0.18, seed=seed,
                gusts=[(9.0, 3.0, 8.0, 0.0), (24.0, 3.5, 7.0, 0.3), (50.0, 3.0, 8.0, -0.2), (75.0, 3.0, 6.0, 0.0)])


class Sim:
    def __init__(self, P=None, video=True, out_dir="out", seed=1, W=1280, H=720, fps=24, payload=None, wind="default", start=None):
        self.P = P or Params()
        P = self.P
        if payload is not None:
            P.m.payload = payload
        x0, y0, yaw0 = start if start is not None else (-0.8, 0.0, 0.0)
        rx = model.build(P, pos=(x0, y0), yaw=yaw0, z0=0.0)
        xml, self.cmap = world.build_world(rx)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.rb = Robot(m, P)
        self.ll = LowLevel(P)
        self.sen = SensorModel(P, seed=seed)
        self.legs = Legs(P, self.rb)
        self.pe = Perception(m, seed=seed)
        self.pl = Planner(self.cmap, self.ll)
        self.sens = SensorSim(seed)
        self.est = Estimator(x0, y0, yaw0)
        self.pl.s = self.pl.route.project(np.array([x0, y0]))[0]
        if self.pl.s > self.pl.s_xout:
            self.pl.crossed = True
        self.eyes = Eyes(m, P)
        self.dir = Director()
        self.safety = SafetyMCU(m, self.rb)
        self.wind = default_wind(seed) if wind == "default" else wind
        self.video, self.W, self.H, self.fps = video, W, H, fps
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.agents = self.cmap.agents
        self.mocap = {}
        for ag in self.agents:
            self.mocap[ag.name] = m.body_mocapid[m.body(ag.name).id]
            if ag.kind == "ped":
                self.mocap[ag.name + "_l"] = m.body_mocapid[m.body(ag.name + "_l").id]
                self.mocap[ag.name + "_r"] = m.body_mocapid[m.body(ag.name + "_r").id]
        self.eq_band = m.equality("brake_band").id
        self.eq_yoke = m.equality("brake_yoke").id
        self.eq_lock = m.equality("lock_level").id
        self.payload_geoms = [m.geom(n).id for n in ("food_bag", "food_cup", "food_cup_lid")]
        self.agent_geoms = set()
        for ag in self.agents:
            for suffix in ("", "_l", "_r"):
                try:
                    bid = m.body(ag.name + suffix).id
                except KeyError:
                    continue
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == bid:
                        self.agent_geoms.add(g)
        self.robot_bodies = self.rb.body_ids
        self.log = []
        self.energy_J = 0.0
        self.energy_parts = {"drive": 0.0, "steer": 0.0, "level": 0.0, "legs": 0.0, "hotel": 0.0}
        self.gnss_pts = []
        self.trail, self.trail_est = [], []
        self.contacts_people = 0
        self.min_clear = {}
        self.food_taken = False
        self.climber = None
        self.cliff = CliffSensor(self.m, self.rb, seed=seed)
        self.climb_done = False
        self.climb_log = []
        self.wind_est = 0.0
        self.food_af = np.array([0, 0, 9.81])
        self.agent_prev = {}
        self.park_t = None
        self.still_t = 0.0
        self.legs_ok_t = None
        self.last_frame = None

    # ------------------------------------------------------------------ helpers
    def brake(self, on):
        m, d, rb = self.m, self.d, self.rb
        if on and not d.eq_active[self.eq_band]:
            m.eq_data[self.eq_band][0] = d.qpos[rb.qa["pod_pitch"]]
            m.eq_data[self.eq_yoke][0] = d.qpos[rb.qa["drive"]]
        d.eq_active[self.eq_band] = int(on); d.eq_active[self.eq_yoke] = int(on)
        self.legs.lock_wheels(m, on)

    def lock_level(self, on):
        m, d, rb = self.m, self.d, self.rb
        if on and not d.eq_active[self.eq_lock]:
            m.eq_data[self.eq_lock][0] = d.qpos[rb.qa["pod_pitch"]] - d.qpos[rb.qa["drive"]]
        d.eq_active[self.eq_lock] = int(on)

    def update_agents(self, t, rpos):
        d = self.d
        for ag in self.agents:
            if ag.started is None and ag.trigger is not None and ag.trigger(rpos, t):
                ag.started = t
            p, yaw, s = ag.pose(t)
            q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            mid = self.mocap[ag.name]
            d.mocap_pos[mid] = [p[0], p[1], ag.z]
            d.mocap_quat[mid] = q
            if ag.kind == "ped":
                moving = ag.started is not None and s is not None and ag.speed > 0
                ph = (s or 0.0) / (0.72 * ag.scale) * np.pi if moving else 0.0
                sw = 0.22 * ag.scale * np.sin(ph) if moving else 0.0
                fwd = np.array([np.cos(yaw), np.sin(yaw)])
                for suf, sg in (("_l", 1), ("_r", -1)):
                    dm = self.mocap[ag.name + suf]
                    d.mocap_pos[dm] = [p[0] + sg * sw * fwd[0] * 0.5, p[1] + sg * sw * fwd[1] * 0.5, 0.0 + ag.z]
                    d.mocap_quat[dm] = q
            rad = {"ped": 0.2 * ag.scale, "bike": 0.35, "car": 1.0}[ag.kind]
            dist = np.hypot(p[0] - rpos[0], p[1] - rpos[1]) - (0.325 + rad)
            self.min_clear[ag.name] = min(self.min_clear.get(ag.name, 9e9), dist)

    def agent_vel(self, ag, t):
        p1 = np.array(ag.pose(t)[0]); p0 = np.array(ag.pose(max(0.0, t - 0.1))[0])
        return p1, (p1 - p0) / 0.1

    def radar(self, t, st):
        """Two 24 GHz radars looking left/right of the heading: range, bearing, radial speed of cars."""
        out = []
        rng = self.sens.rng
        for ag in self.agents:
            if ag.kind != "car":
                continue
            p, v = self.agent_vel(ag, t)
            rel = p - st["pos"][:2]
            dist = float(np.linalg.norm(rel))
            if dist > 80 or rng.random() > 0.95:
                continue
            bearing = wrap(np.arctan2(rel[1], rel[0]) - st["heading"])
            if not (np.radians(20) < abs(bearing) < np.radians(160)):
                continue
            p_m = p + rng.normal(0, 0.3, 2); v_m = v + rng.normal(0, 0.2, 2)
            yc, half = self.cmap.crosswalk[2], 1.4 + 2.3
            dy = yc - p_m[1]
            approaching = dy * v_m[1] > 0 and abs(v_m[1]) > 0.3
            in_zone = abs(dy) < half
            tta = 0.0 if in_zone else (max(0.0, (abs(dy) - half) / max(abs(v_m[1]), 0.3)) if approaching else 99.0)
            out.append(dict(p=p_m, v=v_m, dist=dist, tta=tta, approaching=bool(approaching or in_zone)))
        return out

    def rear(self, t, st):
        """Rear fisheyes: anything coming up from behind (bikes, scooters, runners)."""
        out = []
        fwd, left = st["fwd"][:2], st["left"][:2]
        for ag in self.agents:
            if ag.kind == "car" or ag.started is None:
                continue
            p, v = self.agent_vel(ag, t)
            rel = p - st["pos"][:2]
            along, lat = float(rel @ fwd), float(rel @ left)
            if -10.0 < along < -0.4 and abs(lat) < 3.0:
                out.append(dict(dist=float(np.hypot(along, lat)), lat=lat, closing=float(v @ fwd) - st["v"]))
        return out

    def road_objects(self):
        cnt = 0
        rp = self.cmap.road_poly
        for cam, v in self.pe.last.items():
            if "world" not in v:
                continue
            Pw = v["world"]
            m_ = (Pw[:, 0] > rp[0] + 0.2) & (Pw[:, 0] < rp[1] - 0.2) & (Pw[:, 2] > world.ROAD_Z + 0.35) & (Pw[:, 2] < world.ROAD_Z + 2.0) & (np.abs(Pw[:, 1] - 24) < 30)
            cnt += int(m_.sum())
        return cnt

    # ------------------------------------------------------------------ run
    def run(self, T=200.0, verbose=True, hook=None):
        m, d, rb, ll, pe, pl = self.m, self.d, self.rb, self.ll, self.pe, self.pl
        P = self.P
        dt = m.opt.timestep
        n_ctrl = max(1, int(round(P.control_dt / dt)))
        cdt = dt * n_ctrl
        n_perc = int(round(0.1 / dt))
        W = P.total_mass() * 9.81
        writer = None
        if self.video:
            ren = mujoco.Renderer(m, self.H, self.W)
            path = os.path.join(self.out_dir, "delivery.mp4")
            writer = cv2.VideoWriter(path + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.W, self.H))
        next_frame = 0.0
        step = 0
        wall0 = time.time()
        yaw_cmd, v_lim = 0.0, 0.0
        mujoco.mj_forward(m, d)
        pos_prev = rb.state(d)["pos"].copy()
        done_t = None
        road_obj = 0
        info = {}
        hb = 0.0
        s_meas = None
        climb_x = 0.0
        while d.time < T:
            t = d.time
            w = self.wind.at(t) if self.wind is not None else np.zeros(2)
            self.wind_force = apply_wind(m, d, P, w, rb.bid_shell, rb.bid_lid, d.qpos[rb.qa["lid"]])
            if step % n_ctrl == 0:
                st = rb.state(d)
                s_meas = self.sen.measure(st)
                # ---- odometry + gyro -> estimator
                dpos = st["pos"] - pos_prev; pos_prev = st["pos"].copy()
                ds_true = float(dpos[:2] @ st["fwd"][:2])
                ds_odo = self.sens.odom(ds_true)
                self.est.predict(ds_odo, self.sens.gyro(st["yaw_rate"], cdt))
                pl.odo_s += ds_odo
                climb_x += ds_odo
                if step % int(1.0 / dt) == 0:
                    z = self.sens.gnss(st["pos"][:2])
                    self.gnss_pts.append(z)
                    if pl.state not in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE", "ALIGN", "CURB"):
                        self.est.gnss(z, k=0.03 if abs(st["v"]) > 0.1 else 0.004)
                self.update_agents(t, st["pos"])
                # ---- the robot's own wind estimate (from the tray offset its steering needed)
                F = ll.crosswind_estimate()
                straight = abs(st["v"]) > 0.7 and abs(yaw_cmd) < 0.06 and pl.state in ("DRIVE", "APPROACH")
                U_est = float(np.sqrt(abs(F) / (0.5 * P.a.rho * P.a.cd_sphere * np.pi * P.geo.R ** 2))) if straight else self.wind_est
                self.wind_est += 0.01 * (U_est - self.wind_est)
                # ---- perception + planning at 10 Hz
                if step % n_perc == 0:
                    hb = t
                    cams = ["cam_left", "cam_right"]
                    if pl.state == "XWALK_WAIT":
                        cams += ["cam_left_side", "cam_right_side"]
                    if pl.state not in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE", "CURB"):
                        pe.step(d, st["pos"], st["heading"], cams=cams)
                        road_obj = self.road_objects()
                        va = pl.vision_along(self.est, pe.bev)
                        if va is not None:
                            self.est.along(*va)
                        vl = pl.vision_lateral(self.est, pe.bev)
                        if vl is not None:
                            innov, nrm, dpsi = vl
                            if abs(innov) < 1.5:
                                self.est.lateral(innov, nrm)
                                self.est.heading_fix(dpsi)
                    ctx = dict(wind=self.wind_est, radar=self.radar(t, st) if pl.state == "XWALK_WAIT" else [],
                               rear=self.rear(t, st), climb_done=self.climb_done)
                    v_lim, yaw_cmd, info = pl.tick(t, self.est, pe.bev, st, road_obj, 0.1, ctx)
                    self._delivery_fsm(t, st, W)
                    self.trail.append(st["pos"][:2].copy()); self.trail_est.append(self.est.p.copy())
                    if pl.state == "CURB" and self.climber is None and pl.curb_req is not None:
                        cr = pl.curb_req
                        self.climber = CurbClimber(P, self.legs, cr["h"], climb_x + cr["d_edge"], up=cr["up"], ll=ll)
                        self.climb_t0 = t; self.climb_done = False
                    if pl.state == "CURB_EXIT" and self.climb_done:
                        self.climb_done = False
                # ---- independent safety MCU
                legs_down = pl.state in ("CURB", "PARK", "OPEN", "HANDOFF", "CLOSE", "BRACE")
                trig = self.safety.update(d, t, abs(st["v"]), st["lean"], st["pod_pitch"], hb, legs_down)
                # ---- low-level control (250 Hz)
                self._park_fast(t, st, W)
                # parked = braked: the speed loop holds the ball until it's still, then the brake goes on
                parked = (pl.state in ("PARK", "OPEN", "HANDOFF", "CLOSE", "BRACE") and self.park_t is not None) or (pl.state == "DONE" and done_t is not None and t - done_t < 0.8)
                lock = False
                if parked:
                    d.ctrl[rb.act["drive"]] = 0.0
                    ll.reset()
                    d.ctrl[rb.act["steer"]] = 0.0
                elif self.climber is not None:
                    vd = self.climber.step(t, dict(s_meas, legF=self.legs.forces(d), cliff=self.cliff.read(d),
                                                   wheel_v=float(np.mean([d.qvel[m.jnt_dofadr[m.joint(f'wheel_{k}').id]] for k in (3, 4)]) * P.geo.leg_wheel_r)),
                                           climb_x)
                    lock = isinstance(vd, dict) and "frame" in vd
                    if lock:
                        c = ll.hold_frame(s_meas)
                    elif isinstance(vd, dict):
                        c = ll.hold(s_meas, vd.get("hold", 0.0))
                    else:
                        c, _ = ll.speed(s_meas, vd, cdt)
                    d.ctrl[rb.act["drive"]] = c
                    d.ctrl[rb.act["steer"]] = ll.steer(s_meas, 0.0, cdt)
                    if self.climber.result is not None:
                        self.climb_log.append(dict(result=self.climber.result, h=self.climber.h, up=self.climber.up,
                                                   t=round(t - self.climb_t0, 1), events=self.climber.events))
                        self.climber = None; self.climb_done = True
                        ll.on_legs = False; ll.th_max = np.radians(60); ll.soft_reset(s_meas["v"])
                else:
                    ll.emergency = bool(pl.emergency or trig)
                    c, _ = ll.speed(s_meas, 0.0 if trig else v_lim, cdt)
                    d.ctrl[rb.act["drive"]] = c
                    d.ctrl[rb.act["steer"]] = ll.steer(s_meas, yaw_cmd, cdt)
                self.lock_level(lock)
                d.ctrl[rb.act["pod_level"]] = 0.0 if (parked or lock) else ll.level(s_meas)
                # ---- eyes
                col = STATE_COLORS.get(pl.state, CYAN)
                alarm = bool(trig or pl.emergency)
                if alarm:
                    col = RED
                elif pl.msg.startswith(("yielding", "waiting")) and pl.state in ("DRIVE", "XWALK_CROSS", "APPROACH"):
                    col = AMBER
                if step % (n_ctrl * 10) == 0:
                    self.eyes.update(t, pl.gaze if not parked else (0.35 * np.sin(t * 0.8), -0.4), col,
                                     happy=pl.state in ("HANDOFF", "CLOSE", "DONE"), alarm=alarm)
            self.legs.step(d, dt)
            mujoco.mj_step(m, d)
            step += 1
            self._energy(dt)
            if hook is not None and hook(self) is False:
                break
            # ---- people contacts
            for i in range(d.ncon):
                cg = d.contact[i]
                if (cg.geom1 in self.agent_geoms and m.geom_bodyid[cg.geom2] in self.robot_bodies) or \
                   (cg.geom2 in self.agent_geoms and m.geom_bodyid[cg.geom1] in self.robot_bodies):
                    self.contacts_people += 1
            # ---- log (50 Hz)
            if step % 10 == 0:
                st = rb.state(d)
                acc = d.sensor("pod_acc").data
                self.food_af = self.food_af + 0.4 * (acc - self.food_af)
                af = self.food_af
                soup = np.degrees(np.arctan2(np.hypot(af[0], af[1]), af[2])) if af[2] > 4.9 else np.nan
                self.log.append(dict(t=d.time, x=st["pos"][0], y=st["pos"][1], z=st["pos"][2], v=st["v"],
                                     ex=self.est.x, ey=self.est.y, state=pl.state,
                                     pod_pitch=np.degrees(st["pod_pitch"]), pod_roll=np.degrees(st["pod_roll"]),
                                     soup=soup, food_g=float(np.linalg.norm(af - [0, 0, 9.81]) / 9.81),
                                     tau=float(d.actuator_force[rb.act["drive"]]), lean=np.degrees(st["lean"]),
                                     steer=np.degrees(st["steer"]), E=self.energy_J,
                                     loc_err=float(np.hypot(self.est.x - st["pos"][0], self.est.y - st["pos"][1])),
                                     free=pl.free, msg=pl.msg, reason=pl.speed_reason, v_lim=float(v_lim),
                                     wind=float(np.linalg.norm(w)), wind_est=self.wind_est, safety=int(self.safety.trig),
                                     emerg=int(pl.emergency), tof=float(self.safety.tof.min())))
            # ---- video
            if self.video and d.time >= next_frame:
                next_frame += 1.0 / self.fps
                st = rb.state(d)
                vstate = pl.state
                if self.safety.trig or pl.emergency:
                    vstate = "EMERG"
                elif pl.msg.startswith("yielding"):
                    vstate = "YIELD"
                cam = self.dir.update(d.time, st["pos"], st["heading"], vstate, 1.0 / self.fps)
                ren.update_scene(d, cam)
                frame = ren.render()
                frame = self._overlay(frame, st, info, w)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.last_frame = frame
            if pl.state == "DONE":
                done_t = done_t or d.time
                if d.time - done_t > 4.0:
                    break
            if verbose and step % int(5.0 / dt) == 0:
                st = rb.state(d)
                print(f"t={d.time:6.1f}s  {pl.state:11s} {pl.msg[:30]:30s} x={st['pos'][0]:6.2f} y={st['pos'][1]:6.2f} v={st['v']:+.2f} "
                      f"cap={pl.speed_reason[:16]:16s} wind={np.linalg.norm(w):4.1f}/{self.wind_est:4.1f} loc={np.hypot(self.est.x-st['pos'][0], self.est.y-st['pos'][1]):.2f} wall={time.time()-wall0:5.0f}s", flush=True)
        if writer:
            writer.release()
            src = os.path.join(self.out_dir, "delivery.mp4.tmp.mp4")
            dst = os.path.join(self.out_dir, "delivery.mp4")
            os.system(f'ffmpeg -y -loglevel error -i "{src}" -c:v libx264 -pix_fmt yuv420p -crf 21 -preset medium "{dst}" && rm -f "{src}"')
        return self.summary()

    # ------------------------------------------------------------------ parking (250 Hz)
    def _park_fast(self, t, st, W):
        """PARK / BRACE: speed loop stops the ball; once still, brake; then legs."""
        if self.pl.state not in ("PARK", "BRACE"):
            return
        if self.park_t is None:
            self.still_t = self.still_t + self.P.control_dt if abs(st["v"]) < 0.02 else 0.0
            if self.still_t > 0.3:
                self.brake(True); self.park_t = t; self.legs_ok_t = None
            return
        if self.legs.park(self.d, W, t - self.park_t) and self.legs_ok_t is None:
            self.legs_ok_t = t

    # ------------------------------------------------------------------ delivery actuators
    def _delivery_fsm(self, t, st, W):
        m, d, rb, pl = self.m, self.d, self.rb, self.pl
        g = self.P.geo
        legs_q = np.array([d.qpos[q] for q in rb.leg_q])
        if pl.state == "BRACE":
            return
        if pl.state == "PARK" and self.park_t is None:
            pl.msg = "stopping to park"
            return
        if pl.state == "DRIVE" and self.park_t is not None and not self.food_taken:
            # wind dropped: legs up, brake off
            self.legs.set(0.0)
            if legs_q.max() < 0.01:
                self.brake(False); self.park_t = None
            return
        if pl.state == "PARK":
            pl.msg = "parking brake + legs down"
            if self.legs_ok_t is not None and t - self.legs_ok_t > 0.3:
                pl.set_state("OPEN", t)
        elif pl.state == "OPEN":
            if self.wind_est > 12.0 or np.linalg.norm(self.wind.last if self.wind else [0, 0]) > 16.0:
                pl.msg = "too gusty - hatch stays shut for now"
                return
            pl.msg = "opening hatch"
            q = d.qpos[rb.qa["lid"]]
            d.ctrl[rb.act["lid"]] = max(-1.95, d.ctrl[rb.act["lid"]] - 0.08)
            if q < -1.85:
                pl.set_state("HANDOFF", t)
                cust = [a for a in self.agents if a.name == "ped_cust"][0]
                cust.path = [(60.7, 26.15), (st["pos"][0] + 0.25, st["pos"][1] + 0.9)]
                cust.started = t
        elif pl.state == "HANDOFF":
            pl.msg = "customer picking up order"
            if t - pl.t_state > 3.2 and not self.food_taken:
                self.food_taken = True
                for gid in self.payload_geoms:
                    m.geom_rgba[gid][3] = 0.0
                m.body_mass[rb.bid_payload] = 1e-3
                m.body_inertia[rb.bid_payload] = [1e-5] * 3
                cust = [a for a in self.agents if a.name == "ped_cust"][0]
                p0 = (st["pos"][0] + 0.25, st["pos"][1] + 0.9)
                cust.path = [p0, (60.7, 26.15)]
                cust.started = t
            if t - pl.t_state > 5.0:
                pl.set_state("CLOSE", t)
        elif pl.state == "CLOSE":
            pl.msg = "closing + locking hatch"
            d.ctrl[rb.act["lid"]] = min(0.0, d.ctrl[rb.act["lid"]] + 0.08)
            if d.qpos[rb.qa["lid"]] > -0.03 and t - pl.t_state > 1.5:
                self.legs.set(0.0)
                if legs_q.max() < 0.01:
                    self.brake(False)
                    pl.set_state("DONE", t)
        elif pl.state == "DONE":
            pl.msg = "delivered - heading back"

    # ------------------------------------------------------------------ energy
    def _energy(self, dt):
        d, rb, E = self.d, self.rb, self.P.e
        tau = d.actuator_force[rb.act["drive"]]; w = d.qvel[rb.va["drive"]]
        p_drive = max(tau * w, 0.0) / E.drivetrain_eff + 0.02 * tau * tau
        ts = d.actuator_force[rb.act["steer"]]; ws = d.qvel[rb.va["steer"]]
        p_steer = abs(ts * ws) / 0.6 + 0.01 * ts * ts + 0.3
        tl = d.actuator_force[rb.act["pod_level"]]
        p_lvl = 0.02 * tl * tl
        p_legs = sum(abs(d.actuator_force[rb.act[f"leg_{k}"]] * d.qvel[rb.leg_v[k - 1]]) / 0.35 for k in range(1, 5))
        p_hotel = sum(E.hotel_w.values())
        for k, p in (("drive", p_drive), ("steer", p_steer), ("level", p_lvl), ("legs", p_legs), ("hotel", p_hotel)):
            self.energy_parts[k] += p * dt
        self.energy_J += (p_drive + p_steer + p_lvl + p_legs + p_hotel) * dt

    # ------------------------------------------------------------------ overlay
    def _overlay(self, frame, st, info, wind):
        img = frame.copy()
        W, H = self.W, self.H
        pe, pl = self.pe, self.pl
        x0 = W - 500; y0 = 12
        cv2.rectangle(img, (x0 - 6, y0 - 6), (W - 6, y0 + 186 + 200), (15, 18, 24), -1)
        for k, cam in enumerate(("cam_left", "cam_right")):
            if cam in pe.last:
                v = cv2.resize(pe.last[cam]["rgb"], (240, 180), interpolation=cv2.INTER_NEAREST)
                img[y0:y0 + 180, x0 + k * 246:x0 + k * 246 + 240] = v
        cv2.putText(img, "LEFT CAP CAM", (x0 + 6, y0 + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "RIGHT CAP CAM", (x0 + 252, y0 + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        bev = pe.bev_image(scale=3)
        bh, bw = bev.shape[:2]
        by = y0 + 190
        bev = bev.copy()
        b = pe.bev
        def to_px(x, y):
            return int((b.y1 - y) / b.res * 3), int((b.x1 - x) / b.res * 3)
        cx, cy = to_px(0, 0)
        cv2.circle(bev, (cx, cy), int(0.325 / b.res * 3), (80, 230, 255), 2)
        if "target" in info:
            q = info["target"] - self.est.p
            c, s_ = np.cos(-self.est.psi), np.sin(-self.est.psi)
            tx, ty = c * q[0] - s_ * q[1], s_ * q[0] + c * q[1]
            px = to_px(tx, ty)
            cv2.line(bev, (cx, cy), px, (80, 230, 255), 2)
            cv2.circle(bev, px, 4, (80, 230, 255), -1)
        img[by:by + bh, x0:x0 + bw] = bev[:, :, :3]
        cv2.putText(img, "BIRD'S-EYE (from cameras)", (x0 + 4, by + bh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        lx = x0 + bw + 10
        for k, (name, col) in enumerate((("sidewalk", (185, 185, 180)), ("grass", (60, 120, 50)), ("road", (70, 70, 80)),
                                          ("obstacle", (220, 40, 50)), ("step/curb", (230, 150, 30)), ("path", (80, 230, 255)))):
            yy = by + 14 + k * 18
            cv2.rectangle(img, (lx, yy - 9), (lx + 12, yy + 1), col, -1)
            cv2.putText(img, name, (lx + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        # wind gauge
        gx, gy = lx + 40, by + 150
        cv2.circle(img, (gx, gy), 26, (60, 64, 72), 1, cv2.LINE_AA)
        wv = np.asarray(wind, float)
        sp = float(np.linalg.norm(wv))
        if sp > 0.3:
            ang = np.arctan2(wv[1], wv[0]) - st["heading"]        # robot frame: up = forward
            ex, ey = int(gx - 24 * np.sin(ang) * min(1, sp / 15)), int(gy - 24 * np.cos(ang) * min(1, sp / 15))
            cv2.arrowedLine(img, (gx, gy), (ex, ey), (255, 200, 90), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(img, f"wind {sp:4.1f} m/s", (lx - 6, gy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 150), 1, cv2.LINE_AA)
        cv2.putText(img, f"robot est {self.wind_est:4.1f}", (lx - 6, gy + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        # telemetry
        soc = 100 * (1 - self.energy_J / (self.P.e.battery_wh * 3600))
        L = self.log[-1] if self.log else {}
        saf = "STOP: " + self.safety.reason if self.safety.trig else f"clear ({self.safety.tof.min():.1f} m)"
        legs = np.array([self.d.qpos[q] for q in self.rb.leg_q])
        lines = [
            f"STATE  {pl.state}",
            f"       {pl.msg[:34]}",
            f"speed  {st['v']*3.6:4.1f} km/h  (cap: {pl.speed_reason[:18]})",
            f"food   {L.get('food_g', 0):4.2f} g   tilt {abs(np.degrees(st['pod_pitch'])):3.1f} deg",
            f"safety MCU  {saf}",
            f"legs   {'DOWN ' + str(np.round(legs.max()*100)) + ' cm' if legs.max() > 0.01 else 'stowed'}",
            f"fused position err {np.hypot(self.est.x-st['pos'][0], self.est.y-st['pos'][1]):4.2f} m",
            f"battery {soc:5.1f} %    t = {self.d.time:5.1f} s",
        ]
        cv2.rectangle(img, (10, 10), (372, 20 + 22 * len(lines)), (15, 18, 24), -1)
        for k, ln in enumerate(lines):
            col = (255, 255, 255)
            if k == 0:
                col = tuple(int(255 * c) for c in STATE_COLORS.get(pl.state, (1, 1, 1)))
            if k == 4 and self.safety.trig:
                col = (255, 80, 60)
            cv2.putText(img, ln, (18, 32 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if pl.emergency or self.safety.trig:
            cv2.rectangle(img, (W // 2 - 190, H - 70), (W // 2 + 190, H - 36), (200, 30, 30), -1)
            cv2.putText(img, "EMERGENCY STOP", (W // 2 - 120, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "ORB v2  |  campus delivery sphere  |  MuJoCo physics, onboard perception, wind", (14, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ summary
    def summary(self):
        import pandas as pd
        df = pd.DataFrame(self.log)
        moving = df[df.state.isin(["DRIVE", "XWALK_CROSS", "APPROACH"]) & (df.v > 0.05)]
        tr = np.array(self.trail)
        dist = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
        E = self.energy_J / 3600
        dur = float(df.t.iloc[-1])
        avg_w = self.energy_J / max(dur, 1e-6)
        usable = self.P.e.battery_wh * self.P.e.usable_frac
        reasons = df[df.state.isin(["DRIVE", "APPROACH"])].reason.value_counts(normalize=True)
        out = dict(
            sim_time_s=round(dur, 1), distance_m=round(dist, 1),
            delivered=bool(self.food_taken), events=self.pl.events,
            mean_speed_moving_mps=round(float(moving.v.mean()), 2), max_speed_mps=round(float(df.v.max()), 2),
            time_at_cruise_pct=round(float(100 * (moving.v > 1.8).mean()), 1),
            speed_cap_reasons_pct={k: round(100 * float(v), 1) for k, v in reasons.items()},
            max_pod_tilt_deg=round(float(df.pod_pitch.abs().max()), 2),
            p99_pod_tilt_deg=round(float(np.percentile(df.pod_pitch.abs(), 99)), 2),
            max_food_g=round(float(df.food_g.max()), 2), p99_food_g=round(float(np.percentile(df.food_g, 99)), 2),
            max_liquid_tilt_deg=round(float(np.nanmax(df.soup)), 1),
            max_lean_deg=round(float(df.lean.abs().max()), 2),
            loc_err_mean_m=round(float(df.loc_err.mean()), 2), loc_err_max_m=round(float(df.loc_err.max()), 2),
            vision_fixes=self.est.n_vis,
            max_wind_mps=round(float(df.wind.max()), 1), robot_wind_est_max=round(float(df.wind_est.max()), 1),
            emergency_stops=int(((df.emerg.diff() == 1)).sum()), safety_mcu_events=self.safety.events,
            curb_climbs=self.climb_log,
            energy_Wh=round(E, 2), avg_power_W=round(avg_w, 1),
            energy_split_Wh={k: round(v / 3600, 3) for k, v in self.energy_parts.items()},
            runtime_est_h_at_this_duty=round(usable / avg_w, 1),
            robot_contacts_with_people_or_cars=int(self.contacts_people),
            min_clearance_m={k: round(v, 2) for k, v in self.min_clear.items() if k != "ped_bg"},
            peak_drive_torque_Nm=round(float(df.tau.abs().max()), 1),
        )
        df.to_csv(os.path.join(self.out_dir, "delivery_log.csv"), index=False)
        with open(os.path.join(self.out_dir, "delivery_summary.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
        return out
