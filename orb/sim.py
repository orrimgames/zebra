"""Closed-loop campus delivery simulation + video.

    python run_delivery.py            # renders out/delivery.mp4 + metrics
"""
import json, time, os
import numpy as np
import mujoco
import cv2

from .params import P as DEFAULT_P
from . import model, world
from .control import Robot, LowLevel
from .perception import Perception
from .planner import Planner, Estimator, SensorSim, wrap
from . import meshes

STATE_COLORS = {
    "DRIVE": (0.35, 0.95, 1.0), "XWALK_CROSS": (0.35, 0.95, 1.0), "APPROACH": (0.35, 0.95, 1.0),
    "YIELD": (1.0, 0.72, 0.2), "XWALK_WAIT": (1.0, 0.72, 0.2), "ALIGN": (0.6, 0.8, 1.0),
    "PARK": (0.45, 1.0, 0.55), "OPEN": (0.45, 1.0, 0.55), "HANDOFF": (0.45, 1.0, 0.55),
    "CLOSE": (0.45, 1.0, 0.55), "DONE": (0.45, 1.0, 0.55),
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

    def update(self, t, gaze, color, happy=False):
        self.gz += 0.15 * (np.array(gaze) - self.gz)
        blink = 0.0 <= (t - self.next_blink) < 0.14
        if t - self.next_blink > 0.14:
            self.next_blink = t + 2.5 + 3.0 * ((t * 7.31) % 1.0)
        for side, gid in ((+1, self.g[0]), (-1, self.g[1])):
            lat = side * np.radians(10.5) + self.gz[0] * np.radians(9)
            lon = np.radians(90 - 11) - self.gz[1] * np.radians(6) - (np.radians(2.0) if happy else 0)
            p = meshes._sph(self.R + 0.0015, lon, lat)
            n = p / np.linalg.norm(p)
            q = np.zeros(4); mujoco.mju_quatZ2Vec(q, n)
            self.m.geom_pos[gid] = p; self.m.geom_quat[gid] = q
            h = 0.036 * (0.12 if blink else (0.42 if happy else 1.0))
            self.m.geom_size[gid] = [h, 0.026 if not happy else 0.030, 0.0015]
        self.m.mat_rgba[self.mat_eye][:3] = color
        self.m.mat_rgba[self.mat_acc][:3] = color
        pulse = 0.65 + 0.35 * np.sin(t * 3.0) if color != STATE_COLORS["DRIVE"] else 0.85
        self.m.mat_emission[self.mat_acc] = pulse


class Director:
    """Chase camera that frames the action."""

    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.lookat = None
        self.az = 200.0; self.el = -16.0; self.dist = 3.4

    def update(self, t, pos, heading, state, dt):
        want_az = np.degrees(heading) + 180 - 25
        want_el, want_d = -16.0, 3.4
        la = pos + np.array([0.6 * np.cos(heading), 0.6 * np.sin(heading), 0.0])
        if state in ("XWALK_WAIT",):
            want_az, want_el, want_d = np.degrees(heading) + 180 + 55, -22, 9.0
            la = pos + np.array([3.5, 0, 0])
        elif state in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE"):
            # shoot from the street side so the customer (coming from the door) is never between camera and robot
            want_az, want_el, want_d = 70 + 10 * np.sin(t * 0.3), -26, 3.1
            la = pos + np.array([0.15, 0.55, 0.15])
        elif state == "YIELD":
            want_d, want_el = 4.4, -32
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


class Sim:
    def __init__(self, P=DEFAULT_P, video=True, out_dir="out", seed=1, W=1280, H=720, fps=24, payload=None):
        self.P = P
        if payload is not None:
            P.m.payload = payload
        rx = model.build(P, pos=(-0.8, 0.0), yaw=0.0)
        xml, self.cmap = world.build_world(rx)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.rb = Robot(m, P)
        self.ll = LowLevel(P)
        self.pe = Perception(m, seed=seed)
        self.pl = Planner(self.cmap, self.ll)
        self.sen = SensorSim(seed)
        self.est = Estimator(-0.8, 0.0, 0.0)
        self.eyes = Eyes(m, P)
        self.dir = Director()
        self.video, self.W, self.H, self.fps = video, W, H, fps
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.agents = self.cmap.agents
        self.mocap = {}
        for ag in self.agents:
            b = m.body(ag.name)
            self.mocap[ag.name] = m.body_mocapid[b.id]
            if ag.kind == "ped":
                self.mocap[ag.name + "_l"] = m.body_mocapid[m.body(ag.name + "_l").id]
                self.mocap[ag.name + "_r"] = m.body_mocapid[m.body(ag.name + "_r").id]
        self.eq_band = m.equality("brake_band").id
        self.eq_yoke = m.equality("brake_yoke").id
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
        self.robot_bodies = set(range(m.body("shell").id, m.body("shell").id + 11))
        self.log = []
        self.energy_J = 0.0
        self.energy_parts = {"drive": 0.0, "steer": 0.0, "level": 0.0, "hotel": 0.0}
        self.gnss_pts = []
        self.trail, self.trail_est = [], []
        self.contacts_people = 0
        self.min_clear = {}
        self.lid_t = None
        self.food_taken = False
        self.v_des = 0.0; self.yaw_cmd = 0.0
        self.last_frame = None

    # ------------------------------------------------------------------ agents
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
                ph = (s or 0.0) / 0.72 * np.pi if moving else 0.0
                sw = 0.22 * np.sin(ph) if moving else 0.0
                fwd = np.array([np.cos(yaw), np.sin(yaw)])
                for suf, sg in (("_l", 1), ("_r", -1)):
                    dm = self.mocap[ag.name + suf]
                    d.mocap_pos[dm] = [p[0] + sg * sw * fwd[0] * 0.5, p[1] + sg * sw * fwd[1] * 0.5, 0.0 + ag.z]
                    d.mocap_quat[dm] = q
            # clearance book-keeping
            dist = np.hypot(p[0] - rpos[0], p[1] - rpos[1]) - (0.325 + (0.2 if ag.kind == "ped" else 1.0))
            self.min_clear[ag.name] = min(self.min_clear.get(ag.name, 9e9), dist)

    # ------------------------------------------------------------------ road check for crossing
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
    def run(self, T=150.0, verbose=True, hook=None):
        m, d, rb, ll, pe, pl = self.m, self.d, self.rb, self.ll, self.pe, self.pl
        dt = m.opt.timestep
        n_perc = int(round(0.1 / dt))
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
        s_prev = rb.state(d)
        pos_prev = s_prev["pos"].copy()
        done_t = None
        road_obj = 0
        leg_cmd, lid_cmd = 0.0, 0.0
        info = {}
        while d.time < T:
            t = d.time
            st = rb.state(d)
            # -------- sensors -> estimator (every step)
            dpos = st["pos"] - pos_prev; pos_prev = st["pos"].copy()
            ds_true = float(dpos[:2] @ st["fwd"][:2])
            self.est.predict(self.sen.odom(ds_true), self.sen.gyro(st["yaw_rate"], dt))
            pl.odo_s += ds_true
            if step % int(1.0 / dt) == 0:
                z = self.sen.gnss(st["pos"][:2])
                self.gnss_pts.append(z)
                if pl.state not in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE", "ALIGN"):
                    self.est.gnss(z, k=0.03 if abs(st["v"]) > 0.1 else 0.004)
            self.update_agents(t, st["pos"])
            # -------- perception + planning at 10 Hz
            if step % n_perc == 0:
                cams = ["cam_left", "cam_right"]
                if pl.state == "XWALK_WAIT":
                    cams += ["cam_left_side", "cam_right_side"]
                if pl.state not in ("PARK", "OPEN", "HANDOFF", "CLOSE", "DONE"):
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
                v_lim, yaw_cmd, info = pl.tick(t, self.est, pe.bev, st, road_obj, 0.1)
                # delivery sequence (actuators)
                self._delivery_fsm(t, st)
                self.trail.append(st["pos"][:2].copy()); self.trail_est.append(self.est.p.copy())
            # -------- low level control (every step)
            parked = pl.state in ("PARK", "OPEN", "HANDOFF", "CLOSE") or (pl.state == "DONE" and done_t is not None and t - done_t < 0.8)
            if parked:
                d.ctrl[rb.act["drive"]] = 0.0
                d.ctrl[rb.act["steer"]] = 0.0
                ll.reset()
            else:
                c, _ = ll.speed(st, v_lim, dt)
                d.ctrl[rb.act["drive"]] = c
                d.ctrl[rb.act["steer"]] = ll.steer(st, yaw_cmd, dt)
            d.ctrl[rb.act["pod_level"]] = ll.level(st) if not parked else 0.0
            # -------- eyes
            col = STATE_COLORS.get(pl.state, (0.35, 0.95, 1.0))
            if pl.msg.startswith("yielding") and pl.state in ("DRIVE", "XWALK_CROSS", "APPROACH"):
                col = STATE_COLORS["YIELD"]
            self.eyes.update(t, pl.gaze if not parked else (0.35 * np.sin(t * 0.8), -0.4), col,
                             happy=pl.state in ("HANDOFF", "CLOSE", "DONE"))
            mujoco.mj_step(m, d)
            step += 1
            self._energy(dt, parked)
            if hook is not None and hook(self) is False:
                break
            # -------- people contacts
            for i in range(d.ncon):
                cg = d.contact[i]
                if (cg.geom1 in self.agent_geoms and m.geom_bodyid[cg.geom2] in self.robot_bodies) or \
                   (cg.geom2 in self.agent_geoms and m.geom_bodyid[cg.geom1] in self.robot_bodies):
                    self.contacts_people += 1
            # -------- log (50 Hz)
            if step % 10 == 0:
                acc = d.sensor("pod_acc").data
                fmag = float(np.linalg.norm(acc))
                # liquid-surface tilt relative to the pod floor; undefined while the ball hops (f ~ 0)
                soup = np.degrees(np.arctan2(np.hypot(acc[0], acc[1]), acc[2])) if fmag > 4.9 else np.nan
                self.log.append(dict(t=t, x=st["pos"][0], y=st["pos"][1], z=st["pos"][2], v=st["v"],
                                     ex=self.est.x, ey=self.est.y, state=pl.state,
                                     pod_pitch=np.degrees(st["pod_pitch"]), pod_roll=np.degrees(st["pod_roll"]),
                                     soup=soup, acc=float(np.hypot(acc[0], acc[1])), fz_g=float(acc[2] / 9.81),
                                     tau=float(-d.ctrl[rb.act["drive"]]), lean=np.degrees(st["lean"]),
                                     steer=np.degrees(d.qpos[rb.qa["steer"]]), E=self.energy_J,
                                     loc_err=float(np.hypot(self.est.x - st["pos"][0], self.est.y - st["pos"][1])),
                                     free=pl.free, msg=pl.msg))
            # -------- video
            if self.video and t >= next_frame:
                next_frame += 1.0 / self.fps
                cam = self.dir.update(t, st["pos"], st["heading"], pl.state if not pl.msg.startswith("yielding") else "YIELD", 1.0 / self.fps)
                ren.update_scene(d, cam)
                frame = ren.render()
                frame = self._overlay(frame, st, info)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.last_frame = frame
            if pl.state == "DONE":
                done_t = done_t or t
                if t - done_t > 4.0:
                    break
            if verbose and step % int(5.0 / dt) == 0:
                print(f"t={t:6.1f}s  {pl.state:11s} {pl.msg:28s} x={st['pos'][0]:6.2f} y={st['pos'][1]:6.2f} v={st['v']:+.2f} "
                      f"loc_err={np.hypot(self.est.x-st['pos'][0], self.est.y-st['pos'][1]):.2f}  wall={time.time()-wall0:5.0f}s", flush=True)
        if writer:
            writer.release()
            src = os.path.join(self.out_dir, "delivery.mp4.tmp.mp4")
            dst = os.path.join(self.out_dir, "delivery.mp4")
            os.system(f'ffmpeg -y -loglevel error -i "{src}" -c:v libx264 -pix_fmt yuv420p -crf 21 -preset medium "{dst}" && rm -f "{src}"')
        return self.summary()

    # ------------------------------------------------------------------ delivery actuators
    def _delivery_fsm(self, t, st):
        m, d, rb, pl = self.m, self.d, self.rb, self.pl
        g = self.P.geo
        legs_q = np.array([d.qpos[q] for q in rb.leg_q])
        if pl.state == "PARK":
            pl.msg = "parking brake + legs down"
            if not d.eq_active[self.eq_band]:
                m.eq_data[self.eq_band][0] = d.qpos[rb.qa["pod_pitch"]]
                m.eq_data[self.eq_yoke][0] = d.qpos[rb.qa["drive"]]
                d.eq_active[self.eq_band] = 1; d.eq_active[self.eq_yoke] = 1
            for k in range(4):
                d.ctrl[rb.act[f"leg_{k+1}"]] = g.leg_stroke
            if legs_q.min() > g.leg_stroke - 0.012 or t - pl.t_state > 3.0:
                pl.set_state("OPEN", t)
        elif pl.state == "OPEN":
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
                for k in range(4):
                    d.ctrl[rb.act[f"leg_{k+1}"]] = 0.0
                if legs_q.max() < 0.01:
                    d.eq_active[self.eq_band] = 0; d.eq_active[self.eq_yoke] = 0
                    pl.set_state("DONE", t)
        elif pl.state == "DONE":
            pl.msg = "delivered - heading back"

    # ------------------------------------------------------------------ energy
    def _energy(self, dt, parked):
        d, rb, E = self.d, self.rb, self.P.e
        tau = d.ctrl[rb.act["drive"]]; w = d.qvel[rb.va["drive"]]
        pm = tau * w
        p_drive = (max(pm, 0.0) / E.drivetrain_eff) + 0.025 * tau * tau
        ts = d.actuator_force[rb.act["steer"]]; ws = d.qvel[rb.va["steer"]]
        p_steer = abs(ts * ws) / 0.6 + 0.05 * ts * ts + 0.3
        tl = d.ctrl[rb.act["pod_level"]]; wl = d.qvel[rb.va["pod_pitch"]] - d.qvel[rb.va["drive"]]
        p_lvl = abs(tl * wl) / 0.6 + 0.2 * tl * tl
        p_hotel = sum(E.hotel_w.values())
        self.energy_parts["drive"] += p_drive * dt
        self.energy_parts["steer"] += p_steer * dt
        self.energy_parts["level"] += p_lvl * dt
        self.energy_parts["hotel"] += p_hotel * dt
        self.energy_J += (p_drive + p_steer + p_lvl + p_hotel) * dt

    # ------------------------------------------------------------------ overlay
    def _overlay(self, frame, st, info):
        img = frame.copy()
        W, H = self.W, self.H
        pe, pl = self.pe, self.pl
        # robot camera feeds
        x0 = W - 500; y0 = 12
        cv2.rectangle(img, (x0 - 6, y0 - 6), (W - 6, y0 + 186 + 200), (15, 18, 24), -1)
        for k, cam in enumerate(("cam_left", "cam_right")):
            if cam in pe.last:
                v = cv2.resize(pe.last[cam]["rgb"], (240, 180), interpolation=cv2.INTER_NEAREST)
                img[y0:y0 + 180, x0 + k * 246:x0 + k * 246 + 240] = v
        cv2.putText(img, "LEFT CAP CAM", (x0 + 6, y0 + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "RIGHT CAP CAM", (x0 + 252, y0 + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        # BEV
        bev = pe.bev_image(scale=3)
        bh, bw = bev.shape[:2]
        by = y0 + 190
        bev = bev.copy()
        # robot marker: x in [-0.6,6], y in [-3,3]; image row = (x1-x)/res*scale, col=(y1-y)/res*scale
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
        # legend
        lx = x0 + bw + 10
        for k, (name, col) in enumerate((("sidewalk", (185, 185, 180)), ("grass", (60, 120, 50)), ("road", (70, 70, 80)),
                                          ("obstacle", (220, 40, 50)), ("step/curb", (230, 150, 30)), ("path", (80, 230, 255)))):
            yy = by + 14 + k * 18
            cv2.rectangle(img, (lx, yy - 9), (lx + 12, yy + 1), col, -1)
            cv2.putText(img, name, (lx + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        # mini map
        mx0, my0, mw, mh = lx, by + 125, W - 12 - lx, 70
        cv2.rectangle(img, (mx0, my0), (mx0 + mw, my0 + mh), (40, 44, 52), -1)
        def mp(p):
            return int(mx0 + 4 + (p[0] + 2) / 66 * (mw - 8)), int(my0 + mh - 4 - (p[1] + 2) / 30 * (mh - 8))
        rp = [mp(p[:2]) for p in self.cmap.route]
        for a_, b_ in zip(rp[:-1], rp[1:]):
            cv2.line(img, a_, b_, (150, 150, 150), 2)
        if self.gnss_pts:
            for z in self.gnss_pts[-40:]:
                cv2.circle(img, mp(z), 1, (255, 90, 90), -1)
        cv2.circle(img, mp(st["pos"][:2]), 3, (80, 230, 255), -1)
        # telemetry
        L = self.log[-1] if self.log else {}
        soc = 100 * (1 - self.energy_J / (self.P.e.battery_wh * 3600))
        lines = [
            f"STATE  {pl.state}",
            f"       {pl.msg}",
            f"speed  {st['v']*3.6:4.1f} km/h",
            f"pod tilt  {abs(np.degrees(st['pod_pitch'])):4.1f} deg",
            f"soup tilt {L.get('soup', 0):4.1f} deg",
            f"GPS-only err {np.hypot(*(self.gnss_pts[-1]-st['pos'][:2])) if self.gnss_pts else 0:4.1f} m",
            f"fused err    {np.hypot(self.est.x-st['pos'][0], self.est.y-st['pos'][1]):4.2f} m",
            f"battery {soc:5.1f} %",
            f"t = {self.d.time:5.1f} s",
        ]
        cv2.rectangle(img, (10, 10), (292, 20 + 22 * len(lines)), (15, 18, 24), -1)
        for k, ln in enumerate(lines):
            col = (255, 255, 255) if k != 0 else tuple(int(255 * c) for c in STATE_COLORS.get(pl.state, (1, 1, 1)))
            cv2.putText(img, ln, (18, 32 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
        cv2.putText(img, "ORB  |  campus delivery sphere  |  MuJoCo physics, onboard camera perception", (14, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ summary
    def summary(self):
        import pandas as pd
        df = pd.DataFrame(self.log)
        moving = df[df.state.isin(["DRIVE", "XWALK_CROSS", "APPROACH"])]
        tr = np.array(self.trail)
        dist = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
        E = self.energy_J / 3600
        dur = float(df.t.iloc[-1])
        avg_w = self.energy_J / max(dur, 1e-6)
        usable = self.P.e.battery_wh * self.P.e.usable_frac
        out = dict(
            sim_time_s=round(dur, 1), distance_m=round(dist, 1),
            delivered=bool(self.food_taken), events=self.pl.events,
            mean_speed_moving_mps=round(float(moving.v.mean()), 2),
            max_pod_tilt_deg=round(float(df.pod_pitch.abs().max()), 2),
            p99_pod_tilt_deg=round(float(np.percentile(df.pod_pitch.abs(), 99)), 2),
            max_soup_tilt_deg=round(float(np.nanmax(df.soup)), 2),
            p99_soup_tilt_deg=round(float(np.nanpercentile(df.soup, 99)), 2),
            hop_events=int(((df.fz_g < 0.5) & (df.fz_g.shift(1) >= 0.5)).sum()),
            peak_vertical_g=round(float(df.fz_g.max()), 2),
            max_lean_deg=round(float(df.lean.abs().max()), 2),
            loc_err_mean_m=round(float(df.loc_err.mean()), 2), loc_err_max_m=round(float(df.loc_err.max()), 2),
            gnss_err_mean_m=round(float(np.mean([np.hypot(*(g - t)) for g, t in zip(self.gnss_pts, tr[::10][:len(self.gnss_pts)])])), 2) if len(self.gnss_pts) else None,
            vision_fixes=self.est.n_vis,
            energy_Wh=round(E, 2), avg_power_W=round(avg_w, 1),
            energy_split_Wh={k: round(v / 3600, 2) for k, v in self.energy_parts.items()},
            runtime_est_h_at_this_duty=round(usable / avg_w, 1),
            wh_per_km=round(E / max(dist / 1000, 1e-6), 1),
            robot_contacts_with_people_or_cars=int(self.contacts_people),
            min_clearance_m={k: round(v, 2) for k, v in self.min_clear.items() if k != "ped_bg"},
            peak_drive_torque_Nm=round(float(df.tau.abs().max()), 1),
        )
        df.to_csv(os.path.join(self.out_dir, "delivery_log.csv"), index=False)
        with open(os.path.join(self.out_dir, "delivery_summary.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
        return out
