"""Test rig: one robot in a small world, run exactly the way the real robot
runs (250 Hz control on delayed/noisy sensors, wind loads, leg servo).

Used by the stress tests and calibration scripts; the campus sim (sim.py)
uses the same pieces.
"""
import copy
import numpy as np
import mujoco
from . import model
from .params import Params
from .control import Robot, LowLevel, SensorModel, Legs
from . import wind as windmod


class Rig:
    def __init__(self, P_true=None, P_nom=None, extra="", floor_euler=None, wind=None, seed=0,
                 sensor_scale=1.0, pos=(0, 0), yaw=0.0, extra_bodies="", realistic=True):
        self.P = P_true or Params()
        self.Pn = P_nom or self.P
        xml = model.standalone_xml(self.P, floor_extra=extra, extra_bodies=extra_bodies, floor_euler=floor_euler, pos=pos, yaw=yaw)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.rb = Robot(self.m, self.P)
        self.ll = LowLevel(self.Pn)
        self.sen = SensorModel(self.P, seed=seed, scale=sensor_scale)
        self.legs = Legs(self.Pn, self.rb)
        self.wind = wind
        self.realistic = realistic
        self.n_ctrl = max(1, int(round(self.Pn.control_dt / self.m.opt.timestep))) if realistic else 1
        self.lid_cmd = 0.0
        self.steer_override = None
        self.drive_override = None
        self.wind_force = np.zeros(2)
        mujoco.mj_forward(self.m, self.d)
        self.eq_band = self.m.equality("brake_band").id
        self.eq_yoke = self.m.equality("brake_yoke").id
        self.eq_lock = self.m.equality("lock_level").id

    def lock_level(self, on):
        m, d, rb = self.m, self.d, self.rb
        if on and not d.eq_active[self.eq_lock]:
            m.eq_data[self.eq_lock][0] = d.qpos[rb.qa["pod_pitch"]] - d.qpos[rb.qa["drive"]]
        d.eq_active[self.eq_lock] = int(on)

    def brake(self, on):
        m, d, rb = self.m, self.d, self.rb
        if on and not d.eq_active[self.eq_band]:
            m.eq_data[self.eq_band][0] = d.qpos[rb.qa["pod_pitch"]]
            m.eq_data[self.eq_yoke][0] = d.qpos[rb.qa["drive"]]
        d.eq_active[self.eq_band] = int(on); d.eq_active[self.eq_yoke] = int(on)
        self.legs.lock_wheels(m, on)

    def run(self, T, v_fn, yaw_fn=None, cb=None, every=None, parked_fn=None):
        """v_fn(t, s) -> desired speed; yaw_fn(t, s) -> desired yaw rate;
        cb(t, s_true) called every control tick (return False to stop)."""
        m, d, rb, ll = self.m, self.d, self.rb, self.ll
        dt = m.opt.timestep
        cdt = dt * self.n_ctrl
        steps = int(T / dt)
        for i in range(steps):
            if self.wind is not None:
                w = self.wind.at(d.time)
                self.wind_force = windmod.apply(m, d, self.P, w, rb.bid_shell, rb.bid_lid, d.qpos[rb.qa["lid"]])
            if i % self.n_ctrl == 0:
                st = rb.state(d)
                s = self.sen.measure(st) if self.realistic else st
                parked = parked_fn(d.time, st) if parked_fn else False
                if parked:
                    d.ctrl[rb.act["drive"]] = 0.0
                    ll.reset()
                else:
                    vd = v_fn(d.time, s)
                    self.lock_level("frame" in vd if isinstance(vd, dict) else False)
                    if isinstance(vd, dict) and "frame" in vd:   # leg modes: drive motor levels the locked frame
                        c = ll.hold_frame(s)
                    elif isinstance(vd, dict):         # {'hold': th_ref} -> pendulum position hold
                        c = ll.hold(s, vd.get("hold", 0.0))
                    else:
                        c, _ = ll.speed(s, vd, cdt)
                    d.ctrl[rb.act["drive"]] = c if self.drive_override is None else self.drive_override(d.time, s, c)
                    sd = ll.steer(s, yaw_fn(d.time, s) if yaw_fn else 0.0, cdt)
                    d.ctrl[rb.act["steer"]] = sd if self.steer_override is None else self.steer_override(d.time, s)
                d.ctrl[rb.act["pod_level"]] = ll.level(s) if not (parked or d.eq_active[self.eq_lock]) else 0.0
                d.ctrl[rb.act["lid"]] = self.lid_cmd
                if cb is not None and cb(d.time, st) is False:
                    break
            self.legs.step(d, dt)
            mujoco.mj_step(m, d)
        return self


def perturbed(rng, P0=None, level=1.0):
    """A randomly built robot: what the factory + the campus might really give us."""
    P = copy.deepcopy(P0 or Params())
    u = lambda a: 1.0 + level * rng.uniform(-a, a)
    P.m.band *= u(0.10); P.m.caps *= u(0.10); P.m.yoke *= u(0.10); P.m.battery *= u(0.05)
    P.m.payload = rng.uniform(0.0, 7.0) if level > 0 else P.m.payload
    P.m.yoke_cg_r *= u(0.08); P.geo.tray_r *= u(0.04)
    P.m.payload_cg_z += level * rng.uniform(-0.03, 0.03)
    P.c.slide = float(np.clip(P.c.slide * u(0.35), 0.45, 1.2))       # wet concrete .. dry rubber
    P.c.torsion *= u(0.4); P.c.roll *= u(0.6)
    P.d.motor_peak_torque *= u(0.12)                                  # Kt spread + thermal derating
    P.d.motor_lag *= u(0.6)
    P.d.steer_torque *= u(0.12); P.d.steer_kp *= u(0.2)
    P.d.body_damping *= u(0.5); P.d.yoke_damping *= u(0.5)
    P.sensor_delay = 0.004 * (1 + level * rng.uniform(0, 3.0))       # 4 .. 16 ms
    return P
