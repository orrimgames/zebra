"""Independent safety controller (a small MCU, separate from the Jetson).

It reads only its own sensors -- four near-field ToF rangers in the cap
fronts, the IMU, the band encoder -- and can cut drive torque and command a
full-pendulum emergency stop no matter what the main computer is doing:

  * something inside the stopping envelope:  d < v^2 / 2a + 0.25 m + 0.12 s * v
  * main computer silent for > 0.35 s (planner heartbeat watchdog)
  * tipped / lifted: lean > 30 deg or pitch > 35 deg

It latches until the path has been clear for 0.8 s and the robot is stopped.
"""
import numpy as np
import mujoco


class SafetyMCU:
    RANGE = 3.0
    A_STOP = 2.3

    def __init__(self, m, rb):
        self.m, self.rb = m, rb
        self.sites = [m.site(f"tof_{k}").id for k in range(4)]
        # ray yaw (deg) per ToF: two straight ahead, two angled out 25 deg
        self.yaw = np.radians([0.0, 0.0, 25.0, -25.0])
        self.group = np.array([1, 0, 1, 0, 1, 1], np.uint8)   # skip robot visuals (1) + shell contact (3)
        self.gid = np.zeros(1, np.int32)
        self.trig, self.reason = False, ""
        self.clear_t = 0.0
        self.tof = np.full(4, self.RANGE)
        self.events = []

    def ranges(self, d):
        Rb = d.xmat[self.rb.bid_body].reshape(3, 3)
        fwd = Rb[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.array([-fwd[1], fwd[0], 0.0])
        for k, sid in enumerate(self.sites):
            v = np.cos(self.yaw[k]) * fwd + np.sin(self.yaw[k]) * left
            p = d.site_xpos[sid].copy()
            dist = mujoco.mj_ray(self.m, d, p, v, self.group, 1, self.rb.bid_shell, self.gid)
            if dist < 0 or self.m.geom_bodyid[self.gid[0]] in self.rb.body_ids:
                dist = self.RANGE
            self.tof[k] = min(dist, self.RANGE)
        return self.tof

    def update(self, d, t, v, lean, pitch, heartbeat_t, legs_down=False):
        tof = self.ranges(d)
        need = v * v / (2 * self.A_STOP) + 0.25 + 0.12 * abs(v)
        why = ""
        if v > 0.15 and not legs_down and tof.min() < need:
            why = f"ToF {tof.min():.2f} m"
        if t - heartbeat_t > 0.35:
            why = "main computer silent"
        if abs(lean) > np.radians(30) or abs(pitch) > np.radians(35):
            why = "tilted/lifted"
        if why:
            if not self.trig:
                self.events.append((round(t, 2), why))
            self.trig, self.reason, self.clear_t = True, why, 0.0
        elif self.trig:
            self.clear_t += 0.004
            if self.clear_t > 0.8 and abs(v) < 0.05:
                self.trig, self.reason = False, ""
        return self.trig
