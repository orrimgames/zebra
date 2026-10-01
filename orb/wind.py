"""Wind: mean flow + turbulence + discrete gusts, and the aerodynamic loads
it puts on the robot (sphere drag through the centre, plus the open hatch
lid acting as a small sail).

Turbulence is a first-order Gauss-Markov process per axis with the time
scale of a ~25 m eddy (a Dryden-like spectrum without the extra pole), and
gusts are the classic "1 - cosine" shape layered on top. All seeded.
"""
import numpy as np


class Wind:
    def __init__(self, mean=0.0, direction=0.0, turb=0.15, gusts=(), seed=0, L=25.0):
        """mean [m/s], direction [rad, world frame, the way the air moves],
        turb = turbulence intensity (sigma / mean), gusts = [(t0, duration, peak_extra_mps, dir_offset_rad)]."""
        self.U, self.dir, self.ti, self.L = mean, direction, turb, L
        self.gusts = list(gusts)
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(2)          # turbulence state (along, across)
        self.t_last = 0.0
        self.last = np.zeros(2)

    def at(self, t):
        dt = max(1e-4, t - self.t_last)
        self.t_last = t
        U = max(self.U, 0.5)
        tau = self.L / U
        a = np.exp(-dt / tau)
        self.x = a * self.x + np.sqrt(1 - a * a) * self.rng.normal(0, 1, 2)
        sig = self.ti * self.U
        along = self.U + sig * self.x[0]
        across = 0.6 * sig * self.x[1]
        for (t0, dur, peak, doff) in self.gusts:
            if t0 <= t <= t0 + dur:
                g = 0.5 * peak * (1 - np.cos(2 * np.pi * (t - t0) / dur))
                along += g * np.cos(doff); across += g * np.sin(doff)
        c, s = np.cos(self.dir), np.sin(self.dir)
        w = np.array([c * along - s * across, s * along + c * across])
        self.last = w
        return w


def apply(m, d, P, wind_xy, shell_id, lid_id=None, lid_q=0.0):
    """Write the aerodynamic forces into d.xfrc_applied. Returns the drag force (world xy)."""
    import mujoco
    a = P.a
    vel = np.zeros(6)                         # linear velocity of the shell centre
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, shell_id, vel, 0)
    u = np.array([wind_xy[0] - vel[3], wind_xy[1] - vel[4], 0.0])
    A = np.pi * P.geo.R ** 2
    F = 0.5 * a.rho * a.cd_sphere * A * np.linalg.norm(u) * u
    d.xfrc_applied[shell_id, :3] = F
    if lid_id is not None:
        d.xfrc_applied[lid_id, :] = 0.0
        if lid_q < -0.25:
            n = d.xmat[lid_id].reshape(3, 3)[:, 2]
            un = float(u @ n)
            Fl = 0.5 * a.rho * a.cd_lid * a.lid_area * un * abs(un) * n * min(1.0, abs(lid_q) / 1.2)
            d.xfrc_applied[lid_id, :3] = Fl
    return F[:2]
