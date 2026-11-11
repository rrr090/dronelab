"""3D cooperative cargo carrying: three quadrotors hold one load on ropes.

Physics (numpy, vectorised over a batch of B independent environments):

* Each drone is a rigid body with full 3D attitude (rotation matrix R, body
  rates w) and four motors in an X layout. Motor thrusts produce a collective
  force along the body z axis plus roll/pitch/yaw torques.
* The cargo is a point mass hanging from three ropes, one per drone. Ropes are
  elastic only in tension (spring-damper when stretched, no force when slack),
  so the load can swing, bounce and jerk the drones around.
* Wind = steady wind + an Ornstein-Uhlenbeck gust process (a shared component
  plus a smaller per-body component). The cargo catches more wind than a drone.
* Physics integrates at DT / SUBSTEPS (500 Hz); the policy acts at 50 Hz.

Action per drone (4 numbers in [-1, 1]):
    collective thrust (0 = nominal hover share), roll, pitch, yaw rate.
A built-in attitude controller (running at the physics rate) turns these
commands into motor thrusts, the way a real flight controller would.

Observation (OBS_DIM = 63, world frame, roughly unit scale):
    cargo position error (3), cargo velocity / 2 (3), then per drone:
    position relative to cargo (3), velocity / 2 (3), rotation matrix (9),
    body rates / 5 (3), rope tension / 10 (1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81
GRAVITY = np.array([0.0, 0.0, -G])
N_DRONES = 3
DRONE_MASS = 0.5
ARM = 0.15                                   # centre to motor [m]
INERTIA = np.array([0.0035, 0.0035, 0.006])  # body inertia diag [kg m^2]
MOTOR_MAX = 4.0                              # per-motor max thrust [N]
YAW_COEF = 0.016                             # rotor drag torque / thrust [m]
DT = 0.02
SUBSTEPS = 10
H = DT / SUBSTEPS
ROPE_LEN = 1.0
ROPE_K = 800.0                               # rope stiffness [N/m]
ROPE_C = 6.0                                 # rope damping [N s/m]
CARGO_MASS = 0.6
FORMATION_R = 0.55                           # horizontal drone distance from cargo
TARGET = np.array([0.0, 0.0, 1.2])           # where the cargo should be held
MAX_STEPS = 500
DRONE_DRAG = 0.08
CARGO_DRAG = 0.06
CARGO_WIND_GAIN = 1.5                        # cargo has more area than a drone
MAX_TILT_CMD = 0.6                           # rad
MAX_YAW_RATE = 2.0                           # rad/s
F_HOVER_NOM = (DRONE_MASS + CARGO_MASS / N_DRONES) * G
KR = np.array([350.0, 350.0, 120.0])         # attitude P gains (per unit inertia)
KW = np.array([30.0, 30.0, 20.0])            # body-rate D gains
MIN_DRONE_SEP = 0.25                         # closer than this = collision

OBS_PER_DRONE = 19
OBS_DIM = 6 + N_DRONES * OBS_PER_DRONE
ACT_PER_DRONE = 4
ACT_DIM = N_DRONES * ACT_PER_DRONE

_ang = np.pi / 4 + np.pi / 2 * np.arange(4)
MOTOR_XY = ARM * np.stack([np.cos(_ang), np.sin(_ang)], axis=1)   # (4, 2) body frame
SPIN = np.array([1.0, -1.0, 1.0, -1.0])
# wrench [F, tau_x, tau_y, tau_z] = MIX @ motor_thrusts
MIX = np.stack([np.ones(4), MOTOR_XY[:, 1], -MOTOR_XY[:, 0], SPIN * YAW_COEF])
UNMIX = np.linalg.inv(MIX)
PAIRS = [(0, 1), (0, 2), (1, 2)]

CRASH_REASONS = ["", "drone hit ground", "cargo hit ground", "drone flipped",
                 "drones collided", "out of bounds"]


@dataclass(frozen=True)
class Conditions:
    """Physical and sensing conditions for an episode (one test scenario)."""
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)   # steady force on a drone [N]
    gust_std: float = 0.0                                # gust force std [N]
    gust_tau: float = 0.6                                # gust correlation time [s]
    cargo_mass: float = CARGO_MASS
    rope_len: float = ROPE_LEN
    motor_gain: tuple | None = None                      # (3, 4) motor efficiencies
    obs_noise: float = 0.0
    init_noise: float = 0.2
    kick_step: int | None = None                         # step of a sudden cargo kick
    kick_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Realism:
    """Hardware effects the default simulator leaves out. All off by default, so
    existing training and evaluation are unchanged. Values that vary between
    flights are drawn once per episode from a separate random stream, so gusts
    stay the same as without them and every policy meets the same draws.

    With ``realism_random`` (training) each episode draws a strength s ~ U(0, realism_max) that
    scales motor lag, sensor noise and bias, battery sag, motor wear, the extra wind and
    the rope-stiffness spread; latency is drawn uniformly from 0..latency steps. Payload
    and rope-length spreads are not scaled, as they are already uniform draws. With
    realism_max > 1 the continuous effects go beyond the preset (latency stays 0..latency)."""
    motor_tau: float = 0.0        # motor spin-up time constant [s]; 0 = instant thrust
    latency: int = 0              # control steps (20 ms each) before an action takes effect
    sensor_noise: float = 0.0     # std of white noise on every (normalised) observation
    sensor_bias: float = 0.0      # std of a constant per-episode offset on every observation
    battery_sag: float = 0.0      # fraction of motor thrust lost per minute of flight
    motor_wear: float = 0.0       # each motor loses up to this fraction of efficiency
    wind_scale: float = 1.0       # multiplies gusts and wind-zone forces
    payload_spread: float = 0.0   # cargo mass drawn within +-this of nominal [kg]
    rope_len_spread: float = 0.0  # rope length drawn within +-this of nominal [m]
    rope_k_spread: float = 1.0    # rope stiffness drawn log-uniformly in [k/s, k*s]

    @property
    def active(self) -> bool:
        return self != Realism()


# ------------------------------------------------------------------ math helpers

def _hat(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([
        np.stack([z, -v[..., 2], v[..., 1]], -1),
        np.stack([v[..., 2], z, -v[..., 0]], -1),
        np.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def _vee(m):
    return np.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], -1)


def _rot(axis: int, a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == 0:
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, -1) for r in rows], -2)


def _expm_so3(phi):
    """Rodrigues: rotation matrix for rotation vector phi (..., 3)."""
    ang = np.linalg.norm(phi, axis=-1, keepdims=True)
    axis = phi / np.maximum(ang, 1e-12)
    K = _hat(axis)
    s, c = np.sin(ang)[..., None], np.cos(ang)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def formation_offsets(rope_len=ROPE_LEN, cargo_mass=CARGO_MASS):
    """Drone positions relative to the cargo when hanging in static equilibrium.

    Returns (3, 3). Drones sit on a circle of radius FORMATION_R, ropes are
    stretched by the static tension so the start is (nearly) at rest.
    """
    phis = np.pi / 2 + 2 * np.pi * np.arange(N_DRONES) / N_DRONES
    dist = rope_len
    for _ in range(5):
        h = np.sqrt(dist ** 2 - FORMATION_R ** 2)
        tension = cargo_mass * G * dist / (N_DRONES * h)
        dist = rope_len + tension / ROPE_K
    h = np.sqrt(dist ** 2 - FORMATION_R ** 2)
    return np.stack([FORMATION_R * np.cos(phis), FORMATION_R * np.sin(phis), np.full(3, h)], 1)


# ------------------------------------------------------------------ environment

class CargoEnv:
    def __init__(self, num_envs: int = 1, seed: int | None = None,
                 conditions: Conditions | None = None, difficulty: float | None = None,
                 autoreset: bool = False, max_steps: int = MAX_STEPS, realism: Realism | None = None,
                 realism_random: bool = False, realism_max: float = 1.0):
        """``difficulty`` (0..1) turns on domain randomisation at every reset;
        otherwise ``conditions`` are applied to all envs. ``realism_random``: each episode
        meets ``realism`` at a random strength (see Realism), for training."""
        self.B = num_envs
        self.rng = np.random.default_rng(seed)
        self.autoreset = autoreset
        self.max_steps = max_steps
        self.difficulty = difficulty
        B = num_envs
        self.p = np.zeros((B, 3, 3)); self.v = np.zeros((B, 3, 3))
        self.R = np.tile(np.eye(3), (B, 3, 1, 1)); self.w = np.zeros((B, 3, 3))
        self.cp = np.zeros((B, 3)); self.cv = np.zeros((B, 3))
        self.tension = np.zeros((B, 3))
        self.thrust = np.zeros((B, 3, 4))      # last motor thrusts [N]
        self.drone_force = np.zeros((B, 3, 3))  # last external (wind) force on drones
        self.cargo_force = np.zeros((B, 3))
        self.gust_common = np.zeros((B, 3)); self.gust_body = np.zeros((B, 4, 3))
        self.t = np.zeros(B, dtype=np.int64)
        self.last_action = np.zeros((B, ACT_DIM))
        self.crash_code = np.zeros(B, dtype=np.int64)
        self.target = np.tile(TARGET, (B, 1))       # where the cargo should be (can move)
        self.tension_peak = np.zeros((B, 3))         # max rope tension within the last step
        self.set_conditions(conditions or Conditions())
        self.rope_k = np.full(B, ROPE_K)
        self.wear = np.ones((B, 3, 4))
        self.motor_f = np.zeros((B, 3, 4))      # actual motor thrusts when motors lag [N]
        self.realism = realism or Realism()
        self.real_rng = np.random.default_rng(None if seed is None else [seed, 7919])
        self.act_buf = np.zeros((B, self.realism.latency, N_DRONES, ACT_PER_DRONE))
        self.realism_random, self.realism_max = realism_random, realism_max
        self.real_s = np.ones(B)                                 # per-episode strength of the effects
        self.latency = np.full(B, self.realism.latency)          # per-episode delay [steps]

    # ------------------------------------------------------------ conditions
    def set_conditions(self, c: Conditions, idx=None):
        idx = np.arange(self.B) if idx is None else idx
        if not hasattr(self, "wind"):
            B = self.B
            self.wind = np.zeros((B, 3)); self.gust_std = np.zeros(B); self.gust_tau = np.ones(B)
            self.cargo_mass = np.zeros(B); self.rope_len = np.zeros(B)
            self.motor_gain = np.ones((B, 3, 4)); self.obs_noise = np.zeros(B)
            self.init_noise = np.zeros(B); self.kick_step = np.full(B, -1); self.kick_vel = np.zeros((B, 3))
        self.wind[idx] = c.wind; self.gust_std[idx] = c.gust_std; self.gust_tau[idx] = c.gust_tau
        self.cargo_mass[idx] = c.cargo_mass; self.rope_len[idx] = c.rope_len
        self.motor_gain[idx] = 1.0 if c.motor_gain is None else np.asarray(c.motor_gain)
        self.obs_noise[idx] = c.obs_noise; self.init_noise[idx] = c.init_noise
        self.kick_step[idx] = -1 if c.kick_step is None else c.kick_step
        self.kick_vel[idx] = c.kick_vel

    def _randomize(self, idx):
        """Domain randomisation, scaled by self.difficulty in [0, 1]."""
        d, n, r = self.difficulty, len(idx), self.rng
        heading = r.uniform(0, 2 * np.pi, n)
        speed = r.uniform(0, 0.8 * d, n)
        self.wind[idx] = np.stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros(n)], 1)
        self.gust_std[idx] = r.uniform(0, 0.8 * d, n)
        self.gust_tau[idx] = r.uniform(0.3, 1.0, n)
        self.cargo_mass[idx] = r.uniform(CARGO_MASS - 0.3 * d, CARGO_MASS + 0.3 * d, n)
        self.rope_len[idx] = r.uniform(ROPE_LEN - 0.15 * d, ROPE_LEN + 0.15 * d, n)
        self.motor_gain[idx] = r.uniform(1 - 0.2 * d, 1.0, (n, 3, 4))
        self.obs_noise[idx] = r.uniform(0, 0.02 * d, n)
        self.init_noise[idx] = 0.1 + 0.2 * d
        kick = r.random(n) < 0.4 * d
        self.kick_step[idx] = np.where(kick, r.integers(25, MAX_STEPS - 50, n), -1)
        kv = r.normal(size=(n, 3)) * np.array([1, 1, 0.3])
        kv /= np.linalg.norm(kv, axis=1, keepdims=True) + 1e-9
        self.kick_vel[idx] = kv * r.uniform(0.3, 1.5, (n, 1))

    # ------------------------------------------------------------ reset
    def reset(self, idx=None) -> np.ndarray:
        idx = np.arange(self.B) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return self.observe()
        if self.difficulty is not None:
            self._randomize(idx)
        if self.realism.active:
            self._draw_realism(idx)
        r, n = self.rng, len(idx)
        noise = self.init_noise[idx][:, None]
        for j, i in enumerate(idx):
            off = formation_offsets(self.rope_len[i], self.cargo_mass[i])
            self.cp[i] = self.target[i] + r.uniform(-1, 1, 3) * noise[j] * np.array([1, 1, 0.5])
            self.p[i] = self.cp[i] + off + r.uniform(-1, 1, (3, 3)) * 0.15 * noise[j]
        self.v[idx] = r.normal(0, 0.3, (n, 3, 3)) * noise[:, :, None]
        self.cv[idx] = r.normal(0, 0.3, (n, 3)) * noise
        yaw = r.uniform(-1, 1, (n, 3)) * 1.5 * noise
        tilt_x = r.uniform(-1, 1, (n, 3)) * 0.3 * noise
        tilt_y = r.uniform(-1, 1, (n, 3)) * 0.3 * noise
        self.R[idx] = _rot(2, yaw) @ _rot(1, tilt_y) @ _rot(0, tilt_x)
        self.w[idx] = 0.0
        self.gust_common[idx] = 0.0; self.gust_body[idx] = 0.0
        self.t[idx] = 0
        self.last_action[idx] = 0.0
        self.crash_code[idx] = 0
        self.act_buf[idx] = 0.0
        self.motor_f[idx] = ((DRONE_MASS + self.cargo_mass[idx] / N_DRONES) * G / 4)[:, None, None]
        self._ropes()
        return self.observe()

    def _draw_realism(self, idx):
        """Per-episode hardware variation (see Realism)."""
        c, n, r = self.realism, len(idx), self.real_rng
        if self.realism_random:
            self.real_s[idx] = r.uniform(0, self.realism_max, n)
            self.latency[idx] = np.minimum(np.floor(self.real_s[idx] * (c.latency + 1)), c.latency)
        s = self.real_s[idx]
        mass, rope = r.uniform(-1, 1, n), r.uniform(-1, 1, n)   # drawn always, so the stream is fixed
        if c.payload_spread > 0:          # otherwise keep the mass set by conditions or training
            self.cargo_mass[idx] = CARGO_MASS + mass * c.payload_spread
        if c.rope_len_spread > 0:
            self.rope_len[idx] = ROPE_LEN + rope * c.rope_len_spread
        self.rope_k[idx] = ROPE_K * np.exp(r.uniform(-1, 1, n) * np.log(c.rope_k_spread) * s)
        self.wear[idx] = 1.0 - r.uniform(0, c.motor_wear * s[:, None, None], (n, 3, 4))

    @property
    def wind_factor(self):
        """Per-env multiplier on gusts and wind-zone forces, shape (B,)."""
        return 1.0 + (self.realism.wind_scale - 1.0) * self.real_s

    # ------------------------------------------------------------ physics
    def _ropes(self):
        d = self.p - self.cp[:, None]
        dist = np.linalg.norm(d, axis=-1)
        u = d / np.maximum(dist, 1e-9)[..., None]
        ext = dist - self.rope_len[:, None]
        rate = np.sum((self.v - self.cv[:, None]) * u, -1)
        self.tension = np.where(ext > 0, np.maximum(self.rope_k[:, None] * ext + ROPE_C * rate, 0.0), 0.0)
        return u

    def _substep(self, F_cmd, Rd, yaw_rate_cmd):
        # Attitude controller (geometric, on SO(3)).
        R, w = self.R, self.w
        eR = 0.5 * _vee(np.swapaxes(Rd, -1, -2) @ R - np.swapaxes(R, -1, -2) @ Rd)
        w_des = np.zeros_like(w); w_des[..., 2] = yaw_rate_cmd
        gyro = np.cross(w, INERTIA * w)
        tau = INERTIA * (-KR * eR - KW * (w - w_des)) + gyro
        wrench = np.concatenate([F_cmd[..., None], tau], -1)
        f = np.clip(wrench @ UNMIX.T, 0.0, MOTOR_MAX) * self._gain
        if self.realism.motor_tau > 0:
            tau = self.realism.motor_tau * self.real_s
            alpha = np.where(tau > 0, np.minimum(H / np.maximum(tau, 1e-12), 1.0), 1.0)
            self.motor_f += (f - self.motor_f) * alpha[:, None, None]
            f = self.motor_f
        self.thrust = f
        actual = f @ MIX.T
        F, tau = actual[..., 0], actual[..., 1:]

        u = self._ropes()
        T = self.tension[..., None]
        acc = (R[..., :, 2] * F[..., None] - T * u + self.drone_force
               - DRONE_DRAG * self.v) / DRONE_MASS + GRAVITY
        cacc = (np.sum(T * u, 1) + self.cargo_force - CARGO_DRAG * self.cv) \
            / self.cargo_mass[:, None] + GRAVITY
        wdot = (tau - np.cross(w, INERTIA * w)) / INERTIA

        self.v = self.v + acc * H
        self.p = self.p + self.v * H
        self.cv = self.cv + cacc * H
        self.cp = self.cp + self.cv * H
        self.w = w + wdot * H
        self.R = R @ _expm_so3(self.w * H)

    def _zone_wind(self, bodies: np.ndarray) -> np.ndarray:
        """Extra position-dependent wind force on bodies (B, 4, 3). Hook for maps."""
        return np.zeros_like(bodies)

    def _advance(self, action: np.ndarray):
        """Apply one 50 Hz control step of physics. Returns (clipped action, kicked mask)."""
        B = self.B
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(B, N_DRONES, ACT_PER_DRONE), -1, 1)
        c = self.realism
        ap = a
        if c.latency > 0:                     # the action that takes effect now was sent `latency` steps ago
            j = np.minimum(c.latency - self.latency, c.latency - 1)   # slot holding the action from `latency` steps ago
            ap = np.where((self.latency > 0)[:, None, None], self.act_buf[np.arange(B), j], a)
            self.act_buf[:, :-1] = self.act_buf[:, 1:]
            self.act_buf[:, -1] = a
        self._gain = self.motor_gain
        if c.motor_wear > 0 or c.battery_sag > 0:
            sag = np.maximum(1.0 - c.battery_sag * self.real_s * self.t * DT / 60.0, 0.0)
            self._gain = self.motor_gain * self.wear * sag[:, None, None]
        F_cmd = F_HOVER_NOM * (1.0 + ap[..., 0])
        Rd = _rot(2, yaw_of(self.R)) @ _rot(1, MAX_TILT_CMD * ap[..., 2]) @ _rot(0, MAX_TILT_CMD * ap[..., 1])
        yaw_rate_cmd = MAX_YAW_RATE * ap[..., 3]

        # Gusts: OU process, shared + per-body, weaker vertically.
        decay = np.exp(-DT / self.gust_tau)
        kick = self.gust_std * self.wind_factor * np.sqrt(1 - decay ** 2)
        self.gust_common = decay[:, None] * self.gust_common + kick[:, None] * self.rng.normal(size=(B, 3))
        self.gust_body = decay[:, None, None] * self.gust_body + kick[:, None, None] * self.rng.normal(size=(B, 4, 3))
        g = (self.gust_common[:, None] + 0.35 * self.gust_body) * np.array([1, 1, 0.3])
        bodies = np.concatenate([self.p, self.cp[:, None]], 1)       # (B, 4, 3)
        ext = self.wind[:, None] + g + self._zone_wind(bodies)
        self.drone_force = ext[:, :3]
        self.cargo_force = CARGO_WIND_GAIN * ext[:, 3]

        kicked = self.t == self.kick_step
        if kicked.any():
            self.cv[kicked] += self.kick_vel[kicked]

        self.tension_peak[:] = 0.0
        for _ in range(SUBSTEPS):
            self._substep(F_cmd, Rd, yaw_rate_cmd)
            np.maximum(self.tension_peak, self.tension, out=self.tension_peak)
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        self.t += 1
        return a, kicked

    def step(self, action: np.ndarray):
        B = self.B
        a, kicked = self._advance(action)
        reward, crash = self._reward_and_crash(a.reshape(B, ACT_DIM))
        self.last_action = a.reshape(B, ACT_DIM)
        self.crash_code = crash
        terminated = crash > 0
        truncated = (self.t >= self.max_steps) & ~terminated
        reward = reward - 10.0 * terminated
        info = {"crash_code": crash.copy(), "kicked": kicked}
        if self.autoreset:
            done = np.nonzero(terminated | truncated)[0]
            if len(done):
                self.reset(done)
        return self.observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------ reward / obs
    def cargo_error(self):
        return np.linalg.norm(self.cp - self.target, axis=-1)

    def tilt(self):
        return np.arccos(np.clip(self.R[..., 2, 2], -1, 1))

    def _reward_and_crash(self, a):
        cerr = self.cargo_error()
        tilt = self.tilt()
        sep = np.stack([np.linalg.norm(self.p[:, i] - self.p[:, j], axis=-1) for i, j in PAIRS], 1)
        reward = (1.0 - cerr - 0.1 * np.sum(self.cv ** 2, -1)
                  - 0.3 * np.mean(tilt ** 2, -1) - 0.005 * np.mean(np.sum(self.w ** 2, -1), -1)
                  - 2.0 * np.sum(np.maximum(0.5 - sep, 0), -1)
                  - 0.02 * np.mean(a ** 2, -1)
                  - 0.05 * np.mean((a - self.last_action) ** 2, -1))
        crash = np.zeros(self.B, dtype=np.int64)
        crash[np.any(np.abs(self.p[..., :2]) > 4, -1).any(-1) | np.any(self.p[..., 2] > 5, -1)
              | (cerr > 3)] = 5
        crash[sep.min(-1) < MIN_DRONE_SEP] = 4
        crash[(tilt > np.radians(80)).any(-1)] = 3
        crash[self.cp[:, 2] < 0.02] = 2
        crash[(self.p[..., 2] < 0.05).any(-1)] = 1
        return reward, crash

    def observe(self) -> np.ndarray:
        B = self.B
        rel = self.p - self.cp[:, None]
        drone = np.concatenate([
            rel, self.v / 2, self.R.reshape(B, 3, 9), self.w / 5, self.tension[..., None] / 10,
        ], -1)
        obs = np.concatenate([self.cp - self.target, self.cv / 2, drone.reshape(B, -1)], -1)
        if np.any(self.obs_noise > 0):
            obs = obs + self.rng.normal(size=obs.shape) * self.obs_noise[:, None]
        return obs.astype(np.float32)
