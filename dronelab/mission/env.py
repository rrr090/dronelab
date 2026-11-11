"""Cargo delivery missions: fly the 3-drone team through a generated map.

Built on CargoEnv's physics. Each episode:

1. A map is generated (or given) and a route with formation clearance is
   planned from the start pad to the goal pad.
2. A *reference point* moves along the route at a safe speed (slower in
   turns, braking before the goal). It never runs more than LEAD metres
   ahead of the cargo, so if the team falls behind, the reference waits.
3. The policy flies the team so the cargo follows the reference, using
   obstacle sensing to keep clear of trees, walls and buildings, and riding
   out wind zones and gusts.
4. Cargo health starts at 100 and drops with rope jerks, high g-forces,
   cargo impacts and ropes rubbing on obstacles. The mission is delivered
   when the cargo is held steady over the goal pad.

Score = cargo health if delivered, else 0.

Observation (MISSION_OBS_DIM = 93): the 63 CargoEnv features (with the cargo
error measured from the moving reference), plus reference velocity (3),
two lookahead route points relative to the cargo (6), for each drone and the
cargo the obstacle distance and direction (16), obstacle distance at each
rope midpoint (3), cargo health (1) and route distance remaining (1).

With ``history=True``, HISTORY_DIM more inputs follow: the peak rope tensions
within the last step (3), which is what causes damage, then rope tension, cargo
velocity and cargo offset from the team centre as they were HISTORY_LAGS steps ago.
"""
from __future__ import annotations

import numpy as np

from dronelab.mission import maps as M
from dronelab.cargo.env import (
    CARGO_MASS, DT, G, OBS_DIM, PAIRS, CargoEnv, Conditions, Realism,
)
from dronelab.cargo.testing import FormationController

MISSION_OBS_DIM = OBS_DIM + 30
HISTORY_LAGS = (5, 15, 30)   # steps back: 0.1, 0.3 and 0.6 s, against a swing period near 2 s
HISTORY_FEATURES = 9
HISTORY_DIM = 3 + HISTORY_FEATURES * len(HISTORY_LAGS)
JERK_WARNING = 1.5         # peak tension / T0 where the optional early penalty starts; damage starts at 2
PRIV_DIM = 63              # size of MissionEnv.privileged()
SPIKE_CAP = 3.0            # the optional spike penalty stops growing this far above the damage level (in T0)
LEAD = 1.0                 # reference may run at most this far ahead of the cargo [m]
A_LAT = 0.6                # max lateral accel in turns for the reference [m/s^2]
A_DEC = 0.35               # braking before the goal [m/s^2]
DRONE_R = 0.22             # drone collision radius [m]
CARGO_R = 0.12
DELIVER_RADIUS = 0.35
DELIVER_SPEED = 0.35
DELIVER_HOLD = 25          # steps the cargo must be held steady over the goal
PMAX = 900                 # max route points (0.25 m spacing)
NC_LOCAL, NB_LOCAL = 32, 16   # nearest obstacles kept for fast collision/sensing
LOCAL_REFRESH = 10            # steps between nearest-obstacle refreshes
FAIL_REASONS = ["", "drone hit ground", "cargo hit ground", "drone flipped", "drones collided",
                "out of bounds", "drone hit obstacle", "cargo destroyed"]
DAMAGE_KINDS = ["rope jerk", "g-force", "impact", "rope rub"]

_BASELINE = FormationController()


def baseline_actions(obs: np.ndarray) -> np.ndarray:
    """Formation baseline on mission observations (N, 93) -> (N, 12): follows the
    moving route reference with velocity feedforward; ignores obstacles."""
    return _BASELINE.act_batch(obs[:, :OBS_DIM], 2 * obs[:, OBS_DIM:OBS_DIM + 3])


def compose_residual(obs: np.ndarray, correction: np.ndarray, scale: float) -> np.ndarray:
    """Residual control: baseline action plus a bounded learned correction."""
    return np.clip(baseline_actions(obs) + scale * np.clip(correction, -1, 1), -1, 1)


def _pad(arr, n, width, far):
    out = np.zeros((n, width))
    out[:, 0:2] = far
    k = min(len(arr), n)
    if k:
        out[:k] = arr[:k]
    return out


def speed_profile(path, cruise):
    """Max reference speed along a route: cruise, slower in turns, brake to stop at goal."""
    d = np.diff(path[:, :2], axis=0)
    seg = np.linalg.norm(d, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    curv = np.abs(np.gradient(head)) / np.maximum(seg, 1e-6)
    curv = np.convolve(np.concatenate([curv, curv[-1:]]), np.ones(9) / 9, mode="same")
    v = np.minimum(cruise, np.sqrt(A_LAT / np.maximum(curv, 1e-6)))
    v = np.minimum(v, np.sqrt(2 * A_DEC * np.maximum(s[-1] - s, 0)) + 0.02)
    return s, v


class MissionEnv(CargoEnv):
    def __init__(self, num_envs: int = 1, seed: int | None = None, difficulty: float | None = None,
                 themes=None, maps=None, autoreset: bool = False, cruise: float | None = None,
                 conditions: Conditions | None = None, pool_size: int = 40, residual: float = 0.0,
                 history: bool = False, jerk_weight: float = 0.0, spike_weight: float = 0.0,
                 realism: Realism | None = None, realism_random: bool = False, realism_max: float = 1.0):
        """Training: pass ``difficulty`` (0..1) to draw random maps and conditions.
        Evaluation: pass ``maps`` (one MissionMap per env) and optionally ``conditions``.
        ``residual > 0``: actions are corrections added to the formation baseline,
        scaled by ``residual`` (see ``compose_residual``).
        ``history``: add the HISTORY_DIM inputs described at the top of this file.
        ``jerk_weight``: penalise rope tension above JERK_WARNING, before it does damage.
        ``spike_weight``: quadratic penalty on tension above the damage level, capped at SPIKE_CAP,
        because a few large peaks cause most of the rope damage.
        ``realism``: hardware effects left out by default (see cargo.env.Realism). Sensor noise
        and bias reach the whole observation, so the baseline inside a residual policy sees them too.
        ``realism_random``: each episode meets ``realism`` at a random strength in
        [0, realism_max] (training)."""
        self.residual = residual
        self.history, self.jerk_weight, self.spike_weight = history, jerk_weight, spike_weight
        self.past = np.zeros((num_envs, max(HISTORY_LAGS) + 1, HISTORY_FEATURES))
        self.past_i = 0
        self._obs = None
        self.fixed_maps = maps
        self.themes = themes or M.THEMES
        self.cruise_fixed = cruise
        self.pool, self.pool_size = {}, pool_size
        B = num_envs
        self.cyl = np.zeros((B, M.NC_MAX, 4)); self.box = np.zeros((B, M.NB_MAX, 6))
        self.zones = np.zeros((B, M.NZ_MAX, 8))
        self.path = np.zeros((B, PMAX, 3)); self.s_path = np.zeros((B, PMAX))
        self.v_path = np.zeros((B, PMAX)); self.n_path = np.zeros(B, dtype=np.int64)
        self.path_len = np.zeros(B); self.length = np.zeros(B); self.goal = np.zeros((B, 3))
        self.s_ref = np.zeros(B); self.v_ref = np.zeros((B, 3)); self.proj_idx = np.zeros(B, dtype=np.int64)
        self.s_proj = np.zeros(B); self.health = np.full(B, 100.0); self.hold = np.zeros(B, dtype=np.int64)
        self.time_limit = np.zeros(B, dtype=np.int64); self.prev_cv = np.zeros((B, 3))
        self.contact = np.zeros(B, dtype=bool); self.damage = np.zeros((B, 4))
        self.zone_ou = np.zeros((B, 4, 3)); self.map_info = [None] * B
        self.step_damage = np.zeros(B); self.impact_event = np.zeros(B, dtype=bool)
        self.cyl_l = np.zeros((B, NC_LOCAL, 4)); self.box_l = np.zeros((B, NB_LOCAL, 6))
        self._cache = None
        self.sensor_bias = np.zeros((B, MISSION_OBS_DIM + (HISTORY_DIM if history else 0)))
        super().__init__(num_envs, seed=seed, conditions=conditions or Conditions(init_noise=0.05),
                         difficulty=difficulty, autoreset=autoreset, max_steps=10 ** 9, realism=realism,
                         realism_random=realism_random, realism_max=realism_max)

    # ------------------------------------------------------------ maps
    def _draw_map(self):
        d = self.difficulty
        theme = self.themes[self.rng.integers(len(self.themes))]
        bucket = (theme, int(round(d * 4)))
        lst = self.pool.setdefault(bucket, [])
        if len(lst) < self.pool_size:
            lvl = bucket[1] / 4
            m = M.generate_map(int(self.rng.integers(0, 1_000_000)), theme,
                               float(np.clip(self.rng.uniform(lvl - 0.25, lvl), 0, 1)))
            lst.append(m)
            return m
        return lst[self.rng.integers(len(lst))]

    def _load_map(self, i, m: M.MissionMap):
        self.map_info[i] = m
        self.cyl[i] = _pad(m.cylinders, M.NC_MAX, 4, M.PAD_FAR); self.cyl[i, len(m.cylinders):, 3] = -10
        self.box[i] = _pad(m.boxes, M.NB_MAX, 6, M.PAD_FAR)
        self.zones[i] = _pad(m.zones, M.NZ_MAX, 8, M.PAD_FAR)
        self.zones[i, len(m.zones):, 3] = 1e-6
        cruise = self.cruise_fixed or float(self.rng.uniform(0.7, 1.4))
        s, v = speed_profile(m.path, cruise)
        n = min(len(m.path), PMAX)
        self.path[i, :n] = m.path[:n]; self.path[i, n:] = m.path[n - 1]
        self.s_path[i, :n] = s[:n]; self.s_path[i, n:] = s[n - 1] + 1e-3 * np.arange(1, PMAX - n + 1)
        self.v_path[i, :n] = v[:n]; self.v_path[i, n:] = 0
        self.n_path[i] = n; self.path_len[i] = s[n - 1]
        self.length[i] = m.length; self.goal[i] = m.path[n - 1]
        self.time_limit[i] = int((self.path_len[i] / 0.3 + 20) / DT)

    def _randomize(self, idx):
        d, n, r = self.difficulty, len(idx), self.rng
        self.wind[idx] = 0.0
        self.cargo_mass[idx] = r.uniform(CARGO_MASS - 0.2 * d, CARGO_MASS + 0.2 * d, n)
        self.rope_len[idx] = r.uniform(1 - 0.1 * d, 1 + 0.1 * d, n)
        self.motor_gain[idx] = r.uniform(1 - 0.15 * d, 1.0, (n, 3, 4))
        self.obs_noise[idx] = r.uniform(0, 0.02 * d, n)
        self.init_noise[idx] = 0.05 + 0.1 * d
        self.kick_step[idx] = -1
        self.gust_tau[idx] = r.uniform(0.3, 1.0, n)

    # ------------------------------------------------------------ reset
    def reset(self, idx=None):
        idx = np.arange(self.B) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return self.observe()
        for i in idx:
            m = self.fixed_maps[i] if self.fixed_maps is not None else self._draw_map()
            self._load_map(i, m)
            self.target[i] = m.path[0]
        if self.realism.sensor_bias > 0:
            self.sensor_bias[idx] = self.real_rng.normal(size=(len(idx), self.sensor_bias.shape[1]))                 * self.realism.sensor_bias
        super().reset(idx)
        for i in idx:
            self.gust_std[i] = self.map_info[i].gust_std
        self.s_ref[idx] = 0; self.v_ref[idx] = 0; self.proj_idx[idx] = 0; self.s_proj[idx] = 0
        self.health[idx] = 100.0; self.hold[idx] = 0; self.prev_cv[idx] = self.cv[idx]
        self.contact[idx] = False; self.damage[idx] = 0; self.zone_ou[idx] = 0
        self.step_damage[idx] = 0; self.impact_event[idx] = False
        self._refresh_local(idx)
        self._cache = None
        self.tension_peak[idx] = self.tension[idx]
        self.past[idx] = self._history_features()[idx, None]
        return self.observe()

    def _refresh_local(self, idx=None):
        """Keep only the obstacles nearest to each team for per-step geometry."""
        idx = np.arange(self.B) if idx is None else np.asarray(idx)
        c = self.cp[idx, None, :]
        dc = M.sdf_cylinders(c, self.cyl[idx])[:, 0]
        db = M.sdf_boxes(c, self.box[idx])[:, 0]
        ic = np.argsort(dc, axis=1)[:, :NC_LOCAL]
        ib = np.argsort(db, axis=1)[:, :NB_LOCAL]
        self.cyl_l[idx] = np.take_along_axis(self.cyl[idx], ic[..., None], 1)
        self.box_l[idx] = np.take_along_axis(self.box[idx], ib[..., None], 1)

    # ------------------------------------------------------------ history
    def _history_features(self):
        return np.concatenate([self.tension / 10, self.cv / 2, self.cp - self.p.mean(1)], -1)

    # ------------------------------------------------------------ wind zones
    def _zone_wind(self, bodies):
        steady, turb = M.zone_wind(bodies, self.zones)
        decay = np.exp(-DT / 0.4)
        self.zone_ou = decay * self.zone_ou + np.sqrt(1 - decay ** 2) * self.rng.normal(size=self.zone_ou.shape)
        return (steady + turb[..., None] * self.zone_ou) * self.wind_factor[:, None, None]

    # ------------------------------------------------------------ route helpers
    def _interp(self, s):
        i = np.clip((s / 0.25).astype(np.int64), 0, self.n_path - 2)
        b = np.arange(self.B)
        s0, s1 = self.s_path[b, i], self.s_path[b, i + 1]
        f = np.clip((s - s0) / np.maximum(s1 - s0, 1e-9), 0, 1)[:, None]
        p = self.path[b, i] * (1 - f) + self.path[b, i + 1] * f
        tan = self.path[b, i + 1] - self.path[b, i]
        tan /= np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-9)
        v = self.v_path[b, i] * (1 - f[:, 0]) + self.v_path[b, i + 1] * f[:, 0]
        return p, tan, v

    def _project(self):
        b = np.arange(self.B)
        win = self.proj_idx[:, None] + np.arange(-8, 41)[None]
        win = np.clip(win, 0, self.n_path[:, None] - 1)
        d = np.linalg.norm(self.path[b[:, None], win, :2] - self.cp[:, None, :2], axis=-1)
        self.proj_idx = win[b, d.argmin(1)]
        self.s_proj = self.s_path[b, self.proj_idx]

    def _advance_reference(self):
        self._project()
        _, _, v = self._interp(self.s_ref)
        s_new = np.minimum(self.s_ref + v * DT, np.minimum(self.s_proj + LEAD, self.path_len))
        s_new = np.maximum(self.s_ref, s_new)
        p, tan, _ = self._interp(s_new)
        self.v_ref = tan * ((s_new - self.s_ref) / DT)[:, None]
        self.s_ref = s_new
        self.target = p

    # ------------------------------------------------------------ sensing
    def _sense(self, pts):
        """Obstacle distance and outward direction for points (B, P, 3)."""
        e = 0.05
        off = np.array([[0, 0, 0], [e, 0, 0], [-e, 0, 0], [0, e, 0], [0, -e, 0], [0, 0, e], [0, 0, -e]])
        q = (pts[:, :, None, :] + off).reshape(self.B, -1, 3)
        d = M.sdf(q, self.cyl_l, self.box_l).reshape(self.B, pts.shape[1], 7)
        grad = np.stack([d[..., 1] - d[..., 2], d[..., 3] - d[..., 4], d[..., 5] - d[..., 6]], -1) / (2 * e)
        grad /= np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-9)
        return d[..., 0], grad

    def _rope_points(self):
        f = np.array([0.25, 0.5, 0.75])[None, None, :, None]
        pts = self.cp[:, None, None] + (self.p[:, :, None] - self.cp[:, None, None]) * f
        return pts.reshape(self.B, 9, 3)

    # ------------------------------------------------------------ step
    def step(self, action):
        prev_s = self.s_proj.copy()
        if self.residual > 0:
            action = compose_residual(self._obs, np.asarray(action).reshape(self.B, -1), self.residual)
        a, _ = self._advance(action)
        self._advance_reference()
        B = self.B
        self._n_steps = getattr(self, "_n_steps", 0) + 1
        if self._n_steps % LOCAL_REFRESH == 0:
            self._refresh_local()

        bodies = np.concatenate([self.p, self.cp[:, None]], 1)
        dist, grad = self._sense(bodies)
        rope_d = M.sdf(self._rope_points(), self.cyl_l, self.box_l).reshape(B, 3, 3)
        self._cache = (dist, grad, rope_d[:, :, 1])

        # --- cargo damage
        T0 = self.cargo_mass * G / 3 * 1.2
        jerk = 0.4 * np.sum(np.maximum(self.tension_peak / T0[:, None] - 2.0, 0), -1)
        jerk_warning = np.sum(np.maximum(self.tension_peak / T0[:, None] - JERK_WARNING, 0), -1)
        spike = np.sum(np.clip(self.tension_peak / T0[:, None] - 2.0, 0, SPIKE_CAP) ** 2, -1)
        acc = np.linalg.norm(self.cv - self.prev_cv, axis=-1) / DT
        gforce = 0.5 * np.maximum(acc / G - 0.8, 0)
        touching = dist[:, 3] < CARGO_R
        onset = touching & ~self.contact
        impact = np.where(onset, 10 + 15 * np.linalg.norm(self.cv, axis=-1), 0) + 0.5 * touching
        rub = 0.2 * np.sum((rope_d < 0.02).any(-1), -1)
        dmg = np.stack([jerk, gforce, impact, rub], 1)
        self.damage += dmg
        self.step_damage = dmg.sum(1)
        self.health = np.maximum(self.health - self.step_damage, 0)
        self.contact = touching; self.impact_event = onset
        self.prev_cv = self.cv.copy()

        # --- failures
        tilt = self.tilt()
        sep = np.stack([np.linalg.norm(self.p[:, i] - self.p[:, j], axis=-1) for i, j in PAIRS], 1)
        fail = np.zeros(B, dtype=np.int64)
        out = ((bodies[..., 0] < -1) | (bodies[..., 0] > self.length[:, None] + 1)
               | (np.abs(bodies[..., 1]) > M.WIDTH / 2) | (bodies[..., 2] > 8)).any(-1)
        fail[out] = 5
        fail[self.health <= 0] = 7
        fail[sep.min(-1) < 0.25] = 4
        fail[(dist[:, :3] < DRONE_R).any(-1)] = 6
        fail[(tilt > np.radians(80)).any(-1)] = 3
        fail[self.cp[:, 2] < 0.1] = 2
        fail[(self.p[..., 2] < 0.05).any(-1)] = 1

        # --- delivery
        at_goal = ((self.s_ref >= self.path_len - 1e-6)
                   & (np.linalg.norm(self.cp - self.goal, axis=-1) < DELIVER_RADIUS)
                   & (np.linalg.norm(self.cv, axis=-1) < DELIVER_SPEED))
        self.hold = np.where(at_goal, self.hold + 1, 0)
        delivered = (self.hold >= DELIVER_HOLD) & (fail == 0)

        # --- reward
        err = self.cargo_error()
        clear_pen = (np.sum(np.maximum(0.6 - dist[:, :3], 0), -1) + np.maximum(0.4 - dist[:, 3], 0))
        # Cargo health dominates: every health point lost costs 3, and the delivery
        # bonus is mostly proportional to the health left. Route tracking is only
        # a weak guide, so the policy may lag the reference to fly more gently.
        reward = (0.1 - 0.25 * err - 0.02 * np.sum((self.cv - self.v_ref) ** 2, -1)
                  - 0.2 * np.mean(tilt ** 2, -1) - 0.003 * np.mean(np.sum(self.w ** 2, -1), -1)
                  - 2.0 * np.sum(np.maximum(0.5 - sep, 0), -1)
                  - 0.02 * np.mean(a.reshape(B, -1) ** 2, -1)
                  - 0.1 * np.mean((a.reshape(B, -1) - self.last_action) ** 2, -1)
                  + 1.0 * (self.s_proj - prev_s) - 3.0 * self.step_damage - 3.0 * clear_pen)
        reward = reward - self.jerk_weight * jerk_warning - self.spike_weight * spike
        reward = reward - 30.0 * (fail > 0) + delivered * (20.0 + 80.0 * self.health / 100)
        self.last_action = a.reshape(B, -1)
        self.crash_code = fail

        self.past_i = (self.past_i + 1) % self.past.shape[1]
        self.past[:, self.past_i] = self._history_features()

        terminated = (fail > 0) | delivered
        truncated = (self.t >= self.time_limit) & ~terminated
        info = {"fail": fail.copy(), "delivered": delivered.copy(), "health": self.health.copy()}
        if self.autoreset:
            done = np.nonzero(terminated | truncated)[0]
            if len(done):
                self.reset(done)
        return self.observe(), reward, terminated, truncated, info

    def _sense_all(self):
        if self._cache is not None:
            return self._cache
        bodies = np.concatenate([self.p, self.cp[:, None]], 1)
        dist, grad = self._sense(bodies)
        mid = self.cp[:, None] + 0.5 * (self.p - self.cp[:, None])
        return dist, grad, M.sdf(mid, self.cyl_l, self.box_l)

    # ------------------------------------------------------------ critic-only state
    def privileged(self):
        """What the simulator knows and the policy does not, for a training critic (B, PRIV_DIM):
        gust and turbulence states, the forces they apply now, the wind zones 2 and 4 m further along
        the route, and this episode's gust settings, cargo mass, rope length and motor wear."""
        B = self.B
        ahead = [self._interp(np.minimum(self.s_ref + k, self.path_len))[0] for k in (2.0, 4.0)]
        steady, turb = M.zone_wind(np.stack(ahead, 1), self.zones)
        return np.concatenate([
            self.gust_common, self.gust_body.reshape(B, -1), self.zone_ou.reshape(B, -1),
            self.drone_force.reshape(B, -1), self.cargo_force, steady.reshape(B, -1), turb,
            self.gust_std[:, None], self.gust_tau[:, None], self.cargo_mass[:, None],
            self.rope_len[:, None], self.motor_gain.reshape(B, -1),
        ], -1).astype(np.float32)

    # ------------------------------------------------------------ observation
    def observe(self):
        base = super().observe()
        B = self.B
        ahead = []
        for k in (2.0, 4.0):
            p, _, _ = self._interp(np.minimum(self.s_ref + k, self.path_len))
            ahead.append((p - self.cp) / 4)
        dist, grad, rope_d = self._sense_all()
        sense = np.concatenate([np.minimum(dist, 3.0)[..., None] / 3, grad], -1).reshape(B, -1)
        extra = np.concatenate([
            self.v_ref / 2, *ahead, sense, np.minimum(rope_d, 3.0) / 3,
            self.health[:, None] / 100, ((self.path_len - self.s_proj) / 40)[:, None],
        ], -1)
        self._obs = np.concatenate([base, extra.astype(np.float32)], -1)
        if self.history:
            n = self.past.shape[1]
            lagged = [self.past[:, (self.past_i - lag) % n] for lag in HISTORY_LAGS]
            self._obs = np.concatenate([self._obs, self.tension_peak / 10, *lagged], -1).astype(np.float32)
        c = getattr(self, "realism", None)
        if c is not None and (c.sensor_noise > 0 or c.sensor_bias > 0):
            noise = self.real_rng.normal(size=self._obs.shape) * c.sensor_noise
            self._obs = (self._obs + (self.sensor_bias + noise) * self.real_s[:, None]).astype(np.float32)
        return self._obs
