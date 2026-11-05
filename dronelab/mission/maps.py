"""Procedurally generated mission maps for the cargo team.

A map is a corridor from a start pad to a goal pad with:

* tall obstacles that must be flown around: trees/pillars (cylinders),
  buildings (boxes) and walls with a gap to fly through;
* low obstacles (crates, hedges) the team can overfly, but a swinging cargo
  can clip them;
* wind zones: spheres of strong steady wind plus turbulence, including
  downdrafts and updrafts.

Themes: forest, city, gates, storm, mixed. ``difficulty`` (0..1) scales
obstacle density, map length and wind strength.

``plan_route`` finds a path with enough clearance for the whole formation
(A* on a clearance grid, preferring to avoid strong wind), then smooths it.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from dronelab.cargo.env import FORMATION_R

THEMES = ["forest", "city", "gates", "storm", "mixed"]
WIDTH = 16.0                 # map width (y from -8 to 8) [m]
CRUISE_Z = 1.5               # cargo altitude while flying the route [m]
LOW_MAX_H = 0.9              # obstacles at most this tall can be overflown
CLEARANCE = FORMATION_R + 0.25 + 0.5   # route clearance for the formation [m]
GRID = 0.25                  # planner grid resolution [m]
NC_MAX, NB_MAX, NZ_MAX = 96, 40, 12    # padding sizes for batched envs
PAD_FAR = 1e4


@dataclass
class MissionMap:
    seed: int
    theme: str
    difficulty: float
    length: float
    start: np.ndarray                     # (3,) cargo start position
    goal: np.ndarray                      # (3,) cargo delivery point
    cylinders: np.ndarray                 # (Nc, 4) x, y, radius, height
    boxes: np.ndarray                     # (Nb, 6) cx, cy, half_x, half_y, height, kind
    zones: np.ndarray                     # (Nz, 8) cx, cy, cz, radius, wx, wy, wz, turbulence
    gust_std: float                       # background gusts everywhere [N]
    path: np.ndarray = field(default=None, repr=False)   # (P, 3) planned route

    BOX_KINDS = ["building", "wall", "crate"]

    def to_dict(self) -> dict:
        r = lambda a, nd=3: np.round(np.asarray(a, float), nd).tolist()
        return {
            "seed": self.seed, "theme": self.theme, "difficulty": self.difficulty,
            "length": self.length, "width": WIDTH, "start": r(self.start), "goal": r(self.goal),
            "cylinders": r(self.cylinders), "boxes": r(self.boxes), "zones": r(self.zones),
            "gust_std": self.gust_std, "path": r(self.path[::2] if self.path is not None else []),
        }


# ------------------------------------------------------------------ signed distances

def sdf_cylinders(pts, cyl):
    """pts (..., P, 3), cyl (..., N, 4) -> (..., P, N) signed distance (ground-standing)."""
    dx = pts[..., :, None, 0] - cyl[..., None, :, 0]
    dy = pts[..., :, None, 1] - cyl[..., None, :, 1]
    rr = np.hypot(dx, dy) - cyl[..., None, :, 2]
    qz = pts[..., :, None, 2] - cyl[..., None, :, 3]
    out = np.hypot(np.maximum(rr, 0), np.maximum(qz, 0))
    return out + np.minimum(np.maximum(rr, qz), 0)


def sdf_boxes(pts, box):
    """pts (..., P, 3), box (..., N, 6) -> (..., P, N) signed distance (ground-standing)."""
    qx = np.abs(pts[..., :, None, 0] - box[..., None, :, 0]) - box[..., None, :, 2]
    qy = np.abs(pts[..., :, None, 1] - box[..., None, :, 1]) - box[..., None, :, 3]
    hz = box[..., None, :, 4] / 2
    qz = np.abs(pts[..., :, None, 2] - hz) - hz
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return out + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0)


def sdf(pts, cyl, box):
    """Distance from points to the nearest obstacle surface, (..., P)."""
    d = np.full(pts.shape[:-1], PAD_FAR)
    if cyl.shape[-2]:
        d = np.minimum(d, sdf_cylinders(pts, cyl).min(-1))
    if box.shape[-2]:
        d = np.minimum(d, sdf_boxes(pts, box).min(-1))
    return d


def zone_wind(pts, zones):
    """Steady wind force at points from wind zones. pts (..., P, 3), zones (..., Nz, 8).

    Returns (steady force (..., P, 3), turbulence weight (..., P))."""
    d = np.linalg.norm(pts[..., :, None, :] - zones[..., None, :, 0:3], axis=-1)
    f = np.clip(1.0 - d / np.maximum(zones[..., None, :, 3], 1e-6), 0, 1)
    f = f * f * (3 - 2 * f)                                  # smooth falloff
    steady = np.einsum("...pn,...nk->...pk", f, zones[..., 4:7])
    turb = np.einsum("...pn,...n->...p", f, zones[..., 7])
    return steady, turb


# ------------------------------------------------------------------ generation

def _theme_spec(theme, d, rng):
    """Obstacle counts and wind strength for a theme at difficulty d."""
    s = 0.5 + 0.5 * d
    if theme == "forest":
        return dict(trees=int(18 + 50 * d), buildings=0, walls=0, low=int(3 + 6 * d),
                    zones=int(1 + 2 * d), zone_force=1.0 * s, drafts=0.2)
    if theme == "city":
        return dict(trees=int(2 + 4 * d), buildings=int(6 + 12 * d), walls=0, low=int(4 + 6 * d),
                    zones=int(2 + 3 * d), zone_force=1.6 * s, drafts=0.3)
    if theme == "gates":
        return dict(trees=int(3 + 6 * d), buildings=0, walls=int(2 + 3 * d), low=int(2 + 4 * d),
                    zones=int(1 + 2 * d), zone_force=1.3 * s, drafts=0.2)
    if theme == "storm":
        return dict(trees=int(3 + 4 * d), buildings=int(1 + 2 * d), walls=0, low=int(2 + 3 * d),
                    zones=int(4 + 5 * d), zone_force=2.2 * s, drafts=0.5)
    return dict(trees=int(8 + 22 * d), buildings=int(2 + 6 * d), walls=int(1 + 2 * d),
                low=int(3 + 5 * d), zones=int(2 + 3 * d), zone_force=1.6 * s, drafts=0.35)


def generate_map(seed: int, theme: str = "mixed", difficulty: float = 0.5) -> MissionMap:
    """Generate a solvable map (a route with formation clearance always exists)."""
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}, choose from {THEMES}")
    for attempt in range(30):
        m = _generate_once(seed * 101 + attempt, seed, theme, difficulty, thin=attempt / 30)
        path = plan_route(m)
        if path is not None:
            m.path = path
            return m
    m.cylinders = m.cylinders[:0]; m.boxes = m.boxes[:0]
    m.path = plan_route(m)
    return m


def _generate_once(sub_seed, seed, theme, d, thin=0.0):
    rng = np.random.default_rng(sub_seed)
    spec = _theme_spec(theme, d, rng)
    keep = 1.0 - thin                      # thin the map out on retries
    L = float(30 + 15 * d)
    half = WIDTH / 2
    start = np.array([2.5, rng.uniform(-3, 3), CRUISE_Z])
    goal = np.array([L - 2.5, rng.uniform(-3, 3), CRUISE_Z])

    def free(x, y, rad):
        return (np.hypot(x - start[0], y - start[1]) > rad + 3.0
                and np.hypot(x - goal[0], y - goal[1]) > rad + 3.0)

    cyl, box = [], []
    for _ in range(int(spec["trees"] * keep)):
        x, y, r = rng.uniform(4, L - 4), rng.uniform(-half + 0.5, half - 0.5), rng.uniform(0.25, 0.6)
        if free(x, y, r):
            cyl.append([x, y, r, rng.uniform(4, 10)])
    for _ in range(int(spec["buildings"] * keep)):
        x, y = rng.uniform(6, L - 6), rng.uniform(-half + 1, half - 1)
        hx, hy = rng.uniform(1, 3), rng.uniform(1, 3)
        if free(x, y, max(hx, hy)):
            box.append([x, y, hx, hy, rng.uniform(3, 12), 0])
    n_walls = int(spec["walls"] * keep)
    if n_walls:
        xs = np.linspace(8, L - 8, n_walls) + rng.uniform(-1.5, 1.5, n_walls)
        for x in xs:
            gap = rng.uniform(3.4, 4.6) - 0.6 * d
            gy = rng.uniform(-half + gap / 2 + 1, half - gap / 2 - 1)
            lo, hi = gy - gap / 2, gy + gap / 2
            h = rng.uniform(4, 6)
            box.append([x, (-half + lo) / 2, 0.15, (lo + half) / 2, h, 1])
            box.append([x, (hi + half) / 2, 0.15, (half - hi) / 2, h, 1])
    for _ in range(int(spec["low"] * keep)):
        x, y = rng.uniform(5, L - 5), rng.uniform(-half + 1, half - 1)
        if rng.random() < 0.5:
            box.append([x, y, rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8), rng.uniform(0.4, LOW_MAX_H), 2])
        else:
            cyl.append([x, y, rng.uniform(0.3, 0.7), rng.uniform(0.4, LOW_MAX_H)])

    zones = []
    for _ in range(spec["zones"]):
        x, y = rng.uniform(5, L - 5), rng.uniform(-4.5, 4.5)
        rad = rng.uniform(2.0, 4.0)
        f = spec["zone_force"] * rng.uniform(0.6, 1.0)
        if rng.random() < spec["drafts"]:
            w = np.array([rng.normal(0, 0.2), rng.normal(0, 0.2), -f if rng.random() < 0.7 else 0.7 * f])
        else:
            a = rng.uniform(0, 2 * np.pi)
            w = np.array([f * np.cos(a), f * np.sin(a), 0.0])
        zones.append([x, y, CRUISE_Z + 0.5, rad, *w, rng.uniform(0.2, 0.6) * f])

    return MissionMap(
        seed=seed, theme=theme, difficulty=float(d), length=L, start=start, goal=goal,
        cylinders=np.array(cyl, float).reshape(-1, 4), boxes=np.array(box, float).reshape(-1, 6),
        zones=np.array(zones, float).reshape(-1, 8), gust_std=0.15 + 0.35 * d,
    )


# ------------------------------------------------------------------ planning

def _blocking(m: MissionMap):
    cyl = m.cylinders[m.cylinders[:, 3] > LOW_MAX_H] if len(m.cylinders) else m.cylinders
    box = m.boxes[m.boxes[:, 4] > LOW_MAX_H] if len(m.boxes) else m.boxes
    return cyl, box


def clearance_grid(m: MissionMap):
    """2D clearance (distance to nearest tall obstacle or map edge) on the planner grid."""
    xs = np.arange(0, m.length + 1e-9, GRID)
    ys = np.arange(-WIDTH / 2, WIDTH / 2 + 1e-9, GRID)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, CRUISE_Z + 0.9)], -1)
    cyl, box = _blocking(m)
    d = sdf(pts, cyl, box).reshape(X.shape)
    edge = np.minimum.reduce([X, m.length - X, Y + WIDTH / 2, WIDTH / 2 - Y])
    return xs, ys, np.minimum(d, edge)


def plan_route(m: MissionMap, spacing: float = 0.25):
    xs, ys, clear = clearance_grid(m)
    ok = clear > CLEARANCE
    # Wind cost: prefer to route around strong wind zones.
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, CRUISE_Z)], -1)
    steady, turb = zone_wind(pts, m.zones) if len(m.zones) else (np.zeros((X.size, 3)), np.zeros(X.size))
    wind_cost = (np.linalg.norm(steady, axis=-1) + turb).reshape(X.shape)
    near = np.clip((CLEARANCE + 1.0 - clear) / 1.0, 0, 1)   # prefer some extra margin

    def cell(p):
        return int(round(p[0] / GRID)), int(round((p[1] + WIDTH / 2) / GRID))

    s, g = cell(m.start), cell(m.goal)
    nx, ny = ok.shape
    if not (ok[s] and ok[g]):
        return None
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    gscore = np.full(ok.shape, np.inf); gscore[s] = 0
    parent = {}
    pq = [(0.0, s)]
    while pq:
        _, c = heapq.heappop(pq)
        if c == g:
            break
        for dx, dy in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not ok[n]:
                continue
            step = GRID * (1.4142 if dx and dy else 1.0)
            cost = gscore[c] + step * (1.0 + 0.8 * wind_cost[n] + 0.5 * near[n])
            if cost < gscore[n]:
                gscore[n] = cost; parent[n] = c
                h = GRID * np.hypot(g[0] - n[0], g[1] - n[1])
                heapq.heappush(pq, (cost + h, n))
    if g not in parent and g != s:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts2 = np.array([[xs[i], ys[j]] for i, j in cells])
    pts2[0], pts2[-1] = m.start[:2], m.goal[:2]

    # Smooth with a moving average, keeping clearance; then resample evenly.
    for _ in range(4):
        sm = pts2.copy()
        k = 6
        for i in range(1, len(pts2) - 1):
            lo, hi = max(0, i - k), min(len(pts2), i + k + 1)
            sm[i] = pts2[lo:hi].mean(0)
        cand = np.column_stack([sm, np.full(len(sm), CRUISE_Z + 0.9)])
        cyl, box = _blocking(m)
        d = sdf(cand, cyl, box)
        good = d > CLEARANCE * 0.9
        pts2 = np.where(good[:, None], sm, pts2)
    seg = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
    s_cum = np.concatenate([[0], np.cumsum(seg)])
    s_new = np.arange(0, s_cum[-1], spacing)
    s_new = np.append(s_new, s_cum[-1])
    x = np.interp(s_new, s_cum, pts2[:, 0]); y = np.interp(s_new, s_cum, pts2[:, 1])
    return np.column_stack([x, y, np.full(len(x), CRUISE_Z)])
