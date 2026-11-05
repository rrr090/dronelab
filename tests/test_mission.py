import json

import numpy as np

from dronelab.mission import maps as M
from dronelab.cargo.env import ACT_DIM
from dronelab.mission.env import MISSION_OBS_DIM, MissionEnv
from dronelab.mission.testing import MissionFormationController, run_missions
from dronelab.mission.viewer import build_viewer_data, write_viewer


def test_every_theme_generates_a_clear_route():
    for th in M.THEMES:
        m = M.generate_map(11, th, 1.0)
        assert m.path is not None and len(m.path) > 10
        cyl, box = M._blocking(m)
        clear = M.sdf(np.column_stack([m.path[:, :2], np.full(len(m.path), M.CRUISE_Z + 0.9)]), cyl, box)
        assert clear.min() > M.CLEARANCE * 0.85
        assert np.allclose(m.path[0, :2], m.start[:2]) and np.allclose(m.path[-1, :2], m.goal[:2])


def test_maps_are_deterministic():
    a, b = M.generate_map(3, "mixed", 0.7), M.generate_map(3, "mixed", 0.7)
    assert np.array_equal(a.cylinders, b.cylinders) and np.array_equal(a.path, b.path)


def test_sdf_signs():
    cyl = np.array([[0.0, 0.0, 1.0, 2.0]])
    box = np.zeros((0, 6))
    d = M.sdf(np.array([[0.0, 0, 1], [3.0, 0, 1], [0.0, 0, 3]]), cyl, box)
    assert d[0] < 0 and np.isclose(d[1], 2.0) and np.isclose(d[2], 1.0)


def test_mission_env_shapes_and_reference_moves():
    env = MissionEnv(2, seed=0, maps=[M.generate_map(1, "forest", 0.3), M.generate_map(2, "city", 0.3)])
    obs = env.reset()
    assert obs.shape == (2, MISSION_OBS_DIM)
    s0 = env.s_ref.copy()
    for _ in range(20):
        obs, r, term, trunc, info = env.step(np.zeros((2, ACT_DIM)))
    assert np.all(env.s_ref > s0) and np.all(env.s_ref <= env.s_proj + 1.0 + 1e-6)


def test_hitting_an_obstacle_damages_cargo():
    m = M.generate_map(5, "forest", 0.0)
    env = MissionEnv(1, seed=0, maps=[m])
    env.reset()
    env.cyl[0, 0] = [env.cp[0, 0] + 0.15, env.cp[0, 1], 0.1, 5.0]   # tree right at the cargo
    env._refresh_local()
    env.step(np.zeros((1, ACT_DIM)))
    assert env.damage[0, 2] > 5 and env.health[0] < 95


def test_baseline_delivers_on_an_easy_map():
    res = run_missions(MissionFormationController(), [M.generate_map(9, "storm", 0.0)])[0]
    assert res.delivered and 0 < res.score <= 100


def test_mission_viewer_export(tmp_path):
    data = build_viewer_data({"FORMATION": MissionFormationController()},
                                     [M.generate_map(4, "gates", 0.0)])
    run = data["missions"][0]["runs"][0]
    assert len(run["dp"]) == run["n"] * 9 and len(run["hp"]) == run["n"]
    html = write_viewer(data, tmp_path / "m.html").read_text(encoding="utf-8")
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload)["meta"]["policies"] == ["FORMATION"]


def test_residual_zero_correction_flies_the_baseline():
    from dronelab.mission.env import baseline_actions
    m = M.generate_map(4, "forest", 0.3)
    a, b = MissionEnv(1, seed=0, maps=[m]), MissionEnv(1, seed=0, maps=[m], residual=0.3)
    oa, ob = a.reset(), b.reset()
    for _ in range(30):
        oa = a.step(baseline_actions(oa))[0]
        ob = b.step(np.zeros((1, ACT_DIM)))[0]
    assert np.allclose(a.cp, b.cp) and np.allclose(oa, ob)
