import json

import numpy as np
import pytest
import torch

from types import SimpleNamespace

from dronelab.mission import maps as M
from dronelab.cargo.env import ACT_DIM, CARGO_MASS, G
from dronelab.mission.env import (
    HISTORY_DIM, HISTORY_FEATURES, HISTORY_LAGS, JERK_WARNING, MISSION_OBS_DIM, MissionEnv,
)
from dronelab.mission.testing import (
    MissionFormationController, MissionResult, bootstrap_ci, compare_to_baseline, load_mission_policy,
    run_missions,
)
from dronelab.policy import ActorCritic
from dronelab.mission.site import median_seed, pick_showcase
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


def _scored(scores):
    return [MissionResult(map=None, delivered=s > 0, fail_reason="" if s > 0 else "timeout", health=s,
                          score=s, time_s=1.0, damage={}, mean_track_err=0.0, min_clearance=1.0)
            for s in scores]


def test_bootstrap_ci_brackets_the_mean():
    mean, low, high = bootstrap_ci(np.arange(100.0))
    assert low < mean < high and np.isclose(mean, 49.5)
    assert bootstrap_ci(np.full(10, 3.0)) == (3.0, 3.0, 3.0)


def test_comparison_needs_every_seed_to_agree():
    base = np.linspace(40, 80, 50)
    win = compare_to_baseline({"a": _scored(base + 5), "b": _scored(base + 7)}, _scored(base))
    assert win["verdict"] == "trained policy beats the baseline"
    assert np.isclose(win["score"]["diff"], 6) and win["score"]["ci_low"] > 0

    lose = compare_to_baseline({"a": _scored(base - 5)}, _scored(base))
    assert lose["verdict"] == "baseline beats the trained policy"

    split = compare_to_baseline({"a": _scored(base + 20), "b": _scored(base - 1)}, _scored(base))
    assert split["score"]["ci_low"] > 0 and split["verdict"] == "no clear winner"


def _zero_correction_checkpoints(tmp_path, n=2):
    import torch
    from dronelab.policy import ActorCritic
    from dronelab.ppo import save

    model = ActorCritic(hidden=16, obs_dim=MISSION_OBS_DIM, act_dim=ACT_DIM, obs_norm=True, init_log_std=-1.0)
    with torch.no_grad():
        model.actor_mean.weight.zero_()
        model.actor_mean.bias.zero_()
    paths = [str(tmp_path / f"s{i}.pt") for i in range(n)]
    for path in paths:
        save(model, path, {"residual": 0.3})
    return paths


def test_evaluate_scores_seed_checkpoints_against_the_baseline(tmp_path):
    from dronelab.mission import evaluate

    paths = _zero_correction_checkpoints(tmp_path)
    report = tmp_path / "eval.json"
    evaluate.main(["--checkpoint", *paths, "--themes", "forest", "--difficulties", "0.0",
                   "--per", "2", "--json", str(report)])
    out = json.loads(report.read_text())
    assert set(out) == {"s0", "s1", "formation", "comparison"}
    assert out["comparison"]["score"]["diff"] == 0          # zero correction is the baseline
    assert out["comparison"]["verdict"] == "no clear winner"
    assert [m["map_seed"] for m in out["s0"]["per_map"]] == [m["map_seed"] for m in out["s1"]["per_map"]]
    assert len(out["s0"]["per_map"]) == 2
    assert np.mean([m["score"] for m in out["s0"]["per_map"]]) == pytest.approx(out["s0"]["mean_score"])


def _themed(scores, themes):
    rs = _scored(scores)
    for r, theme in zip(rs, themes):
        r.map = SimpleNamespace(theme=theme)
    return rs


def test_showcase_has_each_themes_best_map_then_the_worst():
    themes = ["forest", "forest", "city", "city", "city"]
    base = _themed([50, 50, 50, 50, 50], themes)
    trained = _themed([60, 45, 80, 20, 55], themes)
    assert pick_showcase(trained, base) == [2, 0, 3]

    one_theme = _themed([60, 40], ["forest", "forest"])
    assert pick_showcase(one_theme, _themed([50, 50], ["forest", "forest"])) == [0, 1]


def test_replays_come_from_the_middle_seed():
    assert median_seed({"low": _scored([10]), "high": _scored([90]), "mid": _scored([50])}) == "mid"


def test_site_replays_are_the_flights_behind_the_numbers(tmp_path):
    from dronelab.mission import site

    out = tmp_path / "site"
    site.main(["--checkpoint", *_zero_correction_checkpoints(tmp_path), "--themes", "forest", "city",
               "--difficulties", "0.0", "--per", "2", "--out", str(out)])
    page = (out / "index.html").read_text(encoding="utf-8")
    viewer = (out / "viewer.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in page and "/*__DATA__*/null" not in viewer
    assert (out / "images" / "mission-forest.png").exists()

    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    assert data["comparison"]["maps"] == 4 and data["comparison"]["verdict"] == "no clear winner"
    assert [t["theme"] for t in data["themes"]] == ["forest", "city"]
    assert data["showcase"][-1]["worst"] and not data["showcase"][0]["worst"]
    assert all(s["trained"] == s["baseline"] for s in data["showcase"])   # zero correction is the baseline


def test_training_keeps_the_best_validation_checkpoint(tmp_path):
    from dronelab.mission.train import make_eval
    from dronelab.policy import ActorCritic

    model = ActorCritic(hidden=16, obs_dim=MISSION_OBS_DIM, act_dim=ACT_DIM, obs_norm=True, init_log_std=-1.0)
    best = tmp_path / "best.pt"
    quick_eval = make_eval(0.1, str(best), maps=[M.generate_map(1, "forest", 0.0)])
    assert "new best" in quick_eval(model, "cpu") and best.exists()

    best.unlink()
    assert "new best" not in quick_eval(model, "cpu") and not best.exists()   # same score is no gain


def history_env():
    return MissionEnv(2, seed=0, maps=[M.generate_map(1, "forest", 0.3), M.generate_map(2, "city", 0.3)],
                      history=True)


def test_history_inputs_only_extend_the_observation():
    a = MissionEnv(1, seed=0, maps=[M.generate_map(1, "forest", 0.3)])
    b = MissionEnv(1, seed=0, maps=[M.generate_map(1, "forest", 0.3)], history=True)
    oa, ob = a.reset(), b.reset()
    assert ob.shape == (1, MISSION_OBS_DIM + HISTORY_DIM)
    for _ in range(40):
        oa, ob = a.step(np.zeros((1, ACT_DIM)))[0], b.step(np.zeros((1, ACT_DIM)))[0]
    assert np.array_equal(oa, ob[:, :MISSION_OBS_DIM])


def test_history_shows_the_state_from_each_lag():
    env = history_env()
    env.reset()
    states = [env._history_features().copy()]
    for _ in range(max(HISTORY_LAGS) + 3):
        obs = env.step(np.zeros((2, ACT_DIM)))[0]
        states.append(env._history_features().copy())
    assert np.allclose(obs[:, MISSION_OBS_DIM:MISSION_OBS_DIM + 3], env.tension_peak / 10, atol=1e-6)
    for n, lag in enumerate(HISTORY_LAGS):
        start = MISSION_OBS_DIM + 3 + n * HISTORY_FEATURES
        assert np.allclose(obs[:, start:start + HISTORY_FEATURES], states[-1 - lag], atol=1e-6)


def test_reset_clears_history_for_that_env_only():
    env = history_env()
    env.reset()
    for _ in range(40):
        env.step(np.zeros((2, ACT_DIM)))
    kept = env.past[1].copy()
    env.reset(np.array([0]))
    assert np.allclose(env.past[0], env._history_features()[0])
    assert np.array_equal(env.past[1], kept)


def test_jerk_weight_only_penalises_tension_above_the_warning_level():
    def reward(weight, spike):
        env = MissionEnv(1, seed=0, maps=[M.generate_map(1, "forest", 0.3)], jerk_weight=weight)
        env.reset()
        original = env._advance

        def advance(action):
            out = original(action)
            env.tension_peak[:] = spike
            return out
        env._advance = advance
        return env.step(np.zeros((1, ACT_DIM)))[1][0]

    assert reward(1.0, 0.0) == reward(0.0, 0.0)
    high = 1.8 * env_t0()                # above the warning, below the damage level
    assert np.isclose(reward(0.0, high) - reward(1.0, high), 3 * (1.8 - JERK_WARNING))


def env_t0():
    return CARGO_MASS * G / 3 * 1.2


def test_spike_weight_is_quadratic_above_the_damage_level_and_capped():
    def reward(weight, ratio):
        env = MissionEnv(1, seed=0, maps=[M.generate_map(1, "forest", 0.3)], spike_weight=weight)
        env.reset()
        original = env._advance

        def advance(action):
            out = original(action)
            env.tension_peak[:] = ratio * env_t0()
            return out
        env._advance = advance
        return env.step(np.zeros((1, ACT_DIM)))[1][0]

    penalty = lambda ratio: reward(0.0, ratio) - reward(1.0, ratio)
    assert np.isclose(penalty(1.9), 0.0)                # below the damage level: unchanged
    assert np.isclose(penalty(3.0), 3 * 1.0 ** 2)       # three ropes, one T0 above the level
    assert np.isclose(penalty(4.0), 3 * 2.0 ** 2)
    assert np.isclose(penalty(5.0), penalty(9.0))       # capped


def test_saved_history_policy_flies_with_the_right_inputs(tmp_path):
    model = ActorCritic(hidden=8, obs_dim=MISSION_OBS_DIM + HISTORY_DIM, act_dim=ACT_DIM, obs_norm=True)
    path = tmp_path / "h.pt"
    torch.save({"config": model.config, "state_dict": model.state_dict(), "residual": 0.1}, path)
    policy = load_mission_policy(str(path))
    assert policy.history
    assert run_missions(policy, [M.generate_map(9, "storm", 0.0)])[0].steps > 0


def test_resume_keeps_history_and_jerk_weight(tmp_path, monkeypatch):
    from dronelab.mission import train
    monkeypatch.setattr(train, "make_eval", lambda *a, **k: None)
    tiny = ["--updates", "1", "--envs", "2", "--steps", "8", "--hidden", "8", "--critic-warmup", "0"]
    first, second = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    train.main(tiny + ["--history", "--jerk-weight", "0.7", "--spike-weight", "0.4", "--out", first])
    train.main(tiny + ["--resume", first, "--out", second])
    ckpt = torch.load(second, map_location="cpu")
    assert ckpt["config"]["obs_dim"] == MISSION_OBS_DIM + HISTORY_DIM
    assert ckpt["jerk_weight"] == 0.7 and ckpt["residual"] == 0.1 and ckpt["spike_weight"] == 0.4
