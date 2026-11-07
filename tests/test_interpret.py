import numpy as np
import torch

from dronelab.mission import maps as M
from dronelab.mission.env import HISTORY_DIM, HISTORY_LAGS, MISSION_OBS_DIM, MissionEnv
from dronelab.mission.interpret import (
    OBSTACLE, PAST_CARGO_VEL, PAST_OFFSET, PAST_TENSION, PEAK_TENSION, REL, TENSION, VEL, ablated_policy,
    collect, concept_signals, forward, matched_control,
    occlude, swing_state, tuning, untrained_copy,
)
from dronelab.mission.testing import MissionFormationController, load_mission_policy
from dronelab.policy import ActorCritic


def net(seed=0):
    torch.manual_seed(seed)
    m = ActorCritic(hidden=16, obs_dim=MISSION_OBS_DIM, act_dim=12, obs_norm=True).eval()
    with torch.no_grad():
        m.actor_mean.weight.normal_()
    return m


def obs(n=50, seed=0):
    return np.random.default_rng(seed).normal(size=(n, MISSION_OBS_DIM)).astype(np.float32)


def test_signals_match_the_simulator():
    env = MissionEnv(2, seed=0, maps=[M.generate_map(1, "forest", 0.5), M.generate_map(2, "storm", 0.5)])
    O = env.reset()
    for _ in range(30):
        O = env.step(np.full((2, 12), 0.05))[0]
    s = concept_signals(O)
    offset, rate = swing_state(O)
    assert np.allclose(offset, (env.cp - env.p.mean(1))[:, :2], atol=1e-4)
    assert np.allclose(rate, (env.cv - env.v.mean(1))[:, :2], atol=1e-4)
    assert np.allclose(s["tension"][:, 0], env.tension.sum(1), atol=1e-3)
    assert np.allclose(s["obstacle"][:, 0], np.minimum(env._sense_all()[0], 3).min(1), atol=1e-4)


def test_occlusion_touches_only_its_concept():
    O = obs()
    mean_obs = O.mean(0)
    for concept, cols in (("tension", TENSION), ("obstacle", OBSTACLE)):
        changed = np.nonzero((occlude(O, concept, mean_obs) != O).any(0))[0]
        assert set(changed) == set(np.asarray(cols).tolist())
        assert np.allclose(occlude(O, concept, mean_obs)[:, cols], mean_obs[cols])


def test_swing_occlusion_removes_swing_and_keeps_the_formation():
    O = obs()
    mean_obs = O.mean(0)
    out = occlude(O, "swing", mean_obs)
    ref_offset, ref_rate = swing_state(mean_obs[None])
    offset, rate = swing_state(out)
    assert np.allclose(offset, ref_offset, atol=1e-5) and np.allclose(rate, ref_rate, atol=1e-5)
    shape = lambda A: A[:, REL] - A[:, REL].mean(1, keepdims=True)
    speeds = lambda A: A[:, VEL] - A[:, VEL].mean(1, keepdims=True)
    assert np.allclose(shape(out), shape(O), atol=1e-5) and np.allclose(speeds(out), speeds(O), atol=1e-5)


def test_history_columns_match_the_env():
    env = MissionEnv(2, seed=0, history=True, maps=[M.generate_map(1, "forest", 0.5), M.generate_map(2, "storm", 0.5)])
    frames = [env.reset()]
    for _ in range(40):
        frames.append(env.step(np.full((2, 12), 0.05))[0])
    now = frames[-1]
    assert np.allclose(now[:, PEAK_TENSION], env.tension_peak / 10, atol=1e-5)
    for i, lag in enumerate(HISTORY_LAGS):
        then = frames[-1 - lag]
        assert np.allclose(now[:, PAST_TENSION.reshape(-1, 3)[i]], then[:, TENSION], atol=1e-5)
        assert np.allclose(now[:, PAST_CARGO_VEL[i]], then[:, 3:5], atol=1e-5)
        assert np.allclose(now[:, PAST_OFFSET.reshape(-1, 2)[i]], swing_state(then)[0], atol=1e-4)


def test_occlusion_hides_the_concept_in_history_too():
    O = np.random.default_rng(0).normal(size=(50, MISSION_OBS_DIM + HISTORY_DIM)).astype(np.float32)
    mean_obs = O.mean(0)
    past = np.arange(MISSION_OBS_DIM, O.shape[1])

    hidden = occlude(O, "tension", mean_obs)
    cols = np.r_[PEAK_TENSION, PAST_TENSION]
    assert np.allclose(hidden[:, cols], mean_obs[cols])
    assert np.array_equal(hidden[:, np.setdiff1d(past, cols)], O[:, np.setdiff1d(past, cols)])

    hidden = occlude(O, "swing", mean_obs)
    assert np.allclose(hidden[:, PAST_OFFSET], mean_obs[PAST_OFFSET])
    assert np.allclose(hidden[:, PAST_CARGO_VEL], O[:, None, 3:5])
    cols = np.r_[PAST_OFFSET, PAST_CARGO_VEL.ravel()]
    assert np.array_equal(hidden[:, np.setdiff1d(past, cols)], O[:, np.setdiff1d(past, cols)])

    assert np.array_equal(occlude(O, "obstacle", mean_obs)[:, past], O[:, past])


def test_forward_matches_the_model():
    m, O = net(), obs()
    with torch.no_grad():
        want = m(torch.as_tensor(O))[0].clamp(-1, 1).numpy()
    assert np.allclose(forward(m, O)[2], want, atol=1e-6)


def test_ablating_a_whole_layer_gives_a_constant_output():
    m, O = net(), obs()
    h2 = forward(m, O)[1]
    idx = np.arange(h2.shape[1])
    out = forward(m, O, {2: (idx, h2.mean(0))})[2]
    assert np.allclose(out, out[0], atol=1e-6)
    assert not np.allclose(forward(m, O)[2], out)


def test_ablation_only_holds_the_chosen_neurons():
    m, O = net(), obs()
    h2 = forward(m, O)[1]
    got = forward(m, O, {2: (np.array([1, 3]), np.array([0.5, -0.5]))})[1]
    assert np.all(got[:, 1] == 0.5) and np.all(got[:, 3] == -0.5)
    assert np.allclose(np.delete(got, [1, 3], 1), np.delete(h2, [1, 3], 1))


def test_tuning_finds_a_planted_neuron():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(2000, 2))
    H = rng.normal(size=(2000, 6))
    H[:, 4] = y[:, 1] + 0.1 * rng.normal(size=2000)     # tracks only the second signal
    score = tuning(H, y)
    assert score.argmax() == 4 and score[4] > 0.95 and np.delete(score, 4).max() < 0.05


def test_matched_control_picks_similar_other_neurons():
    influence = np.array([1.0, 1.1, 1.2, 50.0, 51.0, 52.0, 53.0, 54.0])
    chosen = np.array([3, 4])
    picks = matched_control(influence, chosen, np.random.default_rng(0), pool=3)
    assert len(set(picks.tolist())) == 2 and not set(picks.tolist()) & {3, 4}
    assert all(p >= 5 for p in picks)


def test_untrained_copy_keeps_normalization_but_not_weights():
    m = net()
    m.obs_mean += 2
    fresh = untrained_copy(m, 1)
    assert torch.equal(fresh.obs_mean, m.obs_mean)
    assert not torch.equal(fresh.body[0].weight, m.body[0].weight)


def test_collect_keeps_only_flown_steps():
    maps = [M.generate_map(9, "storm", 0.0), M.generate_map(4, "forest", 0.0)]
    O, map_id, results = collect(MissionFormationController(), maps)
    assert [int((map_id == i).sum()) for i in range(2)] == [r.steps for r in results]
    assert len(O) == len(map_id) and results[0].steps != results[1].steps


def test_unablated_policy_matches_the_saved_policy(tmp_path):
    m = net()
    path = tmp_path / "p.pt"
    torch.save({"config": m.config, "state_dict": m.state_dict(), "residual": 0.1}, path)
    O = obs(5)
    O[:, :] = np.clip(O, -1, 1)
    assert np.allclose(ablated_policy(m, 0.1).batched(O), load_mission_policy(str(path)).batched(O), atol=1e-6)
