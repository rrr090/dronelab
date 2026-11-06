"""Look inside trained mission policies: what do they track, and does it matter?

Three questions per checkpoint, for three concepts (cargo swing, rope tension,
obstacles). All three can be computed exactly from the observation, so finding
them in the network is not surprising by itself. The controls are what count:

1. Use. Replace a concept's inputs with their average: how much does the output change?
2. Tuning. How much of each neuron's activity do the concept's signals explain?
   Untrained networks on the same flights give the chance level, since random
   features of the inputs also correlate with those inputs.
3. Cause. Replace the most tuned neurons with their average activity. Does the
   network lose its response to that concept specifically? Controls are other
   neurons with the same effect on the output when removed one at a time.
   ``--closed-loop`` also flies the ablated networks through unseen maps.

Neurons are chosen on some maps and scored on others, and closed-loop maps are
separate from the training, validation and test maps.

Usage:
    dronelab mission interpret --checkpoint a.pt b.pt c.pt
    dronelab mission interpret --checkpoint a.pt --closed-loop --k 16 --eval-per 6
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dronelab.mission.env import MISSION_OBS_DIM, compose_residual
from dronelab.mission.maps import THEMES
from dronelab.mission.testing import (
    bootstrap_ci, load_mission_policy, mission_maps, run_missions, summarize_missions,
)
from dronelab.policy import ActorCritic

PROBE_SEED = 6_000_000           # maps for choosing neurons; test maps use 5e6
CLOSED_LOOP_SEED = 7_000_000     # maps for flying ablated networks
CONCEPTS = ("swing", "tension", "obstacle")
LAYERS = (1, 2)

# Observation layout, see CargoEnv.observe and MissionEnv.observe.
CARGO_VEL = slice(3, 6)
DRONE0, DRONE_W = 6, 19
REL = np.array([[DRONE0 + DRONE_W * d + i for i in range(3)] for d in range(3)])
VEL = REL + 3
TENSION = np.array([DRONE0 + DRONE_W * d + 18 for d in range(3)])
BODY_DIST = np.array([72, 76, 80, 84])
ROPE_DIST = np.array([88, 89, 90])
OBSTACLE = np.arange(72, 91)
SENSE_RANGE = 3.0


# ------------------------------------------------------------------ concepts

def swing_state(O):
    """Horizontal cargo offset from the team centre [m] and relative velocity [m/s]."""
    offset = -O[:, REL].mean(1)[:, :2]
    rate = 2 * (O[:, CARGO_VEL] - O[:, VEL].mean(1))[:, :2]
    return offset, rate


def concept_signals(O: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step signals for each concept: (N, columns), first column is the headline value.

    Signed components are included because a neuron can track sideways swing
    while barely correlating with swing size.
    """
    offset, rate = swing_state(O)
    tension = 10 * O[:, TENSION]
    dist = SENSE_RANGE * O[:, np.r_[BODY_DIST, ROPE_DIST]]
    return {
        "swing": np.c_[np.linalg.norm(offset, axis=1), np.linalg.norm(rate, axis=1), offset, rate],
        "tension": np.c_[tension.sum(1), tension],
        "obstacle": np.c_[dist[:, :4].min(1), dist],
    }


def occlude(O: np.ndarray, concept: str, mean_obs: np.ndarray) -> np.ndarray:
    """Copy of ``O`` with one concept's information replaced by its average.

    Swing lives in the team's shared offset and velocity relative to the cargo,
    so those are shifted to their average while the formation shape is kept.
    """
    out = O.copy()
    if concept == "tension":
        out[:, TENSION] = mean_obs[TENSION]
    elif concept == "obstacle":
        out[:, OBSTACLE] = mean_obs[OBSTACLE]
    elif concept == "swing":
        offset, rate = swing_state(O)
        ref_offset, ref_rate = swing_state(mean_obs[None])
        out[:, REL[:, :2]] += (offset - ref_offset)[:, None]
        out[:, VEL[:, :2]] += ((rate - ref_rate) / 2)[:, None]
    else:
        raise ValueError(f"unknown concept {concept}")
    return out


# ------------------------------------------------------------------ network access

@torch.no_grad()
def forward(model: ActorCritic, O: np.ndarray, ablate: dict | None = None):
    """Run the actor, returning (h1, h2, output clipped to the valid action range).

    ``ablate`` maps a layer to (neuron indices, replacement values).
    """
    ablate = ablate or {}
    x = model.normalize(torch.as_tensor(O, dtype=torch.float32))
    hidden = []
    for layer, (lin, act) in zip(LAYERS, ((model.body[0], model.body[1]), (model.body[2], model.body[3]))):
        x = act(lin(x))
        if layer in ablate:
            idx, values = ablate[layer]
            x = x.clone()
            x[:, idx] = torch.as_tensor(np.asarray(values), dtype=x.dtype)
        hidden.append(x)
    return hidden[0].numpy(), hidden[1].numpy(), model.actor_mean(x).clamp(-1, 1).numpy()


def ablated_policy(model: ActorCritic, residual: float, ablate: dict | None = None):
    """A mission policy that flies with the given neurons held at fixed values."""
    class Policy:
        history = model.config["obs_dim"] > MISSION_OBS_DIM

        @staticmethod
        def batched(O):
            O = np.asarray(O)
            a = forward(model, O, ablate)[2]
            return compose_residual(O, a, residual) if residual > 0 else a
    return Policy()


def untrained_copy(model: ActorCritic, seed: int) -> ActorCritic:
    """A fresh network with the same input normalization as ``model``."""
    torch.manual_seed(seed)
    fresh = ActorCritic(**model.config)
    state = fresh.state_dict()
    state.update({k: v for k, v in model.state_dict().items() if k.startswith("obs_")})
    fresh.load_state_dict(state)
    return fresh.eval()


def collect(policy, maps, cruise: float = 1.0):
    """Fly ``maps``; returns observations (N, obs), the map index of each, and the results.

    Finished missions keep stepping until the whole batch ends, so each map
    keeps only the steps it actually flew.
    """
    frames = []

    class Recorder:
        history = getattr(policy, "history", False)

        @staticmethod
        def batched(O):
            frames.append(np.array(O))
            return policy.batched(O)

    results = run_missions(Recorder, maps, cruise=cruise)
    frames = np.stack(frames)
    keep = [frames[: r.steps, i] for i, r in enumerate(results)]
    return np.concatenate(keep), np.repeat(np.arange(len(maps)), [len(k) for k in keep]), results


# ------------------------------------------------------------------ measurements

def tuning(H: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """R^2 of each neuron's activity (columns of H) explained linearly by ``signals``."""
    Hc = H - H.mean(0)
    Y = signals - signals.mean(0)
    Y = Y[:, Y.std(0) > 1e-9]
    Q, _ = np.linalg.qr(Y)
    return ((Q.T @ Hc) ** 2).sum(0) / np.maximum((Hc ** 2).sum(0), 1e-12)


def probe_r2(train_X, train_y, test_X, test_y, ridge: float = 1e-2) -> float:
    """Held-out R^2 of a linear readout."""
    mu, sd = train_X.mean(0), train_X.std(0) + 1e-6
    A = np.c_[(train_X - mu) / sd, np.ones(len(train_X))]
    B = np.c_[(test_X - mu) / sd, np.ones(len(test_X))]
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ train_y)
    return float(1 - ((test_y - B @ w) ** 2).sum() / ((test_y - train_y.mean()) ** 2).sum())


def sensitivity(model, O, mean_obs, ablate=None) -> dict[str, float]:
    """Mean squared output change when each concept is occluded."""
    base = forward(model, O, ablate)[2]
    return {c: float(((base - forward(model, occlude(O, c, mean_obs), ablate)[2]) ** 2).sum(1).mean())
            for c in CONCEPTS}


def neuron_influence(model, O, layer: int, means: np.ndarray) -> np.ndarray:
    """Mean squared output change when each neuron alone is held at its average."""
    base = forward(model, O)[2]
    return np.array([((base - forward(model, O, {layer: ([j], means[j:j + 1])})[2]) ** 2).sum(1).mean()
                     for j in range(len(means))])


def matched_control(influence: np.ndarray, chosen: np.ndarray, rng, pool: int = 8) -> np.ndarray:
    """Other neurons with about the same influence as ``chosen``, one for each."""
    free = [j for j in np.argsort(influence) if j not in set(chosen.tolist())]
    picks = []
    for j in chosen[np.argsort(-influence[chosen])]:
        nearest = sorted(free, key=lambda f: abs(np.log(influence[f] + 1e-12) - np.log(influence[j] + 1e-12)))
        pick = nearest[rng.integers(min(pool, len(nearest)))]
        picks.append(pick)
        free.remove(pick)
    return np.array(picks)


def weight_spectrum(model) -> dict:
    """Singular values of the hidden weights, scaled so the initial values are all 1."""
    out = {}
    for layer, lin in zip(LAYERS, (model.body[0], model.body[2])):
        s = torch.linalg.svdvals(lin.weight.detach()).numpy() / 2 ** 0.5
        out[f"h{layer}"] = {"min": float(s.min()), "median": float(np.median(s)), "max": float(s.max())}
    return out


def clipped_share(model, O) -> dict[str, float]:
    """Share of samples past the +-10 normalization clamp, for the tension inputs and overall."""
    z = (torch.as_tensor(O) - model.obs_mean) / torch.sqrt(model.obs_var + 1e-8)
    over = (z.abs() > 10).float().mean(0).numpy()
    return {"tension": float(over[TENSION].mean()), "all_inputs": float(over.mean()), "worst_input": int(over.argmax()),
            "worst_share": float(over.max())}


def analyse(model, O, map_id, ks=(4, 8, 16, 32), n_null: int = 10, n_controls: int = 20, seed: int = 0) -> dict:
    """Use, tuning and offline ablation for one network. Neurons are chosen on 4/5 of
    the maps and everything downstream of that choice is scored on the other 1/5."""
    rng = np.random.default_rng(seed)
    held_out = map_id % 5 == 0
    fit, test = O[~held_out], O[held_out]
    mean_obs = fit.mean(0)
    signals = concept_signals(fit)
    test_signals = concept_signals(test)
    fit_h = dict(zip(LAYERS, forward(model, fit)[:2]))
    test_h = dict(zip(LAYERS, forward(model, test)[:2]))
    nulls = [untrained_copy(model, 1000 + i) for i in range(n_null)]
    null_h = [dict(zip(LAYERS, forward(n, fit)[:2])) for n in nulls]
    null_test_h = [dict(zip(LAYERS, forward(n, test)[:2])) for n in nulls]

    intact = sensitivity(model, test, mean_obs)
    variance = float(((forward(model, test)[2] - forward(model, test)[2].mean(0)) ** 2).sum(1).mean())
    report = {
        "samples": len(O), "output_variance": variance,
        "use": {c: intact[c] / variance for c in CONCEPTS},
        "use_untrained": {c: float(np.mean([sensitivity(n, test, mean_obs)[c] / max(
            ((forward(n, test)[2] - forward(n, test)[2].mean(0)) ** 2).sum(1).mean(), 1e-12) for n in nulls]))
            for c in CONCEPTS},
        "weights": weight_spectrum(model), "clipped": clipped_share(model, O),
        "tuning": {}, "probes": {}, "ablation": {},
    }

    for concept in CONCEPTS:
        y, y_test = signals[concept][:, 0], test_signals[concept][:, 0]
        report["probes"][concept] = {
            "inputs": probe_r2(fit, y, test, y_test),
            **{f"h{l}": probe_r2(fit_h[l], y, test_h[l], y_test) for l in LAYERS},
            **{f"h{l}_untrained": float(np.mean([probe_r2(a[l], y, b[l], y_test)
                                                 for a, b in zip(null_h, null_test_h)])) for l in LAYERS},
        }
        report["tuning"][concept] = {}
        for layer in LAYERS:
            score = tuning(fit_h[layer], signals[concept])
            null_best = np.array([tuning(h[layer], signals[concept]).max() for h in null_h])
            report["tuning"][concept][f"h{layer}"] = {
                "best": float(score.max()), "null_best_mean": float(null_best.mean()),
                "null_best_max": float(null_best.max()),
                "above_null": int((score > null_best.max()).sum()),
                "top": np.argsort(-score)[:32].tolist(),
            }

    for layer in LAYERS:
        means = fit_h[layer].mean(0)
        influence = neuron_influence(model, test, layer, means)
        report["ablation"][f"h{layer}"] = {}
        for k in ks:
            per_k = {}
            for concept in CONCEPTS:
                chosen = np.array(report["tuning"][concept][f"h{layer}"]["top"][:k])
                kept = sensitivity(model, test, mean_obs, {layer: (chosen, means[chosen])})
                controls = []
                for _ in range(n_controls):
                    other = matched_control(influence, chosen, rng)
                    controls.append(sensitivity(model, test, mean_obs, {layer: (other, means[other])}))
                per_k[concept] = {
                    "kept": {c: kept[c] / intact[c] for c in CONCEPTS},
                    "control_kept": {c: float(np.mean([s[c] for s in controls])) / intact[c] for c in CONCEPTS},
                    "control_low": {c: float(np.quantile([s[c] for s in controls], 0.025)) / intact[c]
                                    for c in CONCEPTS},
                    "control_high": {c: float(np.quantile([s[c] for s in controls], 0.975)) / intact[c]
                                     for c in CONCEPTS},
                }
            report["ablation"][f"h{layer}"][str(k)] = per_k
    return report


def closed_loop(model, residual, O, map_id, report, maps, layer=2, k=16, n_controls=2, seed=0) -> dict:
    """Fly each concept's top neurons ablated, plus influence-matched controls, on ``maps``."""
    rng = np.random.default_rng(seed)
    fit = O[map_id % 5 != 0]
    means = forward(model, fit)[layer - 1].mean(0)
    influence = neuron_influence(model, O[map_id % 5 == 0], layer, means)
    intact = run_missions(ablated_policy(model, residual), maps)
    out = {"intact": summarize_missions(intact)}

    def fly(idx):
        rs = run_missions(ablated_policy(model, residual, {layer: (idx, means[idx])}), maps)
        s = summarize_missions(rs)
        mean, low, high = bootstrap_ci([a.score - b.score for a, b in zip(rs, intact)])
        s["score_diff"] = {"mean": mean, "ci_low": low, "ci_high": high}
        s["min_clearance"] = float(np.mean([r.min_clearance for r in rs]))
        return s

    for concept in CONCEPTS:
        chosen = np.array(report["tuning"][concept][f"h{layer}"]["top"][:k])
        out[concept] = {"ablated": fly(chosen),
                        "controls": [fly(matched_control(influence, chosen, rng)) for _ in range(n_controls)]}
        print(f"  {concept}: score {out[concept]['ablated']['mean_score']:.1f} vs intact "
              f"{out['intact']['mean_score']:.1f}, controls "
              + ", ".join(f"{c['mean_score']:.1f}" for c in out[concept]["controls"]))
    return out


# ------------------------------------------------------------------ reporting

def format_report(name: str, r: dict, k: int = 16) -> str:
    lines = [f"{name}: {r['samples']} steps",
             f"  {'concept':<10}{'use':>7}{'untrained':>11}   neurons above chance h1/h2 (best R2 vs chance)"]
    for c in CONCEPTS:
        t1, t2 = r["tuning"][c]["h1"], r["tuning"][c]["h2"]
        lines.append(f"  {c:<10}{r['use'][c]:>7.2f}{r['use_untrained'][c]:>11.2f}   "
                     f"{t1['above_null']:>3} / {t2['above_null']:<3} "
                     f"h1 {t1['best']:.2f} vs {t1['null_best_max']:.2f}, h2 {t2['best']:.2f} vs {t2['null_best_max']:.2f}")
    lines.append("  probe R2    inputs    h1 (untrained)    h2 (untrained)")
    for c in CONCEPTS:
        p = r["probes"][c]
        lines.append(f"  {c:<10}{p['inputs']:>8.2f}{p['h1']:>6.2f} ({p['h1_untrained']:.2f}){p['h2']:>12.2f} ({p['h2_untrained']:.2f})")
    for layer in ("h1", "h2"):
        lines.append(f"  ablate top {k} of {layer}: response kept to swing / tension / obstacle "
                     "[matched control mean]")
        for c in CONCEPTS:
            a = r["ablation"][layer][str(k)][c]
            lines.append(f"    {c:<10}" + "  ".join(
                f"{a['kept'][d]:.2f} [{a['control_kept'][d]:.2f}]" for d in CONCEPTS))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dronelab mission interpret", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--per", type=int, default=4, help="probing maps per theme x difficulty")
    p.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16, 32], help="neurons to ablate")
    p.add_argument("--nulls", type=int, default=10, help="untrained networks for the chance level")
    p.add_argument("--controls", type=int, default=20, help="matched neuron sets per offline ablation")
    p.add_argument("--closed-loop", action="store_true", help="also fly the ablated networks")
    p.add_argument("--k", type=int, default=16, help="neurons to ablate when flying")
    p.add_argument("--layer", type=int, default=2, choices=LAYERS, help="layer to ablate when flying")
    p.add_argument("--eval-per", type=int, default=6, help="flying maps per theme x difficulty")
    p.add_argument("--flown-controls", type=int, default=2, help="matched neuron sets to fly per concept")
    p.add_argument("--json", default="outputs/interpret.json")
    args = p.parse_args(argv)

    probe_maps = mission_maps(THEMES, (0.5, 1.0), args.per, PROBE_SEED)
    out = {}
    for path in args.checkpoint:
        name = os.path.splitext(os.path.basename(path))[0]
        policy = load_mission_policy(path)
        residual = float(torch.load(path, map_location="cpu").get("residual", 0.0))
        O, map_id, flown = collect(policy, probe_maps)
        report = analyse(policy.model, O, map_id, tuple(args.ks), args.nulls, args.controls)
        report["residual"] = residual
        report["delivered_rate"] = summarize_missions(flown)["delivered_rate"]
        print(format_report(name, report, args.k if args.k in args.ks else args.ks[0]) + "\n")
        if args.closed_loop:
            maps = mission_maps(THEMES, (0.5, 1.0), args.eval_per, CLOSED_LOOP_SEED)
            print(f"flying {len(maps)} unseen maps, top {args.k} of h{args.layer}")
            report["closed_loop"] = closed_loop(policy.model, residual, O, map_id, report, maps,
                                                args.layer, args.k, args.flown_controls)
        out[name] = report
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.json}")


if __name__ == "__main__":
    main()
