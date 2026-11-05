# drone-mech-interp

Training a drone hover policy with reinforcement learning, then opening up the
trained network with mechanistic interpretability to find out which neurons
do what, and what happens when we switch them off.

**Status: Step 1 — environment and policy scaffold.** Mech interp comes next.

## Roadmap

1. **Hover policy (this step).** 2D planar quadrotor, 6-dim state, 2 motor
   thrusts, PPO-trained 2×64 tanh MLP.
2. **Activation logging.** Record hidden-layer activations along trajectories.
3. **Simple neuron analysis.** Correlate neurons with state variables
   (height error, pitch, velocity).
4. **Ablation.** Zero or mean-ablate single neurons and measure the effect on
   hover success, crashes and tracking error.
5. **Visualisation.** Side-by-side video: flight path on the left, neuron
   activation heatmap on the right.
6. **Extension (later).** Move to a 3D drone model, then compare MLP
   to other architectures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the RTX 3070, install the CUDA build of torch from
https://pytorch.org/get-started/locally/ before the line above, so that
`torch.cuda.is_available()` returns `True`.

## Train

```bash
python train.py --updates 300
```

Checkpoint is saved to `checkpoints/hover_mlp.pt`.

## Test

```bash
pip install pytest
pytest -q
```

## Layout

```
drone_mi/env.py      planar quadrotor dynamics and reward
drone_mi/policy.py   actor-critic MLP
train.py             PPO training loop
tests/               environment sanity checks
```

## Notes

This is a learning project. Environment physics are simplified, and results
are for a single 2D drone, not a validated flight model.
