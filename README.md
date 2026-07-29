# pinn-turbolence

Physics-informed neural networks for the Kolmogorov flow problem, i.e. the 2D incompressible Navier--Stokes equations.

## Repository Overview

- `src/`: core implementation for the network, losses, training loops, utilities
- `configs/`: YAML configurations for the different training stages and experiments
- `scripts/`: command-line entry points for training and evaluation
- `notebooks/train_colab.ipynb`: step-by-step Colab workflow
- `results/`: sample outputs, checkpoints, and evaluation artifacts

## Training and Evaluation

### 1. Baseline training

This is the default single-window PINN setup.

```bash
python scripts/train.py --config configs/kolmogorov.yaml
```

Outputs are written to `results/` and the final checkpoint is saved at `results/weights/baseline_final.pt`.

### 2. Causal training

The causal configuration splits time-ordered PDE points into disjoint chunks.
Each chunk receives an exponential weight based on the residuals of the previous chunks, following *Simulating Three-dimensional Turbulence with Physics-informed Neural Networks* (Wang et al., 2025).

```bash
python scripts/train.py --config configs/kolmogorov_causal.yaml
```

Outputs are written to `results/causal`. TensorBoard also logs `causal/min_weight` and `causal/mean_weight`, which are useful to track the progressive unlocking of later time chunks.

### 3. Adaptive weighting

This run keeps the baseline training setup but updates the IC/PDE loss weights during training using gradient norms.

```bash
python scripts/train.py --config configs/kolmogorov_adaptive.yaml
```

Outputs are written to `results/adaptive`.

### 4. Time marching

This experiment trains the model over successive time windows and transfers the learned state from one window to the next.

```bash
python scripts/train.py --config configs/kolmogorov_timemarching.yaml
```

Outputs are written to `results/time_marching`, with one folder per window such as `results/time_marching/window_00`.


### 6. Evaluation

Evaluate a saved checkpoint on a regular grid and save metrics plus velocity plots:

```bash
python scripts/evaluate.py --checkpoint results/baseline/weights/baseline_final.pt \
  --config configs/kolmogorov.yaml
```

The evaluation script saves `results/eval_metrics.json` and `results/eval_velocity.png` by default.

## Quick Start

1. Install the dependencies, preferably inside a virtual environment:

```bash
pip install -r requirements.txt
# optional: pip install tensorboard
```

2. Run one of the training configurations listed above.

3. Evaluate the resulting checkpoint with `scripts/evaluate.py`.

## Notes

- The notebook `notebooks/train_colab.ipynb` shows the full workflow step by step.
- Checkpoints save a copy of the YAML configuration used for the run.
- If TensorBoard is installed, you can launch it with:

```bash
tensorboard --logdir results/tensorboard
```

## Contributing

- Open an issue or a pull request.
- Add a new file under `configs/` for any new experiment.

License: MIT (see `LICENSE`).
