# LLP-Shrink

## Learning from Label Proportions with Shrinking Bag

**Xueyou Lu, Yueyi Wang, Wenxu Gao**

Implementation of **LLP-Shrink**, a learning-from-label-proportions method that couples high-confidence pseudo-label supervision with class-specific residual bag constraints.

Given disjoint bags and their class proportions, LLP-Shrink selects confident instances subject to per-class count limits. The selected instances receive instance-level supervision, while class-specific residual sets provide bag-level supervision. The two losses are combined using the fraction of selected instances in each minibatch.

[Method](docs/METHOD.md) · [Experiments](docs/EXPERIMENTS.md) · [GPU training](docs/AUTODL.md) · [Tests](docs/TESTING.md)

## Installation

Use Python 3.9–3.11 with PyTorch 2.6 and torchvision 0.21. Install the PyTorch build appropriate for your GPU driver; see the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v260).

```bash
git clone https://github.com/iiiyouhb/LLP-Shrink.git
cd LLP-Shrink
python -m venv .venv
source .venv/bin/activate
# Example: CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

For CPU-only use, replace `cu124` with `cpu`. On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

## Datasets

The implementation supports **CIFAR-10**, **SVHN**, and **Fashion-MNIST** through torchvision. Pass `--download` to download them into `--root`. SVHN uses the standard train/test splits without the extra split.

Training images are partitioned into fixed, disjoint bags. Instance labels are aggregated into bag-level class counts; the LLP-Shrink objective uses these counts rather than instance labels. Bag indices and dataset fingerprints are saved with each run.

## Training

```bash
# CIFAR-10, bag size 16
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 16 --root data --download --device cuda

# CIFAR-10, bag size 128
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 128 --root data --download --device cuda

# Other datasets
python -m llp_shrink.train --config configs/paper.json --dataset SVHN --bag-size 16 --root data --download --device cuda
python -m llp_shrink.train --config configs/paper.json --dataset FashionMNIST --bag-size 128 --root data --download --device cuda
```

| Setting | Default |
| --- | --- |
| Classifier | WRN-28-2 |
| Bag size | 16 or 128 |
| Instances per minibatch | 1024 |
| Training epochs | 1024 |
| Confidence threshold | 0.95 |
| Optimizer | SGD with Nesterov momentum 0.9 |
| Initial learning rate | 0.03 |
| Weight decay | 0.0005; BatchNorm parameters excluded |
| EMA decay | 0.999 |
| Random seed | 3 |

`--batch-instances` counts original instances. Weak and strong views are concatenated, so the network processes twice as many images. See [experiment settings](docs/EXPERIMENTS.md) for the learning-rate schedule, augmentation, and reporting conventions.

## Evaluation and resuming

```bash
python -m llp_shrink.evaluate runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --root data --device cuda
python -m llp_shrink.evaluate runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --root data --device cuda --ema
python -m llp_shrink.train --resume runs/CIFAR10/shrink_M16_s3_tau0.95/last.pth --device cuda
```

Checkpoints contain the model, EMA model, optimizer, scheduler, random states, and training history. Resume starts at the next completed-epoch boundary. Use a new `--output` directory for a separate experiment.

## Experiment suites

```bash
# Preview the six dataset/bag-size configurations
python scripts/run_experiments.py --suite main
# Execute the suite; resume incomplete runs and skip completed ones
python scripts/run_experiments.py --suite main --root data --download --execute --resume
# Multiple seeds
python scripts/run_experiments.py --suite main --seeds 1 2 3 --root data --execute --resume
# CIFAR-10 threshold sweep for both bag sizes
python scripts/run_experiments.py --suite threshold --thresholds 0 0.25 0.5 0.75 0.95 0.96 0.98 1 --execute --resume
python scripts/summarize.py runs --output experiment_summary.json
```

The sweep is configurable. The summary includes completed runs, excludes synthetic tests by default, and reports final-epoch mean and sample standard deviation across seeds. A single seed has no estimated standard deviation.

## Outputs

| File | Contents |
| --- | --- |
| `*.log` | Epoch losses, accuracy, EMA accuracy, selection ratio, and learning rate |
| `metrics.jsonl` | Structured epoch metrics |
| `config.json`, `environment.json` | Configuration and runtime environment |
| `bags.npz`, `data_manifest.json` | Bag membership, counts, data fingerprints, and remainder statistics |
| `last.pth` | Latest completed-epoch checkpoint |
| `best_raw.pth`, `best_ema.pth` | Checkpoints with the highest observed test accuracy |
| `model_final.pt`, `ema_final.pt` | Final model state dictionaries |
| `summary.json` | Completion status and final/best metrics |

`--export-weights-every N` additionally saves instance IDs, pseudo-labels, confidence, binary selection weights, and minibatch mixing ratios. `--audit-labels` enables pseudo-label accuracy diagnostics without using instance labels in the LLP objective.

Best-test checkpoints are descriptive records, not validation-selected estimates. Pretrained checkpoints and full-run result tables are not included in this release.

## Tests

```bash
python -m unittest discover -s tests -v
python -m llp_shrink.train --config configs/smoke.json --output runs/smoke
python -m llp_shrink.evaluate runs/smoke/last.pth --device cpu --ema
```

The smoke configuration uses synthetic data and a small network. See [testing](docs/TESTING.md) for coverage and a real-data preflight command.

## Acknowledgements

Third-party components and their source references are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
