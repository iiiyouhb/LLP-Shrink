# Experiment settings

## Data and model

| Dataset | Train / test instances | Input shape | Classes |
| --- | --- | --- | --- |
| CIFAR-10 | 50,000 / 10,000 | 3×32×32 | 10 |
| SVHN | 73,257 / 26,032 | 3×32×32 | 10 |
| Fashion-MNIST | 60,000 / 10,000 | 1×28×28 | 10 |

All three datasets use WRN-28-2. Bags are fixed, random, disjoint, and have size 16 or 128. A minibatch contains 1024 original instances: 64 bags of size 16 or 8 bags of size 128.

## Optimization

`configs/paper.json` specifies 1024 epochs, threshold 0.95, SGD with Nesterov momentum 0.9, learning rate 0.03, and weight decay 0.0005. BatchNorm parameters are excluded from weight decay. EMA decay is 0.999. The default seed is 3; multiple seeds are explicitly passed to the experiment launcher.

The default scheduler, named `upstream` in the CLI, is:

```text
lr(t) = lr(0) * cos(pi*t / (2*T))
T = epochs * steps_per_epoch
```

Alternative schedules are `cosine`: `(1+cos(pi*t/T))/2`, and `ahil-paper`: `cos(7*pi*t/(16*T))`. Runs with different schedules are grouped separately when summarizing results.

## Data pipeline

Weak augmentation uses reflect-padded random crops and horizontal flips, with flips disabled for SVHN. Strong augmentation uses a separate crop/flip view, RandAugment with two sampled operations at magnitude 10, and Cutout. Images retain their native spatial resolution.

| Dataset | Mean | Standard deviation |
| --- | --- | --- |
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2471, 0.2435, 0.2616) |
| SVHN | (0.4380, 0.4440, 0.4730) | (0.1751, 0.1771, 0.1744) |
| Fashion-MNIST | (0.1307,) | (0.3081,) |

After randomly permuting all training instances, any remainder smaller than a full bag is recorded and excluded. The training loader shuffles bags each epoch and drops incomplete minibatches. This yields 48, 71, and 58 updates per epoch for CIFAR-10, SVHN, and Fashion-MNIST, respectively. The complete test split is evaluated, including its last partial batch.

## Suites and reporting

- `main`: the three datasets with bag sizes 16 and 128, using threshold 0.95.
- `threshold`: CIFAR-10 with both bag sizes and a configurable threshold list. The supplied list is a configurable evaluation grid, not an exact archived record of the paper's sweep.
- The launcher defaults to one seed, 3. Use `--seeds` for repeated trials.
- Raw and EMA test accuracy are logged separately. Summaries use final-epoch values and sample standard deviation across seeds.
- Best-test checkpoints are retained for inspection. They are not selected using an independent validation split.
- Instance selection is recomputed at each step. Individual labels are used only to create bag counts or, with `--audit-labels`, report diagnostics.

## Optional reference objectives

`--method dllp`, `--method ahil`, and `--method supervised` provide reference objectives in the shared training pipeline. AHIL defaults to `beta_b=beta_i=1` and `ahil_lambda=5`. The supervised mode alone uses instance labels in its objective; it shares the joint weak/strong BatchNorm forward with the other modes.

These entries do not constitute implementations of every baseline in the paper. LLPFC, LLP-VAT, and ROT are not included. This release supplies training configurations and software tests; pretrained checkpoints and complete experiment results are not bundled.
