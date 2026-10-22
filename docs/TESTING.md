# Testing

```bash
python -W error::RuntimeWarning -W error::DeprecationWarning -m unittest discover -s tests -v
```

The test suite covers:

- Hand-calculated class-specific residual targets and losses, selection limits, ties, and strict thresholding.
- Empty selection/residual sets, absent classes, extreme logits, finite gradients, and double-precision gradient checking.
- Fixed disjoint bag partitions, sample/count alignment, three input formats, and multiprocessing data loading.
- Augmentation integer arithmetic, WRN parameter updates, EMA parameters and buffers, and exact evaluation denominators.
- Epoch-boundary resume equivalence on CPU, checkpoint and sample-weight export, and protection against overwriting existing runs.
- Separation of instance-label diagnostics from LLP training.

## End-to-end smoke test

```bash
python -m llp_shrink.train --config configs/smoke.json --output runs/smoke
python -m llp_shrink.evaluate runs/smoke/last.pth --device cpu --ema
```

This test uses synthetic images and WRN-10-1. Its metrics are not dataset benchmark results.

## Real-data preflight

```bash
python scripts/preflight.py --dataset FashionMNIST --root data --download --device cpu
```

The preflight performs a WRN-28-2 update using a small real-data minibatch and checks model reloading. Use `--device cuda` on the target GPU. A preflight does not measure convergence or full-run accuracy.

## Continuous integration

The [CPU workflow](../.github/workflows/test.yml) installs the package on Ubuntu, runs the test suite, trains the smoke configuration, and evaluates its saved EMA checkpoint. See [Actions](https://github.com/iiiyouhb/LLP-Shrink/actions) for current run status.

CPU tests do not establish CUDA/AMP correctness, GPU memory requirements, or identical results across hardware and package versions.
