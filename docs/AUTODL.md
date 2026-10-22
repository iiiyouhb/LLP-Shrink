# GPU training

## Environment

Check the GPU, driver, disk space, and PyTorch installation before starting a long run:

```bash
nvidia-smi
python --version
df -h
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Install a compatible PyTorch build and run `pip install -e .` from the repository root. On AutoDL, keep data and outputs on the persistent data disk, for example under `/root/autodl-tmp`.

## Preflight and one-epoch check

```bash
python -m unittest discover -s tests -v
python -m llp_shrink.train --config configs/smoke.json --device cuda --output runs/cuda-smoke
python scripts/preflight.py --dataset CIFAR10 --root /root/autodl-tmp/data --download --device cuda
python -m llp_shrink.train --config configs/paper.json --dataset CIFAR10 --bag-size 16 --root /root/autodl-tmp/data --download --device cuda --output /root/autodl-tmp/llp-runs/cifar10-m16 --stop-after-epoch 1 --log-every 10
```

`--stop-after-epoch 1` stops after one completed epoch while retaining the full planned schedule. Continue with:

```bash
python -m llp_shrink.train --resume /root/autodl-tmp/llp-runs/cifar10-m16/last.pth --device cuda
```

## Resource considerations

The standard minibatch has 1024 instances and 2048 augmented images in a joint forward. Measure memory use on the target device. AMP is opt-in through `--amp`; changes to batch size affect BatchNorm statistics and the minibatch selection ratio and should be reported.

Training uses one process and one GPU. Separate experiments can be assigned to different GPUs using `CUDA_VISIBLE_DEVICES`, with distinct output directories. Distributed data parallel training is not implemented.

## Long runs and recovery

Run inside a terminal multiplexer or use `nohup`:

```bash
mkdir -p /root/autodl-tmp/llp-runs
nohup python -m llp_shrink.train --resume /root/autodl-tmp/llp-runs/cifar10-m16/last.pth --device cuda > /root/autodl-tmp/llp-runs/launcher.log 2>&1 &
echo $!
```

Keep the process ID and monitor the run's `.log` and `summary.json`. SIGTERM requests a clean stop at the end of the current epoch. Forced termination can lose the incomplete epoch; `last.pth` records the latest completed epoch.

After an abnormal exit, remove `.running.lock` only after confirming that the corresponding process has stopped. Resume from a trusted checkpoint. Retain the full run directory, including logs, configuration, bag membership, and checkpoints.
