import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    dataset: str = "CIFAR10"
    method: str = "shrink"
    root: str = "data"
    output: str = ""
    bag_size: int = 16
    batch_instances: int = 1024
    epochs: int = 1024
    threshold: float = 0.95
    seed: int = 3
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 0.0005
    nesterov: bool = True
    scheduler: str = "upstream"
    depth: int = 28
    width: int = 2
    ema: bool = True
    ema_decay: float = 0.999
    workers: int = 4
    eval_batch: int = 128
    device: str = "auto"
    amp: bool = False
    deterministic: bool = True
    download: bool = False
    audit_labels: bool = False
    save_every: int = 100
    export_weights_every: int = 0
    log_every: int = 0
    threads: int = 4
    beta_b: float = 1.0
    beta_i: float = 1.0
    ahil_lambda: float = 5.0
    synthetic: bool = False
    synthetic_train: int = 32
    synthetic_test: int = 19

    def validate(self):
        if self.dataset not in {"CIFAR10", "SVHN", "FashionMNIST"}:
            raise ValueError("dataset must be CIFAR10, SVHN or FashionMNIST")
        if self.method not in {"shrink", "ahil", "dllp", "supervised"}:
            raise ValueError("unsupported method")
        for key in ("bag_size", "batch_instances", "epochs", "width", "eval_batch", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("workers", "save_every", "export_weights_every", "log_every"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.depth < 10 or (self.depth - 4) % 6:
            raise ValueError("WRN depth must be 6n+4 and at least 10")
        if self.batch_instances % self.bag_size:
            raise ValueError("batch_instances must be divisible by bag_size")
        for key in ("threshold", "lr", "momentum", "weight_decay", "ema_decay", "beta_b", "beta_i", "ahil_lambda"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.threshold <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("threshold or EMA decay out of range")
        if self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.momentum < 1:
            raise ValueError("invalid optimizer settings")
        if self.nesterov and self.momentum == 0:
            raise ValueError("Nesterov requires positive momentum")
        if self.beta_b <= 0 or self.beta_i <= 0 or self.ahil_lambda < 0:
            raise ValueError("invalid AHIL parameters")
        if self.scheduler not in {"upstream", "cosine", "ahil-paper"}:
            raise ValueError("unknown scheduler")
        if self.synthetic and (self.synthetic_train < self.batch_instances or self.synthetic_test < 1):
            raise ValueError("synthetic data must include a full train batch and nonempty test set")
        if not self.output:
            tag = "synthetic-" if self.synthetic else ""
            self.output = f"runs/{tag}{self.dataset}/{self.method}_M{self.bag_size}_s{self.seed}_tau{self.threshold:g}"
        return self


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LLP-Shrink training")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, help="trusted checkpoint created by this project")
    parser.add_argument("--stop-after-epoch", type=int, help="stop after this many completed epochs; keep full schedule")
    for key, default in asdict(Config()).items():
        args = {"default": None}
        if isinstance(default, bool):
            args["action"] = argparse.BooleanOptionalAction
        else:
            args["type"] = type(default)
        parser.add_argument("--" + key.replace("_", "-"), **args)
    ns = parser.parse_args(argv)
    values = {}
    if ns.resume:
        import torch
        # Resuming is explicitly limited to trusted local/project checkpoints.
        values = torch.load(ns.resume, map_location="cpu", weights_only=False)["config"].copy()
    if ns.config:
        values.update(json.loads(ns.config.read_text(encoding="utf-8")))
    for key in asdict(Config()):
        value = getattr(ns, key)
        if value is not None:
            values[key] = value
    cfg = Config(**values).validate()
    if ns.stop_after_epoch is not None and ns.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be >= 1")
    return cfg, ns.resume, ns.stop_after_epoch
