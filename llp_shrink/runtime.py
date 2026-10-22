import copy
import json
import logging
import math
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def make_logger(output):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S")
    logpath = Path(output) / (datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f") + ".log")
    for handler in (logging.FileHandler(logpath, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def environment():
    import torchvision
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for target, source in zip(self.model.parameters(), model.parameters()):
            target.mul_(self.decay).add_(source, alpha=1 - self.decay)
        for target, source in zip(self.model.buffers(), model.buffers()):
            target.copy_(source)


def lr_factor(step, total, kind):
    ratio = min(max(step / total, 0), 1)
    if kind == "upstream":
        return math.cos(math.pi * ratio / 2)  # last active definition in utils.py
    if kind == "cosine":
        return 0.5 * (1 + math.cos(math.pi * ratio))
    if kind == "ahil-paper":
        return math.cos(7 * math.pi * ratio / 16)
    raise ValueError("unknown scheduler")


@torch.inference_mode()
def accuracy(model, loader, device):
    model.eval()
    correct, count = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite evaluation logits")
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    if count == 0:
        raise ValueError("empty evaluation dataset")
    return 100.0 * correct / count
