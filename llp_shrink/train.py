"""Training entrypoint; run with python -m llp_shrink.train --help."""
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import parse_args
from .data import (SPECS, BagDataset, BagPartition, EvalDataset, raw_dataset,
                   synthetic_dataset, fingerprint, train_loader, eval_loader)
from .losses import shrink_loss, reference_loss
from .model import WideResnet
from .runtime import (EMA, accuracy, atomic_torch_save, environment, lr_factor,
                      make_logger, restore_rng, rng_state, seed_all, write_json)


def build_model(cfg):
    return WideResnet(10, k=cfg.width, n=cfg.depth, proj=False,
                      in_channels=SPECS[cfg.dataset][0])


def _device(cfg):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check server torch/CUDA installation")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    if cfg.amp and device.type != "cuda":
        raise ValueError("AMP is opt-in and requires CUDA")
    return device


def _run(cfg, resume, stop_after_epoch, out, logger):
    torch.set_num_threads(cfg.threads)
    seed_all(cfg.seed, cfg.deterministic)
    device = _device(cfg)
    logger.info(asdict(cfg))
    logger.info("***** Running training *****")
    logger.info("  Task = %s | synthetic=%s | device=%s", cfg.dataset, cfg.synthetic, device)
    if cfg.synthetic:
        logger.warning("SYNTHETIC INTEGRATION TEST: these metrics are NOT paper results")
        images, labels = synthetic_dataset(cfg.dataset, cfg.synthetic_train, cfg.seed)
        test_images, test_labels = synthetic_dataset(cfg.dataset, cfg.synthetic_test, cfg.seed + 1)
    else:
        images, labels = raw_dataset(cfg.dataset, cfg.root, True, cfg.download)
        test_images, test_labels = raw_dataset(cfg.dataset, cfg.root, False, cfg.download)
    data_hash = {"train": fingerprint(images, labels), "test": fingerprint(test_images, test_labels)}
    partition = BagPartition.create(labels, cfg.bag_size, cfg.seed)
    labeled = labels if cfg.audit_labels or cfg.method == "supervised" else None
    ds = BagDataset(images, partition, cfg.dataset, labeled)
    test = EvalDataset(test_images, test_labels, cfg.dataset)
    batch_bags = cfg.batch_instances // cfg.bag_size
    steps = len(ds) // batch_bags
    if steps < 1:
        raise ValueError("dataset too small for one batch")
    total_steps = steps * cfg.epochs
    test_loader = eval_loader(test, cfg.eval_batch, cfg.workers, device.type == "cuda")
    metadata = {"fingerprints": data_hash, "train_size": len(images), "test_size": len(test),
                "bag_count": len(ds), "bag_size": cfg.bag_size,
                "partition_omitted": len(partition.omitted), "steps_per_epoch": steps,
                "epoch_omitted_bags": len(ds) % batch_bags, "synthetic": cfg.synthetic}
    logger.info("Data: %s", metadata)
    model = build_model(cfg).to(device)
    logger.info("Total params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    ema = EMA(model, cfg.ema_decay) if cfg.ema else None
    wd, no_wd = [], []
    for name, parameter in model.named_parameters():
        (no_wd if "bn" in name else wd).append(parameter)
    optim = torch.optim.SGD([{"params": wd}, {"params": no_wd, "weight_decay": 0.0}],
                            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                            nesterov=cfg.nesterov)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: lr_factor(step, total_steps, cfg.scheduler))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    start_epoch, global_step = 0, 0
    best = {"raw": -1.0, "raw_epoch": -1, "ema": -1.0, "ema_epoch": -1}
    history = []
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        allowed = {"root", "output", "device", "download", "log_every", "save_every", "export_weights_every"}
        changed = [k for k, v in asdict(cfg).items() if k not in allowed and ckpt["config"].get(k) != v]
        if changed:
            raise ValueError("resume cannot change training semantics: " + ", ".join(changed))
        if ckpt["data_fingerprints"] != data_hash or ckpt["steps_per_epoch"] != steps:
            raise ValueError("resume data/split mismatch")
        if not np.array_equal(ckpt["bag_indices"], partition.indices):
            raise ValueError("resume bag partition mismatch")
        model.load_state_dict(ckpt["model"], strict=True)
        if ema:
            ema.model.load_state_dict(ckpt["ema_model"], strict=True)
        optim.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["global_step"]
        best, history = ckpt["best"], ckpt["history"]
        restore_rng(ckpt["rng"])
        logger.info("Resumed epoch boundary: next epoch=%d, global_step=%d", start_epoch, global_step)
    elif (out / "config.json").exists():
        raise FileExistsError("run directory already contains config.json; choose another output or --resume")
    write_json(out / "config.json", asdict(cfg))
    write_json(out / "environment.json", environment())
    write_json(out / "data_manifest.json", metadata)
    partition.save(out / "bags.npz")
    # Authoritative metrics are checkpoint history; discard uncommitted epoch records on resume.
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8")
    stop_requested = []
    previous_signal = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.append(signum))
    logger.info("-----------start training--------------")
    try:
        for epoch in range(start_epoch, cfg.epochs):
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
            model.train()
            loader = train_loader(ds, batch_bags, cfg.workers, cfg.seed, epoch, device.type == "cuda")
            meters = dict(loss=0.0, loss_u=0.0, loss_prop=0.0, ratio=0.0, entropy=0.0, correct=0.0)
            selected_total, selected_correct = 0, 0
            epoch_start = time.monotonic()
            export = cfg.export_weights_every > 0 and ((epoch + 1) % cfg.export_weights_every == 0 or epoch + 1 == cfg.epochs)
            records = []
            for it, batch in enumerate(loader):
                weak, strong, counts = [batch[k].to(device, non_blocking=True) for k in ("weak", "strong", "counts")]
                b, m, c, h, w = weak.shape
                # Joint weak/strong forward; BatchNorm sees both views.
                inputs = torch.cat([weak.flatten(0, 1), strong.flatten(0, 1)])
                with torch.autocast(device_type=device.type, enabled=cfg.amp):
                    logits = model(inputs)
                weak_logits, strong_logits = [x.reshape(b, m, 10) for x in logits.chunk(2)]
                if cfg.method == "shrink":
                    result = shrink_loss(weak_logits, strong_logits, counts, cfg.threshold)
                else:
                    oracle = batch["audit_labels"].to(device) if cfg.method == "supervised" else None
                    result = reference_loss(weak_logits, strong_logits, counts, cfg.method,
                                            cfg.beta_b, cfg.beta_i, cfg.ahil_lambda, oracle)
                if not torch.isfinite(result.total):
                    raise FloatingPointError(f"non-finite loss at epoch={epoch}, iter={it}")
                optim.zero_grad(set_to_none=True)
                scaler.scale(result.total).backward()
                scaler.unscale_(optim)
                # This checks but does not clip gradients, preserving the paper update.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                global_step += 1
                if ema:
                    ema.update(model)
                for key, value in (("loss", result.total), ("loss_u", result.instance),
                                   ("loss_prop", result.bag), ("ratio", result.ratio), ("entropy", result.entropy)):
                    meters[key] += value.item()
                selected_total += result.selected.sum().item()
                if cfg.audit_labels:
                    truth = batch["audit_labels"].to(device)
                    correct = (result.pseudo == truth)
                    meters["correct"] += correct.sum().item()
                    selected_correct += (correct & result.selected).sum().item()
                if export:
                    records.append({"ids": batch["ids"].numpy(), "bag_id": batch["bag_id"].numpy(),
                                    "pseudo": result.pseudo.cpu().numpy(), "selected": result.selected.cpu().numpy(),
                                    "weights": result.weights.cpu().numpy(), "confidence": result.confidence.cpu().numpy(),
                                    "lambda": np.full(b, result.ratio.item(), dtype=np.float32)})
                if cfg.log_every and (it + 1) % cfg.log_every == 0:
                    logger.info("epoch:%d iter:%d/%d loss=%.6f lambda=%.6f", epoch, it + 1, steps,
                                result.total.item(), result.ratio.item())
            avg = {k: v / steps for k, v in meters.items()}
            seconds = time.monotonic() - epoch_start
            lr = optim.param_groups[0]["lr"]
            corr_text = f"{avg['correct']:.2f}" if cfg.audit_labels else "NA"
            logger.info("%s-s%d, LLP-%s | epoch:%d, iter: %d. loss_u: %.6f. loss_prop: %.6f.  "
                        "n_correct_u: %s.  LR: %.8f. Time: %.2f. Entropy: %.6f. Lambda: %.6f. Selected: %d.",
                        cfg.dataset, cfg.seed, cfg.method, epoch, steps, avg["loss_u"], avg["loss_prop"],
                        corr_text, lr, seconds, avg["entropy"], avg["ratio"], selected_total)
            raw_acc = accuracy(model, test_loader, device)
            ema_acc = accuracy(ema.model, test_loader, device) if ema else None
            improve_raw = raw_acc > best["raw"]
            improve_ema = ema_acc is not None and ema_acc > best["ema"]
            if improve_raw:
                best.update(raw=raw_acc, raw_epoch=epoch)
            if improve_ema:
                best.update(ema=ema_acc, ema_epoch=epoch)
            logger.info("Epoch %d. Acc: %.4f. Ema-Acc: %s. best_acc: %.4f in epoch %d", epoch,
                        raw_acc, f"{ema_acc:.4f}" if ema_acc is not None else "NA", best["raw"], best["raw_epoch"])
            row = dict(epoch=epoch, global_step=global_step, loss=avg["loss"], loss_u=avg["loss_u"],
                       loss_prop=avg["loss_prop"], entropy_bits=avg["entropy"], lambda_mean=avg["ratio"],
                       selected=selected_total, selected_precision=(selected_correct / selected_total
                       if cfg.audit_labels and selected_total else None), raw_acc=raw_acc, ema_acc=ema_acc,
                       lr=lr, seconds=seconds, synthetic=cfg.synthetic)
            history.append(row)
            if export:
                np.savez_compressed(out / f"sample_weights_epoch_{epoch:04d}.npz",
                                    **{k: np.concatenate([r[k] for r in records]) for k in records[0]})
            ckpt = {"format_version": 1, "model": model.state_dict(),
                    "ema_model": ema.model.state_dict() if ema else None, "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "global_step": global_step, "config": asdict(cfg), "best": best, "history": history,
                    "rng": rng_state(), "data_fingerprints": data_hash, "bag_indices": partition.indices,
                    "steps_per_epoch": steps}
            if improve_raw:
                atomic_torch_save(ckpt, out / "best_raw.pth")
            if improve_ema:
                atomic_torch_save(ckpt, out / "best_ema.pth")
            atomic_torch_save(ckpt, out / "last.pth")
            if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
                atomic_torch_save(ckpt, out / f"checkpoint_{epoch:04d}.pth")
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(out / "summary.json", {"completed_epochs": epoch + 1, "planned_epochs": cfg.epochs,
                       "complete": epoch + 1 == cfg.epochs, "synthetic": cfg.synthetic,
                       "final": row, "best_test_descriptive_only": best,
                       "selection_note": "No validation set specified in paper; best test is descriptive, not unbiased model selection."})
            if stop_requested:
                logger.info("SIGTERM: saved epoch boundary, stopping cleanly")
                break
        if history and history[-1]["epoch"] + 1 == cfg.epochs:
            atomic_torch_save(model.state_dict(), out / "model_final.pt")
            if ema:
                atomic_torch_save(ema.model.state_dict(), out / "ema_final.pt")
        logger.info("Saved %d/%d completed epochs to %s", len(history), cfg.epochs, out)
        return history
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


def run(cfg, resume=None, stop_after_epoch=None):
    cfg.validate()
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out / "config.json").exists():
        raise FileExistsError("existing run: choose another output or use --resume")
    lock = out / ".running.lock"
    # Exclusive create prevents two jobs corrupting one output directory.
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    logger = make_logger(out)
    try:
        return _run(cfg, resume, stop_after_epoch, out, logger)
    except BaseException:
        logger.exception("Training interrupted/failed; last.pth is the last completed epoch")
        raise
    finally:
        lock.unlink(missing_ok=True)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def main(argv=None):
    cfg, resume, stop_after_epoch = parse_args(argv)
    run(cfg, resume, stop_after_epoch)


if __name__ == "__main__":
    main()
