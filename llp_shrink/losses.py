"""LLP-Shrink equations (1)-(7). Targets and membership never receive gradients.

The class-specific residual vectors are NOT normalized over classes: each
coordinate is computed on a different set. See docs/METHOD.md.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    instance: torch.Tensor
    bag: torch.Tensor
    ratio: torch.Tensor
    selected: torch.Tensor
    pseudo: torch.Tensor
    confidence: torch.Tensor
    remaining: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    entropy: torch.Tensor
    weights: torch.Tensor


def _validate(weak, strong, counts, threshold):
    if weak.ndim != 3 or weak.shape != strong.shape:
        raise ValueError("logits must have matching [bags, instances, classes] shapes")
    if min(weak.shape) < 1 or weak.shape[-1] < 2:
        raise ValueError("empty bags or fewer than two classes")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if counts.shape != (weak.shape[0], weak.shape[2]):
        raise ValueError("counts must have shape [bags, classes]")
    if counts.is_floating_point() or counts.dtype == torch.bool:
        raise ValueError("counts must be integer class counts, not rounded proportions")
    if counts.requires_grad or (counts < 0).any() or (counts.sum(-1) != weak.shape[1]).any():
        raise ValueError("nonnegative counts must sum to the bag size")
    if not torch.isfinite(weak).all() or not torch.isfinite(strong).all():
        raise FloatingPointError("non-finite logits; stopping before optimizer update")


@torch.no_grad()
def quota_selection(probabilities, counts, threshold):
    """Strict > tau; descending confidence; stable original position breaks ties.

    The per-class quota is the observed bag count, never an individual label.
    Recomputed every forward pass; no permanent deletion across epochs.
    """
    confidence, pseudo = probabilities.max(-1)
    order = confidence.argsort(dim=1, descending=True, stable=True)
    sorted_labels = pseudo.gather(1, order)
    one_hot = F.one_hot(sorted_labels, probabilities.shape[-1])
    class_rank = one_hot.cumsum(1).gather(2, sorted_labels.unsqueeze(-1)).squeeze(-1)
    quotas = counts.gather(1, sorted_labels)
    sorted_selected = (confidence.gather(1, order) > threshold) & (class_rank <= quotas)
    selected = torch.zeros_like(sorted_selected).scatter(1, order, sorted_selected)
    return selected, pseudo, confidence


def shrink_loss(weak, strong, counts, threshold=0.95):
    _validate(weak, strong, counts, threshold)
    # Float32 for AMP, retain float64 for formula/gradient tests.
    dtype = torch.float64 if weak.dtype == torch.float64 else torch.float32
    logp = weak.to(dtype).log_softmax(-1)
    selected, pseudo, confidence = quota_selection(logp.detach().exp(), counts, threshold)
    selected_class = F.one_hot(pseudo, weak.shape[-1]).bool() & selected.unsqueeze(-1)
    keep = ~selected_class                         # B_i^k, shape [B, M, C]
    remaining = keep.sum(1)                        # M - selected_count_for_class_k
    removed = selected_class.sum(1)
    residual_counts = counts - removed
    valid = remaining > 0
    targets = residual_counts.to(dtype) / remaining.clamp_min(1)
    # Empty class-specific sets have zero residual count and zero contribution.
    # Avoid logsumexp(all -inf), whose backward would produce NaNs even at 0 weight.
    safe_keep = keep | (~valid).unsqueeze(1)
    log_prediction = torch.logsumexp(logp.masked_fill(~safe_keep, -torch.inf), dim=1)
    log_prediction = log_prediction - remaining.clamp_min(1).to(dtype).log()
    log_prediction = torch.where(valid, log_prediction, torch.zeros_like(log_prediction))
    bag = -(targets * log_prediction).sum(-1).mean()
    ce = F.cross_entropy(strong.to(dtype).flatten(0, 1), pseudo.flatten(), reduction="none")
    ce = ce.reshape_as(selected)
    # Eq. (2): average within each selected bag, THEN across all B bags.
    # An empty selected bag contributes zero and remains in the outer average.
    instance = ((ce * selected).sum(1) / selected.sum(1).clamp_min(1)).mean()
    ratio = selected.to(dtype).mean().detach()
    total = ratio * instance + (1 - ratio) * bag
    entropy = -(logp.detach().exp() * logp.detach()).sum(-1).mean() / math.log(2)
    return LossOutput(total, instance, bag, ratio, selected, pseudo, confidence,
                      remaining, targets.detach(), log_prediction.exp().detach() * valid,
                      entropy, selected.to(dtype))


def reference_loss(weak, strong, counts, method="dllp", beta_b=1.0, beta_i=1.0,
                   ahil_lambda=5.0, labels=None):
    """Reference objectives with separate method-specific hyperparameters.

    No claim of reproducing the complete baseline table. Supervised is an explicit
    oracle and the only method allowed to consume instance ground-truth labels.
    """
    _validate(weak, strong, counts, 0.95)
    if method not in {"dllp", "ahil", "supervised"}:
        raise ValueError("unknown reference method")
    if beta_b <= 0 or beta_i <= 0 or ahil_lambda < 0:
        raise ValueError("invalid AHIL hyperparameters")
    logp = weak.float().log_softmax(-1)
    p = logp.exp()
    b, m, c = p.shape
    targets = counts.float() / m
    log_mean = torch.logsumexp(logp, 1) - math.log(m)
    bag = -(targets * log_mean).sum(-1).mean()
    confidence, pseudo = p.detach().max(-1)
    weights = torch.zeros_like(confidence)
    if method == "ahil":
        with torch.no_grad():
            h_i = -(p * logp).sum(-1)
            class_log_dist = logp - torch.logsumexp(logp, 1, keepdim=True)
            h_b = -(class_log_dist.exp() * class_log_dist).sum(1)
            h_ref = counts.float().clamp_min(1).log()
            wb = torch.exp(-(h_b - h_ref).square() / beta_b)
            wb = wb * (counts > 0)  # absent-class reference has no distribution
            weights = torch.exp(-h_i.square() / beta_i) * wb.gather(1, pseudo)
        ce = F.cross_entropy(strong.float().flatten(0, 1), pseudo.flatten(), reduction="none")
        instance = (ce.reshape(b, m) * weights).mean()
        total = bag + ahil_lambda * instance
    elif method == "supervised":
        if labels is None or labels.shape != (b, m):
            raise ValueError("supervised mode requires [bags, instances] labels")
        instance = F.cross_entropy(weak.float().flatten(0, 1), labels.flatten())
        total = instance
    else:
        instance = strong.sum() * 0
        total = bag
    entropy = -(p.detach() * logp.detach()).sum(-1).mean() / math.log(2)
    return LossOutput(total, instance, bag, weights.mean().detach(), weights > 0,
                      pseudo, confidence, torch.full_like(counts, m), targets.detach(),
                      log_mean.exp().detach(), entropy, weights)
