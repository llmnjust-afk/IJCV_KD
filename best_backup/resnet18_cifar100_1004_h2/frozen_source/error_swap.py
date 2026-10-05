"""Robust-KD-only error-label swap; no model forwards or random operations."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def robust_kd_logits(logits, labels, cfg, epoch):
    """From epoch 121, swap argmax/true logits on a detached KD-only copy.

    Swapping logits permutes their softmax probabilities at any positive
    temperature. Correct rows are unchanged; tied maxima can yield no change.
    """
    original = logits.detach()
    if not cfg.get('robust_kd_error_swap', False) or epoch < 121:
        return original
    predicted = original.argmax(dim=1, keepdim=True)
    truth = labels[:, None]
    corrected = original.clone()
    corrected.scatter_(1, truth, original.gather(1, predicted))
    corrected.scatter_(1, predicted, original.gather(1, truth))
    return corrected


@torch.no_grad()
def error_swap_metrics(logits, labels, cfg, epoch, temperature):
    """Report eligibility and the actual pre-mix robust target probability change.

    L1 is averaged over all samples after temperature softening, before the
    unchanged clean-target mix/split. It is not a final loss or gradient metric.
    """
    active = cfg.get('robust_kd_error_swap', False) and epoch >= 121
    eligible = logits.detach().argmax(dim=1).ne(labels)
    target_l1 = logits.new_zeros(())
    if active:
        corrected = robust_kd_logits(logits, labels, cfg, epoch)
        target_l1 = (F.softmax(corrected / temperature, dim=1)
                     - F.softmax(logits.detach() / temperature, dim=1)).abs().sum(dim=1).mean()
    return {'els_active': logits.new_tensor(float(active)),
            'els_eligible_fraction': eligible.to(logits.dtype).mean(),
            'els_robust_target_l1': target_l1}
