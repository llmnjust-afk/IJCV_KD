"""Clean supervision candidates; no model forwards or random operations."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def transfer_clean_gate(gate, donor, receiver, ratio):
    """Move a bounded fraction of donor gate mass, preserving its batch sum."""
    original = gate.detach()
    supply = original[donor].sum()
    capacity = (1.0 - original[receiver]).sum()
    amount = torch.minimum(ratio * supply, capacity)
    if ratio == 0 or amount.item() <= 0:
        return original, original.new_zeros(())
    result = original.clone()
    result[donor] -= amount * original[donor] / supply
    result[receiver] += amount * (1.0 - original[receiver]) / capacity
    return result, amount


def clean_ce_loss(clean, labels, natural, gate, cfg, epoch):
    """Return CE before its historical weight/ramp, with per-view diagnostics.

    G2 freezes both operands of the old/new loss ratio. Its forward value equals
    the original CE up to rounding; its gradient follows the rescaled new CE.
    """
    gate = gate.detach()
    ce = F.cross_entropy(clean, labels, reduction='none')
    original = (gate * ce).mean()
    ratio = cfg.get('clean_ce_transfer_ratio', 0.0)
    active = ratio > 0 and cfg['clean_ce_weight'] > 0 and epoch > cfg['ce_start']
    with torch.no_grad():
        natural_correct = natural.detach().argmax(1).eq(labels)
        student_correct = clean.detach().argmax(1).eq(labels)
        donor = natural_correct & student_correct
        receiver = natural_correct & ~student_correct
    new_gate, amount = (transfer_clean_gate(gate, donor, receiver, ratio)
                        if active else (gate.detach(), gate.new_zeros(())))
    # Inactive, empty and zero-capacity cases retain the original loss graph.
    changed = amount.item() > 0
    redistributed = (new_gate * ce).mean() if changed else original
    scale = original.new_ones(())
    result = redistributed
    normalized = False
    if changed and cfg.get('clean_ce_preserve_loss', False):
        if original.detach().item() > 0 and redistributed.detach().item() > 0:
            scale = original.detach() / redistributed.detach()
            result = scale * redistributed
            normalized = True
        else:
            result, new_gate, amount = original, gate.detach(), gate.new_zeros(())
            redistributed = original
    with torch.no_grad():
        old_contribution = gate.detach() * ce.detach()
        new_contribution = new_gate * ce.detach()
        old_sum, new_sum = old_contribution.sum(), new_contribution.sum()
        receiver_count = receiver.sum().clamp_min(1)
        stats = {
            'ce_transfer_active': clean.new_tensor(float(active)),
            'ce_transfer_applied': clean.new_tensor(float(amount.item() > 0)),
            'ce_transfer_donor_fraction': donor.to(clean.dtype).mean(),
            'ce_transfer_receiver_fraction': receiver.to(clean.dtype).mean(),
            'ce_transfer_mass_per_sample': amount / labels.numel(),
            'ce_transfer_gate_mean_before': gate.detach().mean(),
            'ce_transfer_gate_mean_after': new_gate.mean(),
            'ce_transfer_receiver_gate_before': gate.detach()[receiver].sum() / receiver_count,
            'ce_transfer_receiver_gate_after': new_gate[receiver].sum() / receiver_count,
            'ce_transfer_receiver_effective_gate': scale * new_gate[receiver].sum() / receiver_count,
            'ce_transfer_ce_before': original.detach(),
            'ce_transfer_ce_redistributed': redistributed.detach(),
            'ce_transfer_ce_after': result.detach(),
            'ce_transfer_normalization_scale': scale,
            'ce_transfer_normalized': clean.new_tensor(float(normalized)),
            'ce_transfer_receiver_ce_share_before': (old_contribution[receiver].sum() / old_sum
                                                     if old_sum.item() > 0 else clean.new_zeros(())),
            'ce_transfer_receiver_ce_share_after': (new_contribution[receiver].sum() / new_sum
                                                    if new_sum.item() > 0 else clean.new_zeros(())),
            'ce_transfer_ce_share_defined': clean.new_tensor(float(old_sum.item() > 0 and new_sum.item() > 0)),
        }
    return result, stats


def natural_binary_loss(clean, natural, labels, temperature, cfg, epoch):
    """Extra true-vs-rest KL, masked then averaged over the full batch/classes.

    Teacher uses the current natural KD temperature, student uses temperature 1.
    All target and eligibility decisions are detached. This never replaces the
    original natural KD or the diagnostic losses used by update_dynamics.
    """
    ramp = min(1.0, max(0.0, (epoch - cfg.get('natural_binary_start', 120)) /
                             float(cfg.get('natural_binary_warmup', 40))))
    coefficient = cfg.get('natural_binary_weight', 0.0) * ramp
    zero = clean.new_zeros(())
    stats = {'natural_binary_active': clean.new_tensor(float(coefficient > 0)),
             'natural_binary_ramp': clean.new_tensor(ramp),
             'natural_binary_coefficient': clean.new_tensor(coefficient),
             'natural_binary_eligible_fraction': zero,
             'natural_binary_kl': zero,
             'natural_binary_term': zero,
             'natural_binary_logit_grad_norm': zero}
    if coefficient == 0:
        return zero, stats
    classes = clean.size(1)
    mask = F.one_hot(labels, num_classes=classes).bool()

    def binary_log_probs(logits):
        non = logits.masked_fill(mask, -torch.inf)
        return F.log_softmax(torch.stack((logits.gather(1, labels[:, None]).squeeze(1),
                                         torch.logsumexp(non, dim=1)), dim=1), dim=1)

    log_student = binary_log_probs(clean)
    with torch.no_grad():
        log_teacher = binary_log_probs(natural.detach() / temperature)
        teacher_probs = log_teacher.exp()
        eligible = (natural.detach().argmax(1).eq(labels)
                    & clean.detach().argmax(1).ne(labels)
                    & (log_teacher[:, 0] > log_student.detach()[:, 0]))
    # Log-space probabilities remain finite for finite logits, including saturated q.
    kl = (teacher_probs * (log_teacher - log_student)).sum(dim=1)
    masked_kl = torch.where(eligible, kl, torch.zeros_like(kl)).mean() / classes
    term = coefficient * masked_kl
    with torch.no_grad():
        # Analytic norm with respect to the existing clean logits, not parameters.
        true_grad = coefficient * eligible.to(clean.dtype) * (
            log_student[:, 0].exp() - teacher_probs[:, 0]) / (labels.numel() * classes)
        other_probs = F.softmax(clean.detach().masked_fill(mask, -torch.inf), dim=1)
        logit_grad = -true_grad[:, None] * other_probs
        logit_grad.scatter_(1, labels[:, None], true_grad[:, None])
        stats.update(natural_binary_eligible_fraction=eligible.to(clean.dtype).mean(),
                     natural_binary_kl=masked_kl.detach(),
                     natural_binary_term=term.detach(),
                     natural_binary_logit_grad_norm=logit_grad.norm())
    return term, stats
