"""G3-anchored binary/conditional KD correction; no model forwards or RNG."""
import torch
import torch.nn.functional as F


def split_target_mix_loss(base_terms, student_logits, adv_logits, clean_logits,
                          labels, reference_probs, temperature, reference_alpha,
                          target_alpha, nontarget_alpha):
    """Return [batch, classes] terms, preserving the historical class-mean scale.

    Add [B(target)-B(reference) + w*(N(non)-N(reference))]/C to each
    sample's class mean. w is the reference target's non-target mass.
    The original base_terms retain CIARD's log(q+1e-5) convention.
    """
    classes = student_logits.size(1)
    mask = F.one_hot(labels, num_classes=classes).bool()
    with torch.no_grad():
        correct = clean_logits.detach().argmax(dim=1).eq(labels)
        log_adv = F.log_softmax(adv_logits.detach() / temperature, dim=1)
        log_clean = F.log_softmax(clean_logits.detach() / temperature, dim=1)

        def teacher_parts(alpha):
            weight = correct.to(log_adv.dtype).unsqueeze(1) * alpha
            # Zero weights yield -inf in log-space, but no teacher graph is built.
            log_mix = torch.logaddexp(log_adv + torch.log1p(-weight),
                                     log_clean + torch.log(weight))
            non = log_mix[~mask].reshape(-1, classes - 1)
            binary = torch.stack((log_mix[mask].exp(),
                                  torch.logsumexp(non, dim=1).exp()), dim=1)
            return binary, F.softmax(non, dim=1)

        reference = reference_probs.detach()
        # Sum the other probabilities; do not subtract an almost-one p_y.
        other_mass = reference[~mask].reshape(-1, classes - 1).sum(dim=1)
        reference_binary = torch.stack((reference[mask], other_mass), dim=1)
        _, reference_non = teacher_parts(reference_alpha)
        target_binary = (reference_binary if target_alpha == reference_alpha
                         else teacher_parts(target_alpha)[0])
        target_non = (reference_non if nontarget_alpha == reference_alpha
                      else teacher_parts(nontarget_alpha)[1])

    student_non = student_logits[~mask].reshape(-1, classes - 1)
    log_student_non = F.log_softmax(student_non, dim=1)
    student_binary = torch.stack((student_logits[mask],
                                  torch.logsumexp(student_non, dim=1)), dim=1)
    log_student_binary = F.log_softmax(student_binary, dim=1)
    binary_ref = F.kl_div(log_student_binary, reference_binary, reduction='none').sum(dim=1)
    binary_target = F.kl_div(log_student_binary, target_binary, reduction='none').sum(dim=1)
    non_ref = F.kl_div(log_student_non, reference_non, reduction='none').sum(dim=1)
    non_target = F.kl_div(log_student_non, target_non, reduction='none').sum(dim=1)
    correction = (binary_target - binary_ref + other_mass * (non_target - non_ref)) / classes
    # Preserve the exact old terms for ineligible samples, even after rounding.
    terms = torch.where(correct.unsqueeze(1), base_terms + correction.unsqueeze(1), base_terms)
    stats = {
        'binary_reference': binary_ref.detach().mean(),
        'binary_target': binary_target.detach().mean(),
        'conditional_reference': non_ref.detach().mean(),
        'conditional_target': non_target.detach().mean(),
        'reference_other_mass': other_mass.mean(),
        'loss_correction': torch.where(correct, correction.detach(), torch.zeros_like(correction)).mean(),
    }
    return terms, stats
