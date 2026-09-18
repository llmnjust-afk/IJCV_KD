"""CIARD 0909v1: KD-conditioned AWP and adversarial two-view consistency.

References: Wu et al., NeurIPS 2020 (AWP); Tack et al., AAAI 2022
(Consistency Regularization for Adversarial Robustness). This is an adaptation,
not a reproduction of either paper's full recipe. No evaluation attacks live here.
"""
import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from mtard_loss import soft_feature_push_loss, ema_update_teacher


class EpochViews(torch.utils.data.Dataset):
    """Preserve the exact original __getitem__ call when one view is requested."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.views = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        first, label = self.dataset[index]
        if self.views == 1:
            return first, label
        second, _ = self.dataset[index]
        return [first, second], label


def method_ramp(cfg, epoch):
    return min(1.0, max(0.0, (epoch - cfg['method_start']) / float(cfg['method_warmup'])))


def method_enabled(cfg, epoch):
    return epoch > cfg['method_start'] and (
        cfg['training_views'] == 2 or cfg['awp_gamma'] > 0 or cfg['consistency_weight'] > 0)


def validate_config(cfg):
    if cfg['training_views'] not in (1, 2) or cfg['method_warmup'] <= 0:
        raise ValueError('Expected one/two views and a positive method warmup')
    for key in ('awp_gamma', 'consistency_weight', 'consistency_temperature'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid method parameter: ' + key)
    if cfg['consistency_temperature'] == 0 or (cfg['consistency_weight'] > 0 and cfg['training_views'] != 2):
        raise ValueError('Consistency requires positive temperature and two views')
    unsupported = ['adaptive_weight', 'capacity_aware', 'robust_kd_reliable', 'ema_itt',
        'push_feature', 'teacher_margin_clean_gate', 'teacher_margin_adv_gate',
        'teacher_margin_per_sample_conflict', 'teacher_margin_relative',
        'adv_ce_weight', 'adv_margin_weight', 'attack_teacher_alpha']
    if method_enabled(cfg, cfg['method_start'] + 1) and (
            any(cfg.get(key, False) for key in unsupported) or not cfg['push_soft']):
        raise ValueError('0909 method branch requires its documented source recipe')


def js_divergence(first, second, temperature=0.5):
    log_p = F.log_softmax(first / temperature, dim=1)
    log_q = F.log_softmax(second / temperature, dim=1)
    log_m = torch.logaddexp(log_p, log_q) - np.log(2.0)
    return (0.5 * (log_p.exp() * (log_p - log_m)
                   + log_q.exp() * (log_q - log_m)).sum(dim=1)).mean()


def kd_losses(clean, adv, targets, labels, cfg, epoch, temp_adv, temp_nat):
    """Retain source KL's log(q+1e-5), reductions and optional G7 correction."""
    robust, robust_clean, natural, _ = targets
    q_adv = F.softmax(robust.detach() / temp_adv, dim=1)
    q_nat = F.softmax(natural.detach() / temp_nat, dim=1)
    ramp = min(1.0, max(0.0,
        (epoch - cfg.get('target_mix_start', 120)) / float(cfg.get('target_mix_warmup', 40))))
    alpha = cfg.get('target_mix_alpha', 0.0) * ramp
    q = q_adv
    if alpha > 0:
        correct = robust_clean.detach().argmax(dim=1).eq(labels).to(q_adv.dtype)
        mixing = alpha * correct.unsqueeze(1)
        q = (1.0 - mixing) * q_adv + mixing * F.softmax(robust_clean.detach() / temp_adv, dim=1)
    adv_terms = -F.log_softmax(adv, dim=1) * q + torch.log(q + 1e-5) * q
    if cfg.get('split_target_mix', False) and alpha > 0:
        from split_target_mix import split_target_mix_loss
        adv_terms, _ = split_target_mix_loss(
            adv_terms, adv, robust, robust_clean, labels, q, temp_adv, alpha,
            cfg['split_target_alpha'] * ramp, cfg['split_nontarget_alpha'] * ramp)
    # Match the historical two successive means in the robust branch.
    adv_loss = adv_terms.mean(dim=1).mean()
    nat_loss = (-F.log_softmax(clean, dim=1) * q_nat + torch.log(q_nat + 1e-5) * q_nat).mean()
    return adv_loss, nat_loss


def base_losses(clean, adv, targets, labels, cfg, epoch, temp_adv, temp_nat):
    """The active components of the two frozen source recipes, without updates."""
    robust, _, _, natural_adv = targets
    adv_kd, nat_kd = kd_losses(clean, adv, targets, labels, cfg, epoch, temp_adv, temp_nat)
    true = adv.gather(1, labels[:, None]).squeeze(1)
    other = adv.clone()
    other.scatter_(1, labels[:, None], -1e9)
    margin = true - other.max(dim=1)[0]
    clean_gate = torch.ones_like(margin)
    if cfg['clean_ce_robust_gate']:
        clean_gate = cfg['clean_ce_gate_floor'] + (1 - cfg['clean_ce_gate_floor']) * torch.sigmoid(
            margin.detach() / max(cfg['clean_ce_gate_tau'], 1e-6))
    clean_ce = (clean_gate * F.cross_entropy(clean, labels, reduction='none')).mean()
    with torch.no_grad():
        teacher_true = robust.gather(1, labels[:, None]).squeeze(1)
        teacher_other = robust.clone()
        teacher_other.scatter_(1, labels[:, None], -1e9)
        teacher_margin = teacher_true - teacher_other.max(dim=1)[0]
        target = teacher_margin.clamp(min=0, max=cfg['teacher_margin_cap'])
        gate = (teacher_margin > 0).float() * torch.sigmoid(
            teacher_margin / max(cfg['teacher_margin_tau'], 1e-6))
    margin_loss = (gate * F.relu(target - margin)).mean()
    margin_ramp = min(1.0, max(0.0, (epoch - cfg['teacher_margin_start']) /
                                  float(max(1, cfg['teacher_margin_warmup']))))
    ce_ramp = min(1.0, max(0.0, (epoch - cfg['ce_start']) / float(max(1, cfg['ce_warmup']))))
    margin_term = margin_ramp * cfg['teacher_margin_weight'] * margin_loss
    # Source computes conflict_scale for diagnostics but does not apply it.
    total = adv_kd + nat_kd + ce_ramp * cfg['clean_ce_weight'] * clean_ce + margin_term
    push = soft_feature_push_loss(adv, natural_adv, labels, T=cfg['push_T'],
        gamma=cfg['push_gamma'], eta=cfg['push_eta'], guide_logits=robust,
        require_guide_correct=cfg['push_require_robust_correct'])
    push_weight = cfg['push_lambda'] * min(1.0, max(0.0, epoch / float(max(1, cfg['push_warmup']))))
    if torch.isnan(push).any():
        push = adv.new_zeros(())
        push_weight = 0.0
    total = total + push_weight * push
    return total, margin_term, {'adv_kd': adv_kd, 'nat_kd': nat_kd,
        'clean_ce': clean_ce, 'teacher_margin': margin_loss, 'push': push,
        'clean_ce_term': ce_ramp * cfg['clean_ce_weight'] * clean_ce,
        'teacher_margin_term': margin_term, 'push_term': push_weight * push}


def pgd_training(model, images, labels, epsilon, steps=10):
    """Same Gaussian CPU RNG initialization and student-only PGD as the source."""
    model.eval()
    adv = images.detach() + 0.001 * torch.randn(images.shape).to(images.device).detach()
    for _ in range(steps):
        adv.requires_grad_()
        grad = torch.autograd.grad(F.cross_entropy(model(adv), labels), [adv])[0]
        adv = adv.detach() + (2 / 255.0) * grad.detach().sign()
        adv = torch.min(torch.max(adv, images - epsilon), images + epsilon).clamp(0, 1)
    model.train()
    return adv.detach()


@contextmanager
def perturb_weights(student, proxy, gamma):
    """Restore exact weights even on failure; callers backward before exiting."""
    originals = []
    relative_norms = []
    try:
        if gamma > 0:
            proxy_params = dict(proxy.named_parameters())
            with torch.no_grad():
                for name, weight in student.named_parameters():
                    grad = proxy_params[name].grad
                    if weight.ndim <= 1 or grad is None:
                        continue
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Nonfinite AWP gradient: ' + name)
                    delta = gamma * weight.norm() * grad / (grad.norm() + 1e-12)
                    originals.append((weight, weight.detach().clone()))
                    weight.add_(delta)
                    relative_norms.append(float(delta.norm() / (originals[-1][1].norm() + 1e-12)))
        yield max(relative_norms, default=0.0)
    finally:
        with torch.no_grad():
            for weight, original in originals:
                weight.copy_(original)


def backward_student(total_loss, margin_term, optimizer, cfg, epoch):
    """Keep source per-parameter margin PCGrad; MobileNet uses plain backward."""
    conflicts = 0
    if (cfg.get('pcgrad_teacher_margin', False) and epoch >= cfg.get('pcgrad_start', 0)
            and margin_term.requires_grad and float(margin_term.detach()) != 0.0):
        params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        optimizer.zero_grad()
        (total_loss - margin_term).backward(retain_graph=True)
        base = [None if p.grad is None else p.grad.detach().clone() for p in params]
        optimizer.zero_grad()
        margin_term.backward()
        margins = [None if p.grad is None else p.grad.detach().clone() for p in params]
        optimizer.zero_grad()
        for p, bg, mg in zip(params, base, margins):
            if bg is None:
                p.grad = mg
            elif mg is None:
                p.grad = bg
            else:
                dot = torch.sum(mg * bg)
                if dot.item() < 0:
                    mg = mg - dot / (torch.sum(bg * bg) + 1e-12) * bg
                    conflicts += 1
                p.grad = bg + mg
    else:
        total_loss.backward()
    return conflicts


def update_dynamics(state, metrics, targets, weight, epoch, optimizer, teacher_optimizer):
    """Historical temperature, diagnostic weight and LR updates, once per batch."""
    ta, tn, initial_adv, initial_nat, wlr, tlr = state
    def entropy(logits, temperature):
        p = F.softmax(logits / temperature, dim=1)
        return (torch.log(p + 1e-5) * p).mean()
    adv_entropy = torch.stack([entropy(t[0], ta) for t in targets]).mean()
    nat_entropy = torch.stack([entropy(t[2], tn) for t in targets]).mean()
    ta = max(1, min(10, ta - tlr * torch.sign(adv_entropy / nat_entropy - 1).item()))
    tn = max(1, min(10, tn - tlr * torch.sign(nat_entropy / adv_entropy - 1).item()))
    initial_adv = metrics['adv_kd'] if initial_adv is None else initial_adv
    initial_nat = metrics['nat_kd'] if initial_nat is None else initial_nat
    lhat_adv = metrics['adv_kd'] / initial_adv
    lhat_nat = metrics['nat_kd'] / initial_nat
    lhat_avg = (lhat_adv + lhat_nat) / len(weight)
    inv_adv, inv_nat = lhat_adv / lhat_avg, lhat_nat / lhat_avg
    weight['nat_loss'] -= wlr * (weight['nat_loss'] - inv_nat / (inv_adv + inv_nat))
    weight['adv_loss'] -= wlr * (weight['adv_loss'] - inv_adv / (inv_adv + inv_nat))
    weight['nat_loss'] = max(0, weight['nat_loss'])
    weight['adv_loss'] = max(0, weight['adv_loss'])
    coef = 1.0 / (weight['adv_loss'] + weight['nat_loss'])
    weight['adv_loss'] *= coef
    weight['nat_loss'] *= coef
    lr = 0.1 if epoch < 150 else (0.1 * (0.5 + 0.5 * np.cos(np.pi * (epoch - 150) / 150))
        * np.exp(-0.01 * (epoch - 150) ** 2 / 150 ** 2))
    teacher_lr = 0 if epoch < 50 else (0.0001 * (0.5 + 0.5 * np.cos(np.pi * (epoch - 50) / 250))
        * np.exp(-0.01 * (epoch - 50) ** 2 / 250 ** 2))
    for group in optimizer.param_groups:
        group['lr'] = lr
    for group in teacher_optimizer.param_groups:
        group['lr'] = teacher_lr
    # Deliberately preserve source's batch-level decay on these epochs.
    if epoch in [215, 260, 285]:
        wlr *= 0.1
        tlr *= 0.1
    metrics.update(lr=float(lr), teacher_lr=float(teacher_lr), temp_adv=float(ta), temp_nat=float(tn))
    return ta, tn, initial_adv, initial_nat, wlr, tlr


class MethodRunner:
    def __init__(self, student, teacher, natural_teacher, cfg):
        validate_config(cfg)
        self.student, self.teacher, self.natural_teacher, self.cfg = student, teacher, natural_teacher, cfg
        self.proxy = copy.deepcopy(student)

    def step(self, images, labels, optimizer, teacher_optimizer, ema_student,
             epoch, epsilon, state, weight):
        cfg, student, teacher, natural = self.cfg, self.student, self.teacher, self.natural_teacher
        device = next(student.parameters()).device
        views = images if isinstance(images, (list, tuple)) else [images]
        views = [view.float().to(device) for view in views]
        labels = labels.to(device)
        optimizer.zero_grad()
        teacher_optimizer.zero_grad()
        self.proxy.load_state_dict(student.state_dict())
        self.proxy.zero_grad()
        teacher.train()
        natural.eval()
        targets, adversarials, teacher_ce = [], [], []
        for view in views:
            self.proxy.train()
            with torch.no_grad():
                self.proxy(view)  # reproduce clean BN update used by source PGD
                natural_clean = natural(view)
                robust_clean = teacher(view)
            adversarial = pgd_training(self.proxy, view, labels, epsilon)
            with torch.no_grad():
                self.proxy(adversarial)  # next view starts from the corresponding BN state
                natural_adv = natural(adversarial)
            robust_adv = teacher(adversarial)
            ce = F.cross_entropy(robust_adv, labels)
            if epoch > 50:
                (ce / len(views)).backward()  # free each WRN graph; defer its optimizer step
            teacher_ce.append(float(ce.detach()))
            targets.append(tuple(t.detach() for t in (robust_adv, robust_clean, natural_clean, natural_adv)))
            adversarials.append(adversarial)
            del robust_adv, ce

        ramp = method_ramp(cfg, epoch)
        gamma = cfg['awp_gamma'] * ramp
        consistency_weight = cfg['consistency_weight'] * ramp
        if gamma > 0:
            self.proxy.load_state_dict(student.state_dict())
            self.proxy.zero_grad()
            self.proxy.train()
            for view, adversarial, target in zip(views, adversarials, targets):
                kd_adv, kd_nat = kd_losses(self.proxy(view), self.proxy(adversarial), target,
                    labels, cfg, epoch, state[0], state[1])
                ((kd_adv + kd_nat) / len(views)).backward()

        student.train()
        with perturb_weights(student, self.proxy, gamma) as relative_norm:
            totals, margins, components, adv_logits = [], [], [], []
            for view, adversarial, target in zip(views, adversarials, targets):
                clean, adv = student(view), student(adversarial)
                total, margin, component = base_losses(clean, adv, target, labels,
                    cfg, epoch, state[0], state[1])
                totals.append(total)
                margins.append(margin)
                components.append(component)
                adv_logits.append(adv)
            total = torch.stack(totals).mean()
            margin = torch.stack(margins).mean()
            js = total.new_zeros(())
            if consistency_weight > 0:
                if len(views) != 2:
                    raise ValueError('Consistency requires two independent views')
                js = js_divergence(adv_logits[0], adv_logits[1], cfg['consistency_temperature'])
            consistency = consistency_weight * js / adv_logits[0].size(1)
            total = total + consistency
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite 0909 training objective')
            conflicts = backward_student(total, margin, optimizer, cfg, epoch)
            metrics = {key: float(torch.stack([c[key].detach() for c in components]).mean())
                       for key in components[0]}
            metrics.update(total_loss=float(total.detach()), js_raw=float(js.detach()),
                consistency_term=float(consistency.detach()), awp_gamma=gamma,
                consistency_weight=consistency_weight, awp_relative_norm_max=relative_norm,
                views=len(views), pcgrad_conflicts=conflicts, teacher_ce=sum(teacher_ce) / len(views),
                student_updates=1, teacher_updates=int(epoch > 50), temperature_updates=1)
        # The context restored theta exactly. SGD momentum/decay use theta, never theta+v.
        state = update_dynamics(state, metrics, targets, weight, epoch, optimizer, teacher_optimizer)
        optimizer.step()
        if cfg['student_ema'] and ema_student is not None:
            ema_update_teacher(ema_student, student, decay=cfg['student_ema_decay'])
        if epoch > 50:
            teacher_optimizer.step()
        return state, metrics
