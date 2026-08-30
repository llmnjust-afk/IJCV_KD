import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import copy


# =============================================================================
# IJCV extension (CIARD++) helper modules / losses
# -----------------------------------------------------------------------------
# The four components added on top of the conference CIARD are:
#   (A) Soft-weighted feature-level contrastive push loss   -> ProjectionHead +
#                                                              soft_feature_push_loss
#   (B) Gradient-based adaptive weighting                    -> wired in CIARD.py
#   (C) Capacity-aware distillation                          -> capacity_weight
#   (D) EMA-stabilised iterative teacher training (ITT)      -> ema_update_teacher
# All components are OFF by default-equivalent (e.g. ema_decay close to the
# original behaviour) so the file stays backward compatible; toggles live in
# CIARD.py.
# =============================================================================


class ProjectionHead(nn.Module):
    """Maps a backbone penultimate feature into a shared embedding space so that
    student and clean-teacher features (with different dims) can be compared by
    cosine similarity for the feature-level push (component A).

    A 2-layer MLP (Linear-BN-ReLU-Linear) as commonly used in contrastive
    distillation (e.g. CRD). One head per network is instantiated in CIARD.py.
    """

    def __init__(self, in_dim, proj_dim=128, hidden_dim=256):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)   # unit-norm embedding


def soft_push_weight(teacher_logits, labels, gamma=4.0,
                     guide_logits=None, require_guide_correct=False):
    """Soft confidence weight w_i for component A (Eq. 7 in the paper).

    We only want to DECOUPLE the student from the clean teacher on adversarial
    inputs where the clean teacher is WRONG (its prediction is harmful). Where
    the clean teacher is still correct on x_adv there is nothing to push away
    from, so w_i must be 0.

        w_i = 1[ argmax clean_teacher != y ]
              * 1[ argmax guide_teacher == y ]
              * sigmoid( gamma * ( p_top1 - p_true ) )

    The optional guide-teacher mask is important for black-box robustness: if the
    robust teacher is also wrong on the same adversarial input, pushing the
    student away from the clean teacher is not a reliable correction signal; it
    can move the boundary arbitrarily and improve white-box PGD while degrading
    transfer robustness. With `require_guide_correct=True`, the push fires only
    where the clean teacher is wrong AND the robust teacher remains correct.

    FIX: the previous version omitted the wrong-prediction mask, so a correct
    teacher (p_top1 == p_true) still gave sigmoid(0) = 0.5 and the student was
    pushed away from a CORRECT clean teacher at half strength -- this hurt clean
    accuracy. The hard correctness mask restores the conference behaviour (push
    only on the teacher-misclassified subset) while keeping the soft, confidence
    aware magnitude inside that subset. Returns a detached weight of shape [B].
    """
    with torch.no_grad():
        p = F.softmax(teacher_logits.detach(), dim=1)
        p_true = p.gather(1, labels.view(-1, 1)).squeeze(1)
        p_top1, top1 = p.max(dim=1)
        wrong = (top1 != labels).float()          # 1 only where clean teacher is wrong
        if require_guide_correct and guide_logits is not None:
            guide_top1 = torch.argmax(guide_logits.detach(), dim=1)
            guide_correct = (guide_top1 == labels).float()
            wrong = wrong * guide_correct
        w = wrong * torch.sigmoid(gamma * (p_top1 - p_true))
    return w


def soft_feature_push_loss(student_logits, teacher_logits, labels,
                           student_feat=None, teacher_feat=None,
                           student_head=None, teacher_head=None,
                           T=5.0, gamma=4.0, eta=0.5,
                           guide_logits=None, require_guide_correct=False):
    """Soft-weighted feature-level contrastive push loss (component A).

    IMPORTANT (sign convention, fixed):
        This function returns a NON-NEGATIVE penalty that the caller ADDS to the
        total loss. Minimising the total loss therefore MINIMISES this penalty,
        which corresponds to DECOUPLING the student from the clean teacher on
        adversarial inputs. Earlier the caller subtracted a logit term while the
        feature term used raw cosine, so minimisation actually *aligned* the
        student with the clean teacher on x_adv (the opposite of decoupling) and
        the dense logit push collapsed clean accuracy. Both issues are fixed by
        making every sub-term a proper non-negative penalty.

    Sub-terms (both per-sample weighted by the soft push weight w_i):
      * logit decoupling : encourage the student's adversarial softmax to MOVE
        AWAY from the clean teacher's softmax. We penalise *agreement*, measured
        as the probability mass the student puts on the clean teacher's
        (over-confident, often wrong) top-1 class. Lower => more decoupled.
      * feature decoupling: penalise positive cosine similarity between the
        projected student / clean-teacher features. Using relu(cos) keeps the
        term non-negative and only repels when they are actually aligned, so it
        cannot blow up clean features.
    """
    w = soft_push_weight(teacher_logits, labels, gamma=gamma,
                         guide_logits=guide_logits,
                         require_guide_correct=require_guide_correct)   # [B]

    # ---- logit-level decoupling (non-negative agreement penalty) ----
    # penalise the student's probability on the clean teacher's top-1 class on
    # adversarial inputs. This repels the student from the clean teacher's
    # (frequently wrong) prediction without dragging every logit around.
    p_s = F.softmax(student_logits / T, dim=1)
    with torch.no_grad():
        t_top1 = torch.argmax(teacher_logits.detach(), dim=1)        # [B]
    agree = p_s.gather(1, t_top1.view(-1, 1)).squeeze(1)             # [B] in [0,1]
    logit_term = torch.mean(w * agree)

    # ---- feature-level decoupling (relu cosine, non-negative) ----
    feature_term = student_logits.new_zeros(())
    if (student_feat is not None and teacher_feat is not None
            and student_head is not None and teacher_head is not None):
        z_s = student_head(student_feat)
        with torch.no_grad():
            z_t = teacher_head(teacher_feat.detach())
        cos = torch.sum(z_s * z_t, dim=1)          # [B], in [-1, 1]
        feature_term = torch.mean(w * F.relu(cos))  # only repel when aligned

    return logit_term + eta * feature_term


def capacity_weight(student_logits, labels, xi=1.0, floor=0.5):
    """Capacity-aware per-sample gate rho_i (component C, Eq. 9) -- fixed.

    The original rho_i = p_student(y|x*)**xi collapsed: early in training the
    student is weak on adversarial inputs, so p_true ~ 0 for most samples and
    rho ~ 0, which switched OFF the robust KL exactly when it is needed most
    (and the subsequent /sum(rho) normalisation shrank the gradient further).

    Fix: a BOUNDED gate with a non-zero floor that never disables robust
    supervision and only MILDLY emphasises the samples the student is starting
    to get right (gentle easy-to-hard curriculum):

        rho_i = floor + (1 - floor) * p_true ** xi     in [floor, 1].

    With floor=0.5 the robust signal on a hard sample is at most halved rather
    than zeroed. Set floor=1.0 to disable curriculum (rho_i == 1). The caller
    should average rho-weighted losses by the BATCH SIZE (not sum(rho)) to keep
    the overall robust-loss magnitude stable across batches.
    """
    with torch.no_grad():
        p = F.softmax(student_logits.detach(), dim=1)
        p_true = p.gather(1, labels.view(-1, 1)).squeeze(1)
        rho = floor + (1.0 - floor) * torch.clamp(p_true, 0.0, 1.0) ** xi
    return rho


@torch.no_grad()
def ema_update_teacher(ema_teacher, teacher, decay=0.999):
    """EMA-stabilised ITT update (component D, Eq. 10).

    ema_teacher <- decay * ema_teacher + (1 - decay) * teacher
    The `teacher` here is the copy that has just taken an AT (CE on adversarial
    examples) step; `ema_teacher` is the slowly-moving robust teacher actually
    used for distillation. Both params and float buffers (BN running stats) are
    EMA-averaged so the teacher used for soft labels stays self-consistent;
    integer buffers (e.g. num_batches_tracked) are copied verbatim.
    """
    for ema_p, p in zip(ema_teacher.parameters(), teacher.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(ema_teacher.buffers(), teacher.buffers()):
        if ema_b.data.is_floating_point():
            ema_b.data.mul_(decay).add_(b.data, alpha=1.0 - decay)
        else:
            ema_b.data.copy_(b.data)

def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    device = next(model.parameters()).device
    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        model.zero_grad()
        logits = model(train_ifgsm_data)
        loss = ce_loss(logits,train_batch_labels.to(device))
        train_grad = torch.autograd.grad(loss, train_ifgsm_data)[0].detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    model.zero_grad()
    return train_ifgsm_data

def robust_inner_loss_push(model,
                teacher_adv_model,
                teacher_nat,
                x_natural,
                y,
                optimizer,
                teacher_adv_optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0,
                attack_teacher_alpha=0.0,
                teacher_train_mode=True):

    criterion_ce_loss = torch.nn.CrossEntropyLoss().cuda()
    model.eval()
    teacher_adv_model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            student_attack_loss = criterion_ce_loss(model(x_adv), y.cuda())
            if attack_teacher_alpha > 0:
                teacher_attack_loss = criterion_ce_loss(teacher_adv_model(x_adv), y.cuda())
                loss_ce = ((1.0 - attack_teacher_alpha) * student_attack_loss
                           + attack_teacher_alpha * teacher_attack_loss)
            else:
                loss_ce = student_attack_loss
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    # Keep the robust teacher BatchNorm frozen during its warm-up period.  The
    # caller enables train mode only once iterative teacher updates start.
    teacher_adv_model.train(teacher_train_mode)
    teacher_nat.eval()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    teacher_adv_optimizer.zero_grad()

    # Forward with features so the feature-level push loss (component A) can be
    # computed. We extract penultimate features from the student and the CLEAN
    # teacher (the push repels the student from the clean teacher on x_adv).
    student_logits, student_feat = model(x_adv, return_feature=True)
    teacher_logits = teacher_adv_model(x_adv)
    with torch.no_grad():
        nat_logits, nat_feat = teacher_nat(x_adv, return_feature=True)

    return student_logits, teacher_logits, nat_logits, student_feat, nat_feat, x_adv

def CIARD_inner_loss(model,
                teacher_adv_model,
                teacher_nat,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):

    criterion_ce_loss = torch.nn.CrossEntropyLoss().cuda()
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = criterion_ce_loss(model(x_adv), y.cuda())
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    student_logits = model(x_adv)
    teacher_adv_model.eval()
    teacher_nat.eval()
    with torch.no_grad():
        teacher_logits = teacher_adv_model(x_adv)
        nat_logits = teacher_nat(x_adv)
    return student_logits, teacher_logits, nat_logits, x_adv


# =============================================================================
# SARD (Strength-Adaptive Reliability-Calibrated Distillation) components
# =============================================================================

def teacher_reliability_score(teacher_logits, labels, temperature=1.0,
                               tau_m=2.0, floor=0.1):
    """Teacher Reliability Score (TRS) for SARD Module 2 (RCD).

    Computes a per-sample reliability weight for the robust teacher's
    predictions on adversarial inputs. The weight is in [floor, 1.0]:
      - When teacher is CORRECT: full weight = floor + (1-floor) * confidence * margin_gate
      - When teacher is WRONG: floor only (minimum distillation signal preserved)

    The floor ensures that even when the teacher predicts incorrectly, a
    minimum distillation signal passes through (preventing complete signal loss).
    Previously, the formula was correct * margin_gate * (floor + ...) which
    zeroed out the entire weight when correct=0, making floor useless.

    Args:
        teacher_logits: [B, C] raw logits from the robust teacher on x_adv
        labels: [B] ground-truth labels
        temperature: temperature used for the teacher's softmax
        tau_m: margin normalization temperature
        floor: minimum reliability weight (prevents zeroing out distillation)

    Returns:
        trs: [B] tensor in [floor, 1.0], per-sample reliability weight (detached)
    """
    with torch.no_grad():
        p = F.softmax(teacher_logits.detach() / max(temperature, 1e-6), dim=1)
        p_true = p.gather(1, labels.view(-1, 1)).squeeze(1)

        teacher_pred = torch.argmax(teacher_logits.detach(), dim=1)
        correct = (teacher_pred == labels).float()

        true_logit = teacher_logits.detach().gather(
            1, labels.view(-1, 1)).squeeze(1)
        other_logits = teacher_logits.detach().clone()
        other_logits.scatter_(1, labels.view(-1, 1), -1e9)
        max_other = torch.max(other_logits, dim=1)[0]
        margin = true_logit - max_other

        margin_gate = torch.sigmoid(margin / max(tau_m, 1e-6))

        # When correct: floor + (1-floor) * confidence * margin_gate
        # When wrong: floor (minimum signal preserved)
        full_weight = floor + (1.0 - floor) * p_true * margin_gate
        trs = floor + (correct * (full_weight - floor))

    return trs


def sample_epsilon_curriculum(epoch, total_epochs, eps_max=8.0/255.0,
                               eps_min=1.0/255.0):
    """Sample epsilon from a Beta distribution with curriculum scheduling.

    Three-phase curriculum:
      Phase 1 (0-33%): Weak attacks (Beta(2,5) -> mean ~0.29 * eps_max)
      Phase 2 (33-67%): Transition to uniform (Beta(2,2) -> mean ~0.5 * eps_max)
      Phase 3 (67-100%): Strong attacks (Beta(5,2) -> mean ~0.71 * eps_max)

    The sampled epsilon is clamped to [eps_min, eps_max].

    Args:
        epoch: current epoch (1-indexed)
        total_epochs: total number of training epochs
        eps_max: maximum perturbation budget
        eps_min: minimum perturbation budget

    Returns:
        eps_sample: sampled epsilon value (float)
    """
    progress = float(epoch) / float(max(total_epochs, 1))

    if progress < 0.33:
        alpha, beta = 2.0, 5.0
    elif progress < 0.67:
        t = (progress - 0.33) / 0.34
        alpha = 2.0 + t * 0.0  # stays at 2
        beta = 5.0 + t * (2.0 - 5.0)  # 5 -> 2
    else:
        t = (progress - 0.67) / 0.33
        alpha = 2.0 + t * (5.0 - 2.0)  # 2 -> 5
        beta = 2.0 + t * (2.0 - 2.0)  # stays at 2

    sample = float(torch.distributions.Beta(alpha, beta).sample().item())
    eps_sample = eps_min + sample * (eps_max - eps_min)
    return min(max(eps_sample, eps_min), eps_max)
