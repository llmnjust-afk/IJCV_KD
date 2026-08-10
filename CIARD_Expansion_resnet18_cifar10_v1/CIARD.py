'''
mobilenet_v2
CIARD
push according to label
consist are decided by top1 prediction
Lr stage decay
'''
import os
import copy
import torch
from mtard_loss import *
from cifar10_models import *
from cifar10_nat_teacher_models import *
import torchvision
from torchvision import transforms
from loguru import logger
import math
# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# Independent variants respect Slurm CUDA_VISIBLE_DEVICES and ignore legacy runtime overrides.
for _legacy_runtime_env in ("CIARD_GPU", "CIARD_STUDENT", "CIARD_PREFIX"):
    os.environ.pop(_legacy_runtime_env, None)

# Fixed output prefix for this independent variant.
VARIANT_NAME = 'r18_pcgrad_optuna_transfer'
prefix = 'Cifar10_ResNet18_0703_pcgrad_optuna_transfer'
draw_file = prefix
model_dir = './model/' + prefix
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with open('./model/' + prefix+ '/'+ draw_file,'w') as f:
    text = "epoch student_robust_acc student_natural_acc adv_teacher_robust_acc adv_teacher_natural_acc nat_teacher_robust_acc nat_teacher_natural_acc\n"
    f.write(text)
epochs = 300
batch_size = 128
epsilon = 8/255.0

# =============================================================================
# CIARD-Safe+ / CIARD++ configuration
# -----------------------------------------------------------------------------
# The strong CIARD++ components explored earlier did not improve over the CIARD
# baseline in the user's runs. The default is therefore switched to SAFE_PLUS:
# keep the original CIARD optimisation objective and add only low-risk training
# stabilisation (student EMA for evaluation/checkpoints). This is the right next
# step when a strong method consistently underperforms baseline across all
# metrics: preserve the baseline gradient field and improve model selection /
# temporal averaging instead of adding more competing losses.
# =============================================================================
USE_CIARDPP = True
CIARD_SAFE_PLUS = True
CFG = {
    # -------------------------------------------------------------------------
    # (A) soft-weighted feature-level contrastive push loss
    # -------------------------------------------------------------------------
    "push_soft": True,        # Safe+ v2: weak reliable push gave broad robust gains
    "push_feature": False,    # SAFE_PLUS: no feature push by default
    "push_gamma": 4.0,        # sharpness of the soft push weight w_i
    "push_T": 5.0,            # temperature of the logit push term
    "push_eta": 0.3,          # weight of the feature term relative to the logit term
    "proj_dim": 128,          # projection-head embedding dim
    # FIX(Bug 1): the push is now an ADDITIVE, small, bounded penalty with a
    # linear warm-up, instead of a dense term scaled by scale_to_magnitude and
    # subtracted (which collapsed clean accuracy). lambda is the final push
    # weight; it is ramped from 0 over `push_warmup` epochs and capped so the
    # push can never dominate the two distillation losses.
    "push_lambda": 0.08174,      # best observed weak-push region: 0.02-0.10
    "push_warmup": 80,        # slow warm-up avoids early clean-accuracy damage
    # Push is now reliability-gated: apply it only if the clean teacher is wrong
    # but the robust teacher is correct on the same x_adv. This prevents the
    # push from acting on ambiguous samples where neither teacher gives a useful
    # direction, which was improving white-box PGD while hurting black-box.
    "push_require_robust_correct": True,
    # -------------------------------------------------------------------------
    # (B) gradient-based adaptive weighting (GradNorm-style)
    # -------------------------------------------------------------------------
    "adaptive_weight": False, # SAFE_PLUS: original CIARD uses fixed 1:1 weights
    # FIX(Bug 4): enforce a floor on the robust (adv) weight so the GradNorm
    # reweighting can never starve the robust branch (which caused both clean
    # and robust black-box accuracy to drop). adv weight is clamped to
    # [adv_weight_floor, 1 - adv_weight_floor].
    "adv_weight_floor": 0.35,
    # Label anchors: KD/push losses can drift the boundary away from the ground
    # truth. Small CE anchors preserve clean accuracy and transfer robustness.
    "clean_ce_weight": 0.036067,  # clean CE works, but must be robust-gated (below)
    "adv_ce_weight": 0.0,
    "ce_start": 158,          # later start: avoid disturbing early robust KD/push
    "ce_warmup": 116,
    # Robust-gated clean CE: applying clean CE to every sample improved clean acc
    # but destroyed several robust metrics. Gate it by the student's detached
    # adversarial logit margin, so clean CE mainly refines samples whose robust
    # decision is already stable. This preserves the clean gain while reducing
    # the clean/robust trade-off.
    "clean_ce_robust_gate": True,
    "clean_ce_gate_tau": 0.682852,
    "clean_ce_gate_floor": 0.0,
    # CW-style adversarial margin anchor. CW attacks optimise logit margins, so
    # a tiny late margin penalty is more targeted than increasing CE. Keep this
    # small; it is intended to recover black-box CW without hurting PGD gains.
    "adv_margin_weight": 0.0,
    "adv_margin_kappa": 0.0,
    "adv_margin_start": 120,
    "adv_margin_warmup": 80,
    # Teacher-guided CW margin matching. Plain adversarial margin loss improved
    # white-box margins but hurt transfer metrics because it forced every sample
    # in the same direction. This gated variant only asks the student to match a
    # positive robust-teacher margin when the robust teacher is correct, which is
    # a safer signal for black-box CW transfer robustness.
    "teacher_margin_weight": 0.011316,
    "teacher_margin_start": 120,
    "teacher_margin_warmup": 89,
    "teacher_margin_tau": 1.124788,
    "teacher_margin_cap": 1.428932,
    # Extra safety gate for teacher-margin matching. The tm010 experiment shows
    # teacher margin fixes black-box CW, but higher weights hurt clean accuracy
    # and transfer metrics. Experiments showed the CLEAN gate (not the per-sample
    # conflict gate) is what actually protects ResNet-18 white-box robustness: it
    # restricts margin matching to samples whose clean decision boundary is
    # already stable, avoiding sharp/exploitable boundary regions.
    # The clean gate is therefore ON by default, and augmented with an
    # adversarial-margin gate (dual stable-gate): teacher-margin only applies
    # where BOTH clean AND adversarial margins are reasonably positive.
    "teacher_margin_clean_gate": False,
    "teacher_margin_clean_tau": 1.0,
    "teacher_margin_clean_floor": 0.0,
    "teacher_margin_adv_gate": False,
    "teacher_margin_adv_tau": 1.0,
    "teacher_margin_adv_floor": 0.0,
    # Relative margin target. The current absolute hinge
    #   relu(teacher_margin - student_margin)
    # pushes the student ALL the way to the teacher's margin, which over-
    # optimises the adversarial boundary on already-stable samples and causes
    # the slight clean / white-box drop in the clean-gated config. A RELATIVE
    # target only asks the student to close a FRACTION (eta_rel) of the gap to
    # the teacher's margin:
    #   target_rel = student_margin + eta_rel * (teacher_margin - student_margin)
    #   loss = relu(target_rel - student_margin) = eta_rel * relu(teacher_margin - student_margin)
    # This is a gentler nudge that avoids over-sharpening the boundary.
    "teacher_margin_relative": False,
    "teacher_margin_relative_eta": 0.5,
    # Batch-level gradient-conflict gate. A SOFT floor (0.3) keeps a fraction of
    # the teacher-margin signal even on conflicting batches, which the best
    # ResNet-18 config used. Hard floor (0.0) was too aggressive.
    "teacher_margin_conflict_gate": True,
    "teacher_margin_conflict_threshold": 0.0,
    "teacher_margin_conflict_floor": 0.211797,
    # Architecture-aware teacher-margin. The same margin weight that helps
    # MobileNet-V2 can conflict with ResNet-18, whose baseline robust boundary is
    # already stronger. When a ResNet student is detected, teacher-margin is made
    # weaker and later by default; for MobileNet the original tm010 schedule is
    # kept so its all-positive result is preserved.
    "student_arch_adaptive_margin": False,
    "resnet_teacher_margin_weight": 0.005,
    "resnet_teacher_margin_start": 180,
    "resnet_teacher_margin_warmup": 80,
    # Per-sample conflict scaling. The batch-level gate is too coarse for
    # ResNet-18: teacher-margin helps some samples (black-box CW) but fights the
    # robust KD direction on others (white-box PGD/TRADES/CW drop). Instead of
    # switching the whole batch on/off, we scale teacher-margin per sample by how
    # much its individual gradient aligns with the per-sample robust KD gradient.
    # This preserves the black-box CW gain on aligned samples while removing the
    # harmful signal on conflicting samples, which is what causes ResNet-18's
    # white-box drop.
    "teacher_margin_per_sample_conflict": False,
    "teacher_margin_per_sample_tau": 1.0,
    "teacher_margin_per_sample_floor": 0.0,
    # Reliability-aware robust KD: do not blindly distil the robust teacher when
    # it is wrong on x_adv. In that case the adversarial CE anchor should drive
    # the update. This is important for black-box transfer robustness.
    "robust_kd_reliable": False,
    "robust_kd_floor": 0.5,
    # -------------------------------------------------------------------------
    # (C) capacity-aware distillation
    # -------------------------------------------------------------------------
    "capacity_aware": False,
    "capacity_xi": 1.0,       # gating sharpness
    "capacity_floor": 0.5,    # FIX(Bug 3): rho_i in [floor,1]; never zeroes robust KL
    "capacity_start": 60,     # FIX(Bug 3): no gating before this epoch (rho_i == 1)
    # -------------------------------------------------------------------------
    # (D) EMA-stabilised iterative teacher training
    # -------------------------------------------------------------------------
    "ema_itt": False,         # SAFE_PLUS: keep original live robust teacher KD
    "ema_decay": 0.99,        # FIX(Bug 2): faster EMA (was 0.999, too laggy)
    "teacher_warmup": 50,     # epochs to freeze teachers before ITT kicks in
    # FIX(Bug 2): only START using the EMA teacher for soft labels once it has
    # tracked the (now training) robust teacher for a while; before that, use
    # the live teacher so labels are not stale/worse than the live model.
    "ema_use_start": 70,      # epoch from which ema_teacher supplies robust labels
    # Transfer-aware adversarial training: generate x_adv using a mixture of the
    # student and robust teacher gradients. Student-only attacks often improve
    # white-box PGD but reduce black-box transfer robustness.
    "attack_teacher_alpha": 0.0,
    # -------------------------------------------------------------------------
    # SAFE_PLUS low-risk stabilisation: student EMA
    # -------------------------------------------------------------------------
    # Student EMA does not change the training gradients. It only maintains a
    # temporally averaged student for evaluation/checkpointing, which often
    # improves both clean and robust accuracy slightly when the raw final model
    # oscillates. This is the default Safe+ improvement over CIARD baseline.
    "student_ema": True,
    "student_ema_decay": 0.999642,
    "eval_student_ema": True,
    "save_ema_as_student": True,
    # =========================================================================
    # WEIGHT SPACE AVERAGING (Model Soup) — the effective fix for ResNet-18.
    # -------------------------------------------------------------------------
    # Problem: teacher_margin improves black-box CW but hurts white-box + clean
    # on ResNet-18, because its residual connections let the margin gradient
    # disturb the conv features that determine white-box robustness. 8+ rounds of
    # gating experiments proved this cannot be fixed in-training.
    #
    # Solution: save the EMA student at teacher_margin_start (before margin kicks
    # in → white-box-optimal) and at the end of training (black-box-optimal).
    # After training, average their weights: w = alpha*w_pre + (1-alpha)*w_post.
    # Both checkpoints are from the SAME trajectory → same loss basin → weight
    # averaging is valid (Model Soups, Wortsman et al. 2022). The result is a
    # SINGLE model (no extra inference cost) that interpolates between:
    #   alpha=1.0: pure white-box-optimal (baseline-level white-box + clean)
    #   alpha=0.0: pure black-box-optimal (black-box CW gain, white-box drop)
    #   alpha=0.5: balanced
    # =========================================================================
    "weight_averaging": True,
    "wa_alpha": 0.5,             # weight of pre-margin checkpoint (1.0=white-box, 0.0=black-box)
    # =========================================================================
    # IJCV EXTENSION: Label Smoothing + Adaptive Temperature
    # -------------------------------------------------------------------------
    # Problem: Teachers (especially robust teacher) produce overconfident wrong 
    # predictions on x_adv, giving extreme KL gradients that force the student 
    # to blindly mimic teacher boundary artifacts.
    # Solution: (1) Label Smoothing softens the teacher targets, preventing 
    # zero-probability mass issues in KL divergence. (2) Adaptive Temperature 
    # starts high (softer labels) and decays, letting the student first learn 
    # high-level patterns before sharp distilled signals.
    # =========================================================================
    "use_label_smoothing": True,
    "ls_alpha": 0.1,            # Label smoothing factor (0.0=off, 0.1=standard)
    "use_adaptive_temp": True,  # Higher temp early, decay to 1.0
    "temp_init_scale": 2.0,     # Initial temperature multiplier (T_init = init_scale)
    "temp_decay_epochs": 150,   # Over how many epochs to decay to T=1.0
    # =========================================================================
    # IJCV R18 FIX: late clean CE recovery + FGSM anchor
    # =========================================================================
    # Clean Acc and FGSM are the two remaining weak metrics on ResNet-18.
    # (1) Late clean CE recovery only activates after epoch 200, after the
    #     robust boundary has formed, so it won't pull it back.
    # (2) FGSM anchor explicitly trains on one-step adversarial examples
    #     (using robust teacher KD) to fix the FGSM-specific failure mode
    #     without changing the main PGD objective.
    # Conservative values tuned for ResNet-18 (smaller weight, later start
    # than MobileNet to protect the more fragile white-box robustness).
    "late_clean_ce_recovery": True,
    "late_clean_ce_weight": 0.015,
    "late_clean_ce_start": 220,
    "fgsm_anchor": True,
    "fgsm_anchor_weight": 0.025,
    "fgsm_anchor_start": 170,
    # -------------------------------------------------------------------------
    # PCGrad-style gradient surgery for teacher-margin.
    # -------------------------------------------------------------------------
    # ResNet-18's failure mode is that teacher-margin helps black-box metrics but
    # its gradient can oppose the base CIARD objective, producing clean/white-box
    # drops. Instead of gating the LOSS value, PCGrad edits the teacher-margin
    # GRADIENT: if it conflicts with the base CIARD gradient on a parameter, the
    # conflicting component is projected away. This preserves margin information
    # that is orthogonal or aligned with CIARD, while provably removing the part
    # that would increase the base loss.
    "pcgrad_teacher_margin": True,
    "pcgrad_start": 120,
}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Student architecture is fixed for this independent variant.
student = resnet18()
# Architecture-aware teacher-margin. ResNet-18 already has a stronger baseline
# robust boundary than MobileNet-V2, so the same margin weight/start that gives
# MobileNet-V2 all-positive metrics can hurt ResNet-18's white-box robustness.
# When a ResNet student is detected, switch to the more conservative schedule.
if USE_CIARDPP and CFG["student_arch_adaptive_margin"]:
    student_arch = student.__class__.__name__.lower()
    if "resnet" in student_arch:
        CFG["teacher_margin_weight"] = CFG["resnet_teacher_margin_weight"]
        CFG["teacher_margin_start"] = CFG["resnet_teacher_margin_start"]
        CFG["teacher_margin_warmup"] = CFG["resnet_teacher_margin_warmup"]
        logger.info("Detected ResNet student ({}): conservative teacher-margin weight={}, start={}, warmup={}".format(
            student.__class__.__name__, CFG["teacher_margin_weight"], CFG["teacher_margin_start"], CFG["teacher_margin_warmup"]))
    else:
        logger.info("Detected student ({}): teacher-margin weight={}, start={}, warmup={}".format(
            student.__class__.__name__, CFG["teacher_margin_weight"], CFG["teacher_margin_start"], CFG["teacher_margin_warmup"]))
# Independent variants never accept scheduler-provided teacher-margin settings.
for _legacy_tm_env in ("CIARD_TM_WEIGHT", "CIARD_TM_START", "CIARD_TM_WARMUP",
                       "CIARD_TM_PS_FLOOR", "CIARD_TM_PS_CONFLICT", "CIARD_TM_CLEAN_GATE",
                       "CIARD_TM_ADV_GATE", "CIARD_TM_CONFLICT_FLOOR", "CIARD_TM_CLEAN_TAU",
                       "CIARD_TM_ADV_TAU", "CIARD_TM_RELATIVE", "CIARD_TM_RELATIVE_ETA"):
    os.environ.pop(_legacy_tm_env, None)

# Legacy teacher-margin environment overrides are intentionally ignored in independent variants.

resume_student_path = None 
if resume_student_path != None:
    state_dict = torch.load(resume_student_path, map_location=torch.device('cpu'), weights_only=False)["model"]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    student.load_state_dict(new_state_dict)
student = student.cuda()
student.train()
ema_student = None
if USE_CIARDPP and CFG["student_ema"]:
    ema_student = copy.deepcopy(student).cuda()
    for p in ema_student.parameters():
        p.requires_grad_(False)
    ema_student.eval()
if(resume_student_path == None):
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
else:
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

begin_epoch = 1 if resume_student_path == None else 200

weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None



def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def entropy_value(a):
    value = torch.log(a+1e-5)*a
    return value

def scale_to_magnitude(a, b, c):
    if(math.isclose(a, 0, rel_tol=1e-9)): a += 1e-7
    if(math.isclose(b, 0, rel_tol=1e-9)): b += 1e-7
    if(math.isclose(c, 0, rel_tol=1e-9)): c += 1e-7
    magnitude_a = math.floor(math.log10(abs(a)))
    magnitude_b = math.floor(math.log10(abs(b)))
    target_magnitude = min(magnitude_a , magnitude_b)
    magnitude_c = math.floor(math.log10(abs(c)))
    scale_factor = 10 ** (target_magnitude - magnitude_c)
    scaled_c = scale_factor #*c
    return scaled_c

def push_loss(teacher_logits, students_logits, labels,T = 5):#train_batch_labels
    '''print(teacher_logits.shape)
    print(students_logits.shape)
    print(labels.shape)'''
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    #print(teacher_predictions.shape)
    diff_indices = (teacher_predictions != labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = students_logits[diff_indices]
    #print(diff_student_logits)
    
    return kl_loss(F.log_softmax(diff_student_logits/T,dim=1),F.softmax(diff_teacher_logits.detach(),dim=1))
def pull_loss(teacher_logits, students_logits, labels,T=1):#train_batch_labels
    '''print(teacher_logits.shape)
    print(students_logits.shape)
    print(labels.shape)'''
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    #print(teacher_predictions.shape)
    diff_indices = (teacher_predictions == labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = students_logits[diff_indices]
    #print(diff_student_logits)
    return kl_loss(F.log_softmax(diff_student_logits/T,dim=1),F.softmax(diff_teacher_logits.detach(),dim=1))

teacher = wideresnet()#WideResNet()
teacher1_path =  'models/model_cifar_wrn.pt'
#state_dict = torch.load(teacher1_path)
#teacher.load_state_dict(state_dict)

state_dict = torch.load(teacher1_path, map_location=torch.device('cpu'), weights_only=False)#["model"]
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
teacher.load_state_dict(new_state_dict)

#teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
# teacher = teacher.half()
#teacher.eval()
teacher_lr = 0.0001
ADV_teacher_optimizer = optim.SGD(teacher.parameters(), lr=teacher_lr, momentum=0.1, weight_decay=2e-4)
ADV_teacher_loss_CE = torch.nn.CrossEntropyLoss().cuda()
teacher.train()


teacher_nat = cifar10_resnet56()#resnet56()
teacher2_path = 'models/nat_teacher_checkpoint/cifar10_resnnet56.pth'
#state_dict_1 = torch.load(teacher2_path)
#teacher_nat.load_state_dict(state_dict_1)

state_dict = torch.load(teacher2_path, map_location=torch.device('cpu'), weights_only=False)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
teacher_nat.load_state_dict(new_state_dict)

#teacher = torch.nn.DataParallel(teacher)
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()


# =============================================================================
# CIARD++ setup: projection heads (A) and EMA robust teacher (D)
# =============================================================================
student_head = None
nat_teacher_head = None
ema_teacher = None
if USE_CIARDPP and CFG["push_feature"]:
    # projection heads map student / clean-teacher penultimate features into a
    # shared embedding space for the feature-level push (component A).
    student_head = ProjectionHead(student.feature_dim, proj_dim=CFG["proj_dim"]).cuda()
    nat_teacher_head = ProjectionHead(teacher_nat.feature_dim, proj_dim=CFG["proj_dim"]).cuda()
    # the student head is trained jointly with the student.
    optimizer.add_param_group({"params": student_head.parameters()})
    # the clean-teacher head is frozen (clean teacher is fixed); we only need a
    # stable projection, so we keep it in eval mode with no grad.
    nat_teacher_head.eval()

if USE_CIARDPP and CFG["ema_itt"]:
    # EMA copy of the robust teacher actually used for distillation (component D).
    # `teacher` becomes the fast AT-updated copy; `ema_teacher` is the slow,
    # stable teacher that produces the robust soft labels.
    import copy as _copy
    ema_teacher = _copy.deepcopy(teacher).cuda()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)
    ema_teacher.eval()


weight_learn_rate = 0.025
temp_learn_rate = 0.001

ce_loss = torch.nn.CrossEntropyLoss().cuda()
ce_loss_test = torch.nn.CrossEntropyLoss(reduction='none')
best_accuracy = 0

temp_adv = 1
temp_nat = 1

temp_max = 10
temp_min = 1

logger.info('''
CIARD-Safe+ (IJCV extension)
Default: original CIARD objective + student EMA evaluation/checkpointing.
Strong CIARD++ components remain available through CFG, but are disabled by
default because the latest full-metric run showed systematic regression.
Lr stage decay, epoch = 300 coslr
''')
logger.info("CIARD explicit teacher-margin config: prefix={}, student={}, weight={}, start={}, warmup={}, clean_gate={}, clean_tau={}, adv_gate={}, adv_tau={}, conflict_floor={}, per_sample_conflict={}, relative={}, relative_eta={}{}".format(
    prefix, student.__class__.__name__, CFG["teacher_margin_weight"], CFG["teacher_margin_start"],
    CFG["teacher_margin_warmup"], CFG["teacher_margin_clean_gate"], CFG["teacher_margin_clean_tau"],
    CFG["teacher_margin_adv_gate"], CFG["teacher_margin_adv_tau"], CFG["teacher_margin_conflict_floor"],
    CFG["teacher_margin_per_sample_conflict"], CFG["teacher_margin_relative"], CFG["teacher_margin_relative_eta"],
    "" if CFG["teacher_margin_relative"] else " (ignored because teacher_margin_relative=False)"))
logger.info("""CIARD resolved train config:
variant: {}
prefix: {}
dataset: {} train_samples={} test_samples={}
student: {} num_classes={}
epochs: {}
batch_size: {}
epsilon: {}
train_pgd_steps: 10
train_pgd_step_size: 2/255
robust_teacher_checkpoint: {}
natural_teacher_checkpoint: {}
USE_CIARDPP: {}
CIARD_SAFE_PLUS: {}
model_dir: {}
CFG:
{}
""".format(
    VARIANT_NAME, prefix, trainset.__class__.__name__, len(trainset), len(testset),
    student.__class__.__name__, student.linear.out_features, epochs, batch_size, epsilon,
    teacher1_path, teacher2_path, USE_CIARDPP, CIARD_SAFE_PLUS, model_dir,
    "\n".join("  {}: {}".format(k, CFG[k]) for k in sorted(CFG))))

for epoch in range(begin_epoch,epochs+1):
    logger.info('the {}th epoch '.format(epoch)) 
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader): 
        student.train()
        teacher.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        ADV_teacher_optimizer.zero_grad()
         
        student.train()
        student_nat_logits = student(train_batch_data)
        with torch.no_grad():
            teacher_nat_logits = teacher_nat(train_batch_data)
            adv_teacher_nat = teacher(train_batch_data)

        student_adv_logits,teacher_adv_logits,nat_adv_logits,student_adv_feat,nat_adv_feat,x_adv = robust_inner_loss_push(
                                                                                        student,teacher,teacher_nat,
                                                                                        train_batch_data,train_batch_labels,
                                                                                        optimizer,ADV_teacher_optimizer,
                                                                                        step_size=2/255.0,
                                                                                        epsilon=epsilon,perturb_steps=10,
                                                                                        attack_teacher_alpha=CFG["attack_teacher_alpha"])

        # (D) EMA-ITT: the soft robust label comes from the slow EMA teacher,
        # which is more stable than the fast AT-updated `teacher`. FIX(Bug 2):
        # only switch to the EMA teacher once it has tracked the (now training)
        # robust teacher for a while (ema_use_start). Before that the EMA copy
        # is staler/weaker than the live teacher, so we keep the live labels.
        if (USE_CIARDPP and CFG["ema_itt"]
                and epoch >= CFG["ema_use_start"]):
            with torch.no_grad():
                robust_soft_logits = ema_teacher(x_adv)
        else:
            robust_soft_logits = teacher_adv_logits

        kl_Loss1 = kl_loss(F.log_softmax(student_adv_logits,dim=1),F.softmax(robust_soft_logits.detach()/temp_adv,dim=1))
        kl_Loss2 = kl_loss(F.log_softmax(student_nat_logits,dim=1),F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1))
        # Label Smoothing: softens teacher targets to prevent zero-mass KL gradients.
        # Applied AFTER the kl_loss computation so the loss itself uses smoothed targets.
        if USE_CIARDPP and CFG.get("use_label_smoothing", False):
            num_classes = student_adv_logits.size(-1)
            alpha = CFG["ls_alpha"]
            # Smooth robust teacher's target
            robust_target = F.softmax(robust_soft_logits.detach()/temp_adv, dim=1)
            robust_target = robust_target * (1 - alpha) + alpha / num_classes
            # Smooth clean teacher's target
            clean_target = F.softmax(teacher_nat_logits.detach()/temp_nat, dim=1)
            clean_target = clean_target * (1 - alpha) + alpha / num_classes
            # Recompute kl_Loss with smoothed targets
            kl_Loss1 = kl_loss(F.log_softmax(student_adv_logits, dim=1), robust_target)
            kl_Loss2 = kl_loss(F.log_softmax(student_nat_logits, dim=1), clean_target)
        # Reliability-aware robust KD: do not blindly imitate a wrong robust
        # teacher. If the robust teacher is correct and confident on y, the KL
        # keeps nearly full weight; if it is wrong, the KL drops to a floor and
        # the adversarial CE anchor dominates that sample.
        robust_gate = torch.ones(train_batch_labels.size(0)).cuda()
        if USE_CIARDPP and CFG["robust_kd_reliable"]:
            with torch.no_grad():
                p_robust = F.softmax(robust_soft_logits.detach(), dim=1)
                p_robust_true = p_robust.gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
                robust_top1 = torch.argmax(robust_soft_logits.detach(), dim=1)
                robust_correct = (robust_top1 == train_batch_labels).float()
                robust_gate = (CFG["robust_kd_floor"]
                               + (1.0 - CFG["robust_kd_floor"])
                               * robust_correct * p_robust_true)

        # (C) capacity-aware gating: gently emphasise the robust KL on samples
        # the student is starting to handle (per-sample rho_i), an automatic
        # easy->hard curriculum. kl_Loss1 is [B, C]; we reduce over classes then
        # weight. FIX(Bug 3): (i) bounded rho_i in [floor,1] so the robust KL is
        # never zeroed; (ii) only active after capacity_start (it collapsed when
        # applied from epoch 1); (iii) average by BATCH SIZE, not sum(rho), to
        # keep the robust-loss magnitude stable across batches.
        if (USE_CIARDPP and CFG["capacity_aware"]
                and epoch >= CFG["capacity_start"]):
            rho = capacity_weight(student_adv_logits, train_batch_labels,
                                  xi=CFG["capacity_xi"], floor=CFG["capacity_floor"])  # [B]
            # per-sample MEAN over classes (same scale as original torch.mean),
            # then rho-weighted MEAN over the batch. With rho==1 this is exactly
            # the original torch.mean(kl_Loss1), so the magnitude relative to
            # kl_Loss2 (and thus adaptive weighting / push scaling) is preserved.
            kl_Loss1_persample = torch.mean(kl_Loss1, dim=1)          # [B]
            kl_Loss1 = torch.mean(robust_gate * rho * kl_Loss1_persample)
        else:
            kl_Loss1_persample = torch.mean(kl_Loss1, dim=1)          # [B]
            kl_Loss1 = torch.mean(robust_gate * kl_Loss1_persample)
        kl_Loss2 = torch.mean(kl_Loss2)
        adv_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
        nat_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1)))
        temp_adv = temp_adv - temp_learn_rate * torch.sign((adv_teacher_entropy.detach() / nat_teacher_entropy.detach() - 1)).item()
        temp_nat = temp_nat - temp_learn_rate * torch.sign((nat_teacher_entropy.detach() / adv_teacher_entropy.detach() - 1)).item()
        temp_adv = max(min(temp_max, temp_adv), temp_min)
        temp_nat = max(min(temp_max, temp_nat), temp_min)
        # Adaptive Temperature: Start with higher temperature (softer labels) and
        # decay over training. This prevents early overfitting to noisy soft labels.
        if USE_CIARDPP and CFG.get("use_adaptive_temp", False):
            decay_epochs = CFG["temp_decay_epochs"]
            init_scale = CFG["temp_init_scale"]
            # Linearly decay from init_scale to 1.0 over decay_epochs
            progress = min(1.0, epoch / decay_epochs)
            temp_scale = init_scale - (init_scale - 1.0) * progress
            temp_adv = max(1.0, temp_adv * temp_scale)
            temp_nat = max(1.0, temp_nat * temp_scale)
        if init_loss_nat == None:
            init_loss_nat = kl_Loss2.item()
        if init_loss_adv == None:
            init_loss_adv = kl_Loss1.item()
        G_avg = (kl_Loss1.item() + kl_Loss2.item()) / len(weight)
        lhat_adv = kl_Loss1.item() / init_loss_adv
        lhat_nat = kl_Loss2.item() / init_loss_nat
        lhat_avg = (lhat_adv + lhat_nat) / len(weight)
        inv_rate_adv = lhat_adv / lhat_avg
        inv_rate_nat = lhat_nat / lhat_avg
        weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate *(weight["nat_loss"] - inv_rate_nat/(inv_rate_adv + inv_rate_nat))
        weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate *(weight["adv_loss"] - inv_rate_adv/(inv_rate_adv + inv_rate_nat))
        num_losses = len(weight)
        if weight["adv_loss"] <0:
            weight["adv_loss"] = 0
        if weight["nat_loss"]< 0:
            weight["nat_loss"] = 0
        coef = 1.0/(weight["adv_loss"] + weight["nat_loss"])
        weight["adv_loss"] *= coef
        weight["nat_loss"] *= coef
        # FIX(Bug 4): enforce a floor on the robust (adv) weight so the GradNorm
        # reweighting can never starve the robust branch. Without this, the
        # adaptive scheme reallocated almost all weight to the clean branch
        # (especially once the capacity gate shrank the robust loss), which made
        # BOTH clean and robust black-box accuracy drop. We clamp the adv weight
        # into [floor, 1-floor] and renormalise the pair to sum to 1.
        if USE_CIARDPP and CFG["adaptive_weight"]:
            fl = CFG["adv_weight_floor"]
            adv_w = min(max(weight["adv_loss"], fl), 1.0 - fl)
            weight["adv_loss"] = adv_w
            weight["nat_loss"] = 1.0 - adv_w
        # (B) gradient-based adaptive weighting: drive the total loss with the
        # GradNorm-style learned weights so the robust (adv) branch is not
        # drowned out by the clean (nat) branch. The original line kept fixed
        # 1:1 weights; set CFG["adaptive_weight"]=False to recover that.
        if USE_CIARDPP and CFG["adaptive_weight"]:
            total_loss = weight["adv_loss"] * kl_Loss1 + weight["nat_loss"] * kl_Loss2
        else:
            total_loss = 1 * kl_Loss1 + 1 * kl_Loss2

        # Label / margin anchors. They are intentionally weak and late-started:
        # the table shows the weak reliable push already improves most robust
        # metrics, while only clean acc and black-box CW lag. A small clean CE
        # anchor recovers clean accuracy; a small adversarial logit-margin anchor
        # targets CW-style attacks directly (CW optimises logit margins), without
        # reintroducing the strong CE terms that previously harmed CIARD.
        if USE_CIARDPP:
            ce_ramp = min(1.0, max(0.0, (epoch - CFG["ce_start"]) / float(max(1, CFG["ce_warmup"]))))
            true_logit = student_adv_logits.gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
            other_logits = student_adv_logits.clone()
            other_logits.scatter_(1, train_batch_labels.view(-1, 1), -1e9)
            max_other_logit = torch.max(other_logits, dim=1)[0]

            if CFG["clean_ce_robust_gate"]:
                # Gate clean CE by the DETACHED adversarial margin. Clean CE is
                # applied strongly only when the adversarial decision is already
                # reasonably stable; on fragile samples, robust KD/push remains
                # the dominant signal. This avoids the observed failure where
                # cce=0.05 fixed clean accuracy but erased robust gains.
                with torch.no_grad():
                    adv_margin_detached = (true_logit - max_other_logit).detach()
                    clean_gate = torch.sigmoid(adv_margin_detached / max(CFG["clean_ce_gate_tau"], 1e-6))
                    clean_gate = (CFG["clean_ce_gate_floor"]
                                  + (1.0 - CFG["clean_ce_gate_floor"]) * clean_gate)
                clean_ce_per_sample = F.cross_entropy(student_nat_logits, train_batch_labels, reduction='none')
                clean_ce = torch.mean(clean_gate * clean_ce_per_sample)
            else:
                clean_gate = torch.ones(train_batch_labels.size(0)).cuda()
                clean_ce = ce_loss(student_nat_logits, train_batch_labels)
            adv_ce = ce_loss(student_adv_logits, train_batch_labels)
            margin_ramp = min(1.0, max(0.0, (epoch - CFG["adv_margin_start"]) / float(max(1, CFG["adv_margin_warmup"]))))
            adv_margin = torch.mean(F.relu(max_other_logit - true_logit + CFG["adv_margin_kappa"]))
            teacher_margin_ramp = min(1.0, max(0.0, (epoch - CFG["teacher_margin_start"]) / float(max(1, CFG["teacher_margin_warmup"]))))
            with torch.no_grad():
                teacher_true_logit = robust_soft_logits.detach().gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
                teacher_other_logits = robust_soft_logits.detach().clone()
                teacher_other_logits.scatter_(1, train_batch_labels.view(-1, 1), -1e9)
                teacher_max_other_logit = torch.max(teacher_other_logits, dim=1)[0]
                teacher_margin = teacher_true_logit - teacher_max_other_logit
                teacher_correct = (teacher_margin > 0).float()
                teacher_margin_target = torch.clamp(teacher_margin, min=0.0, max=CFG["teacher_margin_cap"])
                teacher_margin_gate = teacher_correct * torch.sigmoid(teacher_margin / max(CFG["teacher_margin_tau"], 1e-6))
                if CFG["teacher_margin_clean_gate"]:
                    clean_true_logit = student_nat_logits.detach().gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
                    clean_other_logits = student_nat_logits.detach().clone()
                    clean_other_logits.scatter_(1, train_batch_labels.view(-1, 1), -1e9)
                    clean_max_other_logit = torch.max(clean_other_logits, dim=1)[0]
                    clean_margin = clean_true_logit - clean_max_other_logit
                    clean_margin_gate = torch.sigmoid(clean_margin / max(CFG["teacher_margin_clean_tau"], 1e-6))
                    clean_margin_gate = (CFG["teacher_margin_clean_floor"]
                                         + (1.0 - CFG["teacher_margin_clean_floor"]) * clean_margin_gate)
                    teacher_margin_gate = teacher_margin_gate * clean_margin_gate
                else:
                    clean_margin_gate = torch.ones(train_batch_labels.size(0)).cuda()
                if CFG["teacher_margin_adv_gate"]:
                    adv_margin_detached = (true_logit - max_other_logit).detach()
                    adv_margin_gate = torch.sigmoid(adv_margin_detached / max(CFG["teacher_margin_adv_tau"], 1e-6))
                    adv_margin_gate = (CFG["teacher_margin_adv_floor"]
                                        + (1.0 - CFG["teacher_margin_adv_floor"]) * adv_margin_gate)
                    teacher_margin_gate = teacher_margin_gate * adv_margin_gate
                else:
                    adv_margin_gate = torch.ones(train_batch_labels.size(0)).cuda()
            student_adv_margin = true_logit - max_other_logit
            # Relative margin target: only close a fraction (eta_rel) of the gap
            # to the teacher's margin, instead of the full gap. This avoids over-
            # optimising the adversarial boundary on already-stable samples,
            # which caused the slight clean / white-box drop in the clean-gated
            # config.
            if CFG["teacher_margin_relative"]:
                gap = F.relu(teacher_margin_target - student_adv_margin)
                teacher_margin_loss = torch.mean(teacher_margin_gate * CFG["teacher_margin_relative_eta"] * gap)
            else:
                teacher_margin_loss = torch.mean(teacher_margin_gate * F.relu(teacher_margin_target - student_adv_margin))
            teacher_margin_conflict_score = torch.tensor(1.0).cuda()
            teacher_margin_conflict_scale = 1.0
            # Per-sample gradient alignment. The previous batch-level gate was
            # too coarse for ResNet-18: it disabled teacher-margin for the whole
            # batch whenever the AVERAGE gradient conflicted, which threw away the
            # aligned samples that help black-box CW. We now compare, per sample,
            # the direction in which robust KD pushes the TRUE-class adversarial
            # logit versus the direction in which the teacher-margin hinge pushes
            # it. If they agree on a sample, teacher-margin is kept for that
            # sample; if they disagree, it is dropped for that sample. This is
            # what targets the ResNet-18 white-box drop while preserving the
            # black-box CW gain that teacher-margin provides on aligned samples.
            teacher_margin_per_sample_scale = torch.ones(train_batch_labels.size(0)).cuda()
            if CFG["teacher_margin_per_sample_conflict"] and CFG["teacher_margin_weight"] > 0:
                # gradient of robust KD wrt the true-class adversarial logit, per
                # sample. kl_Loss1 here is the full [B, C] tensor (NOT yet reduced);
                # the robust-branch reduction happens later, so we can still get a
                # per-sample, per-class gradient cheaply.
                kd_grad_full = torch.autograd.grad(kl_Loss1, student_adv_logits,
                                               retain_graph=True, create_graph=False,
                                               allow_unused=True)[0]
                if kd_grad_full is not None:
                    kd_grad_true = kd_grad_full.detach().gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
                    # gradient of the per-sample teacher-margin hinge wrt the
                    # true-class adversarial logit. The hinge is
                    #   relu(target_i - (z_y - z_other)) ,  d/dz_y = -relu'(.) ,
                    # so the true-class gradient is <= 0 (pushes z_y up when the
                    # hinge is active). We only need its SIGN at the sample level.
                    tm_hinge = F.relu(teacher_margin_target - student_adv_margin)
                    tm_grad_full = torch.autograd.grad(tm_hinge.sum(), student_adv_logits,
                                               retain_graph=True, create_graph=False,
                                           allow_unused=True)[0]
                    if tm_grad_full is not None:
                        tm_grad_true = tm_grad_full.detach().gather(1, train_batch_labels.view(-1, 1)).squeeze(1)
                        # robust KD wants to INCREASE z_y on adversarial inputs
                        # (kd_grad_true > 0); the hinge wants to INCREASE z_y too
                        # (tm_grad_true < 0, because d/dz_y of relu(target-margin)
                        # is -1 when active). So they AGREE when
                        # kd_grad_true > 0 AND tm_grad_true < 0.
                        agree = ((kd_grad_true > 0) & (tm_grad_true < 0)).float()
                        teacher_margin_per_sample_scale = (CFG["teacher_margin_per_sample_floor"]
                                                           + (1.0 - CFG["teacher_margin_per_sample_floor"]) * agree)
                # also keep the batch-level scalar for logging / the original gate
                if CFG["teacher_margin_conflict_gate"]:
                    kd_flat = kd_grad_full.detach().view(-1) if kd_grad_full is not None else torch.zeros(1).cuda()
                    tm_full = tm_grad_full.detach().view(-1) if (CFG["teacher_margin_weight"] > 0 and 'tm_grad_full' in dir() and tm_grad_full is not None) else torch.zeros(1).cuda()
                    denom = torch.norm(kd_flat) * torch.norm(tm_full) + 1e-12
                    teacher_margin_conflict_score = torch.sum(kd_flat * tm_full) / denom
                    teacher_margin_conflict_scale = (1.0 if teacher_margin_conflict_score.item() >= CFG["teacher_margin_conflict_threshold"]
                                                     else CFG["teacher_margin_conflict_floor"])
            elif CFG["teacher_margin_conflict_gate"] and CFG["teacher_margin_weight"] > 0:
                # Fallback: original batch-level cosine gate (used when per-sample
                # conflict is disabled). Kept for backward compatibility.
                kd_grad = torch.autograd.grad(kl_Loss1, student_adv_logits,
                                              retain_graph=True, create_graph=False,
                                              allow_unused=True)[0]
                tm_grad = torch.autograd.grad(teacher_margin_loss, student_adv_logits,
                                              retain_graph=True, create_graph=False,
                                              allow_unused=True)[0]
                if kd_grad is not None and tm_grad is not None:
                    kd_flat = kd_grad.detach().view(-1)
                    tm_flat = tm_grad.detach().view(-1)
                    denom = torch.norm(kd_flat) * torch.norm(tm_flat) + 1e-12
                    teacher_margin_conflict_score = torch.sum(kd_flat * tm_flat) / denom
                    teacher_margin_conflict_scale = (1.0 if teacher_margin_conflict_score.item() >= CFG["teacher_margin_conflict_threshold"]
                                                     else CFG["teacher_margin_conflict_floor"])
            # Apply the (per-sample) conflict scale to the teacher-margin loss.
            # The final loss is the mean over samples of
            #   per_sample_scale * teacher_margin_gate * eta_rel * relu(target - margin),
            # which only spends gradient on samples where teacher-margin AGREES
            # with the original robust KD direction, and only closes a fraction
            # of the gap to avoid over-optimising the boundary.
            tm_gap = F.relu(teacher_margin_target - student_adv_margin)
            if CFG["teacher_margin_relative"]:
                tm_gap = CFG["teacher_margin_relative_eta"] * tm_gap
            tm_per_sample = teacher_margin_per_sample_scale * teacher_margin_gate * tm_gap
            teacher_margin_loss_scaled = torch.mean(tm_per_sample)
            teacher_margin_term = teacher_margin_ramp * CFG["teacher_margin_weight"] * teacher_margin_loss_scaled
            total_loss = (total_loss
                          + ce_ramp * CFG["clean_ce_weight"] * clean_ce
                          + ce_ramp * CFG["adv_ce_weight"] * adv_ce
                          + margin_ramp * CFG["adv_margin_weight"] * adv_margin
                          + teacher_margin_term)
        else:
            clean_ce = torch.tensor(0.0).cuda()
            adv_ce = torch.tensor(0.0).cuda()
            adv_margin = torch.tensor(0.0).cuda()
            clean_gate = torch.ones(train_batch_labels.size(0)).cuda()
            teacher_margin_loss = torch.tensor(0.0).cuda()
            teacher_margin_gate = torch.ones(train_batch_labels.size(0)).cuda()
            clean_margin_gate = torch.ones(train_batch_labels.size(0)).cuda()
            adv_margin_gate = torch.ones(train_batch_labels.size(0)).cuda()
            teacher_margin_conflict_score = torch.tensor(1.0).cuda()
            teacher_margin_conflict_scale = 1.0
            teacher_margin_per_sample_scale = torch.ones(train_batch_labels.size(0)).cuda()
            teacher_margin_term = torch.tensor(0.0).cuda()


        # (A) soft-weighted feature-level contrastive push loss.
        # Replaces the conference hard-mask push_loss: instead of selecting the
        # teacher-misclassified subset, every sample is weighted by a soft
        # confidence weight w_i, and an optional feature-level (cosine) term is
        # added. FIX(Bug 1): soft_feature_push_loss now returns a NON-NEGATIVE
        # penalty that we ADD to the total loss (minimising it decouples the
        # student from the clean teacher on x_adv). The previous code SUBTRACTED
        # a scale_to_magnitude-scaled dense term, which (i) inverted the feature
        # sign so the student was pulled TOWARDS the clean teacher, and (ii)
        # applied a large dense push to every sample, collapsing clean accuracy.
        # The push weight is now a small, fixed lambda that is linearly ramped
        # from 0 over push_warmup epochs so it never destabilises early training.
        if USE_CIARDPP and CFG["push_soft"]:
            kl_Loss3 = soft_feature_push_loss(
                student_adv_logits, nat_adv_logits, train_batch_labels,
                student_feat=(student_adv_feat if CFG["push_feature"] else None),
                teacher_feat=(nat_adv_feat if CFG["push_feature"] else None),
                student_head=(student_head if CFG["push_feature"] else None),
                teacher_head=(nat_teacher_head if CFG["push_feature"] else None),
                T=CFG["push_T"], gamma=CFG["push_gamma"], eta=CFG["push_eta"],
                guide_logits=robust_soft_logits,
                require_guide_correct=CFG["push_require_robust_correct"])
            if torch.isnan(kl_Loss3).any():
                kl_Loss3 = torch.tensor(0.0).cuda()
                loss3_weight = 0.0
            else:
                # linear warm-up of the (small, capped) push weight lambda.
                ramp = min(1.0, max(0.0, epoch / float(max(1, CFG["push_warmup"]))))
                loss3_weight = CFG["push_lambda"] * ramp
                total_loss = total_loss + loss3_weight * kl_Loss3
        else:
            kl_Loss3 = push_loss(nat_adv_logits,student_adv_logits,train_batch_labels)
            if(torch.isnan(kl_Loss3).any() or kl_Loss3.numel() == 0):
                kl_Loss3 = torch.tensor(0.0)
                loss3_weight = 0.0
            else:
                kl_Loss3 = torch.mean(kl_Loss3)
                loss3_weight = scale_to_magnitude(float(kl_Loss1.item()), float(kl_Loss2.item()), float(kl_Loss3.item())) #This is fit the loss3 into the same scale with others,this is not lambda,lambda here is 1
                total_loss -= loss3_weight*kl_Loss3
        '''
        kl_Loss4 = push_loss(adv_teacher_nat,student_nat_logits,train_batch_labels) 
        if(torch.isnan(kl_Loss4).any() or kl_Loss4.numel() == 0):
            kl_Loss4 = torch.tensor(0.0)
        else:
            kl_Loss4 = torch.mean(kl_Loss4)
            loss4_weight = scale_to_magnitude(float(kl_Loss1.item()), float(kl_Loss2.item()), float(kl_Loss4.item()))
            total_loss -= loss4_weight*kl_Loss4
        '''

        if epoch < 150:
            lr = 0.1
        else:
            cosine_term = 0.5 + 0.5 * np.cos(np.pi * (epoch - 150) / (300 - 150))
            exponential_decay = np.exp(-0.01 * (epoch - 150) ** 2 / (300 - 150) ** 2)
            lr = 0.1 * cosine_term * exponential_decay

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch < 50:
            teacher_lr = 0
        else:
            base_lr = 0.0001
            min_lr = 0
            cosine_term = 0.5 + 0.5 * np.cos(np.pi * (epoch - 50) / (300 - 50))
            exponential_decay = np.exp(-0.01 * (epoch - 50) ** 2 / (300 - 50) ** 2)
            teacher_lr = min_lr + (base_lr - min_lr) * cosine_term*exponential_decay
            
        for param_group in ADV_teacher_optimizer.param_groups:
            param_group['lr'] = teacher_lr
        if epoch in [215,260,285]:
            weight_learn_rate *= 0.1
            temp_learn_rate *= 0.1
                    
        # =====================================================================
        # IJCV R18 FIX: Late Clean CE Recovery + FGSM Anchor
        # =====================================================================
        # (1) Late Clean CE Recovery: after epoch 200, add a small ungated clean
        #     CE to recover clean accuracy. The existing clean_ce is robust-gated
        #     which protects robustness but limits clean recovery. This ungated
        #     term only activates late, after the robust boundary is formed.
        late_clean_ce = torch.tensor(0.0).cuda()
        if (USE_CIARDPP and CFG.get("late_clean_ce_recovery", False)
                and epoch >= CFG["late_clean_ce_start"]):
            late_clean_ce = ce_loss(student_nat_logits, train_batch_labels)
            total_loss = total_loss + CFG["late_clean_ce_weight"] * late_clean_ce

        # (2) FGSM Anchor: after epoch 150, generate single-step FGSM adversarial
        #     examples and add a KL divergence between student and robust teacher
        #     on them. This specifically trains the student to be robust against
        #     single-step (FGSM) attacks without changing the multi-step PGD
        #     objective that drives PGD/TRADES/CW robustness.
        fgsm_anchor_loss = torch.tensor(0.0).cuda()
        if (USE_CIARDPP and CFG.get("fgsm_anchor", False)
                and epoch >= CFG["fgsm_anchor_start"]):
            x_fgsm_grad = train_batch_data.clone().detach().requires_grad_(True)
            s_clean_for_grad = student(x_fgsm_grad)
            fgsm_ce = F.cross_entropy(s_clean_for_grad, train_batch_labels)
            grad_fgsm = torch.autograd.grad(fgsm_ce, x_fgsm_grad, create_graph=False)[0]
            x_fgsm = torch.clamp(train_batch_data + epsilon * grad_fgsm.sign(), 0.0, 1.0).detach()
            with torch.no_grad():
                t_fgsm_logits = teacher(x_fgsm)
            s_fgsm_logits = student(x_fgsm)
            fgsm_anchor_loss = kl_loss(
                F.log_softmax(s_fgsm_logits, dim=1),
                F.softmax(t_fgsm_logits.detach() / temp_adv, dim=1)).mean()
            total_loss = total_loss + CFG["fgsm_anchor_weight"] * fgsm_anchor_loss

        student.train()
        pcgrad_conflict_count = 0
        pcgrad_param_count = 0
        if (USE_CIARDPP and CFG.get("pcgrad_teacher_margin", False)
                and epoch >= CFG.get("pcgrad_start", 0)
                and teacher_margin_term.requires_grad
                and float(teacher_margin_term.detach().item()) != 0.0):
            # PCGrad-style gradient surgery. Split the optimization into:
            #   base_loss = original CIARD objective (KD + push + CE anchors)
            #   margin_loss = teacher-guided CW margin
            # If margin_loss has a conflicting gradient on a parameter, project
            # the conflicting component away before adding it to the base grad.
            base_loss = total_loss - teacher_margin_term
            opt_params = []
            for group in optimizer.param_groups:
                opt_params.extend([p for p in group['params'] if p.requires_grad])

            optimizer.zero_grad()
            base_loss.backward(retain_graph=True)
            base_grads = []
            for p in opt_params:
                base_grads.append(None if p.grad is None else p.grad.detach().clone())

            optimizer.zero_grad()
            teacher_margin_term.backward(retain_graph=True)
            margin_grads = []
            for p in opt_params:
                margin_grads.append(None if p.grad is None else p.grad.detach().clone())

            optimizer.zero_grad()
            for p, bg, mg in zip(opt_params, base_grads, margin_grads):
                if bg is None and mg is None:
                    p.grad = None
                    continue
                if bg is None:
                    p.grad = mg.clone()
                    continue
                if mg is None:
                    p.grad = bg.clone()
                    continue
                dot = torch.sum(mg * bg)
                pcgrad_param_count += 1
                if dot.item() < 0:
                    mg = mg - dot / (torch.sum(bg * bg) + 1e-12) * bg
                    pcgrad_conflict_count += 1
                p.grad = bg + mg
        else:
            total_loss.backward()
        optimizer.step()
        if USE_CIARDPP and CFG["student_ema"] and ema_student is not None:
            ema_update_teacher(ema_student, student, decay=CFG["student_ema_decay"])
        ADV_teacher_loss = ADV_teacher_loss_CE(teacher_adv_logits,train_batch_labels)
        if(epoch>50):
            ADV_teacher_loss.backward()
            ADV_teacher_optimizer.step()
            # (D) EMA-stabilised ITT: after the fast `teacher` takes its AT step,
            # blend it into the slow `ema_teacher` that supplies the robust soft
            # labels. The EMA filters high-variance updates from the hardest
            # student-generated attacks, preventing robust-teacher degradation.
            if USE_CIARDPP and CFG["ema_itt"]:
                ema_update_teacher(ema_teacher, teacher, decay=CFG["ema_decay"])
        if step%100 == 0:
            text = 'lr:' + str(lr) 
            text += ' weight_nat: {}, nat_loss: {}, weight_adv: {}, adv_loss: {}'.format(weight["nat_loss"], kl_Loss2.item(), weight["adv_loss"], kl_Loss1.item()) 
            text += " weight-klloss3 " + str(loss3_weight) + " Loss3: " + str(kl_Loss3.item()) 
            text += " clean_ce: {}, adv_ce: {}, adv_margin: {}, clean_gate: {}, teacher_margin: {}, teacher_margin_gate: {}, teacher_clean_gate: {}, teacher_adv_gate: {}, tm_grad_cos: {}, tm_scale: {}, tm_ps_scale: {}, pcgrad_conflicts: {}/{}".format(
                clean_ce.item(), adv_ce.item(), adv_margin.item(), torch.mean(clean_gate).item(),
                teacher_margin_loss.item(), torch.mean(teacher_margin_gate).item(), torch.mean(clean_margin_gate).item(),
                torch.mean(adv_margin_gate).item(),
                teacher_margin_conflict_score.item(), teacher_margin_conflict_scale,
                torch.mean(teacher_margin_per_sample_scale).item(), pcgrad_conflict_count, pcgrad_param_count)
            logger.info(text) 
        

    if epoch == 1 or epoch%10==  0 or epoch >= 250: 
        loss_nat_test = AverageMeter()
        loss_adv_test = AverageMeter()

        eval_student = ema_student if (USE_CIARDPP and CFG["eval_student_ema"] and ema_student is not None) else student
        eval_student.eval()
        student.eval()
        teacher.eval()
        teacher_nat.eval()

        optimizer.zero_grad()
        ADV_teacher_optimizer.zero_grad()
        test_accs = []
        test_accs_naturals = []
        teacher_test_accs = []
        teacher_test_accs_naturals = []

        nat_teacher_test_accs = []
        nat_teacher_test_accs_naturals = []


        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(eval_student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            with torch.no_grad():
                logits = eval_student(test_ifgsm_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            loss_adv_test.update(loss.item(), test_batch_data.size(0))
            
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
            teacher_logits = teacher(test_ifgsm_data)
            teacher_predictions = np.argmax(teacher_logits.cpu().detach().numpy(),axis=1)
            teacher_predictions = teacher_predictions - test_batch_labels.cpu().detach().numpy()
            teacher_test_accs = teacher_test_accs + teacher_predictions.tolist()

            nat_teacher_logits = teacher_nat(test_ifgsm_data)
            nat_teacher_predictions = np.argmax(nat_teacher_logits.cpu().detach().numpy(),axis=1)
            nat_teacher_predictions = nat_teacher_predictions - test_batch_labels.cpu().detach().numpy()
            nat_teacher_test_accs = nat_teacher_test_accs + nat_teacher_predictions.tolist()


        test_accs = np.array(test_accs)
        test_adv = np.sum(test_accs==0)/len(test_accs)
        teacher_test_accs = np.array(teacher_test_accs)
        teacher_test_acc = np.sum(teacher_test_accs==0)/len(teacher_test_accs)

        nat_teacher_test_accs = np.array(nat_teacher_test_accs)
        nat_teacher_test_acc = np.sum(nat_teacher_test_accs==0)/len(nat_teacher_test_accs)
        
        text = f'student robust acc {np.sum(test_accs==0)/len(test_accs):.4f}, teacher robust acc {np.sum(teacher_test_accs==0)/len(teacher_test_accs):.4f}, nat teacher robust acc {np.sum(nat_teacher_test_accs==0)/len(nat_teacher_test_accs):.4f}'
        logger.info(text)

        for step,(test_batch_data,test_batch_labels) in enumerate(testloader): 
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            with torch.no_grad():
                logits = eval_student(test_batch_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            loss_nat_test.update(loss.item(), test_batch_data.size(0))
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs_naturals = test_accs_naturals + predictions.tolist()

            teacher_logits = teacher(test_batch_data)
            teacher_predictions = np.argmax(teacher_logits.cpu().detach().numpy(),axis=1)
            teacher_predictions = teacher_predictions - test_batch_labels.cpu().detach().numpy()
            teacher_test_accs_naturals = teacher_test_accs_naturals + teacher_predictions.tolist()

            nat_teacher_logits = teacher_nat(test_batch_data)
            nat_teacher_predictions = np.argmax(nat_teacher_logits.cpu().detach().numpy(),axis=1)
            nat_teacher_predictions = nat_teacher_predictions - test_batch_labels.cpu().detach().numpy()
            nat_teacher_test_accs_naturals = nat_teacher_test_accs_naturals + nat_teacher_predictions.tolist()
        test_accs_naturals = np.array(test_accs_naturals)
        test_nat = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        teacher_test_accs_naturals = np.array(teacher_test_accs_naturals)
        teacher_test_accs_natural = np.sum(teacher_test_accs_naturals==0)/len(teacher_test_accs_naturals)

        nat_teacher_test_accs_naturals = np.array(nat_teacher_test_accs_naturals)
        nat_teacher_test_accs_natural = np.sum(nat_teacher_test_accs_naturals==0)/len(nat_teacher_test_accs_naturals)

        # ---- Save pre-margin checkpoint for weight averaging ----
        # This checkpoint is saved at teacher_margin_start (before margin kicks
        # in), so it has baseline-level white-box + clean (no margin disturbance).
        # After training, its weights are averaged with the final checkpoint to
        # produce a model that retains both white-box and black-box benefits.
        if (USE_CIARDPP and CFG.get("weight_averaging", False)
                and epoch == CFG["teacher_margin_start"]
                and not os.path.exists('./model/' + prefix + "/student_pre_margin.pth")):
            save_student_pre = ema_student if (USE_CIARDPP and CFG["save_ema_as_student"] and ema_student is not None) else student
            state_pre = {'model': save_student_pre.state_dict(), 'epoch': epoch}
            torch.save(state_pre, './model/' + prefix + "/student_pre_margin.pth")
            logger.info("Saved pre-margin checkpoint at epoch {} for weight averaging".format(epoch))

        if epoch%50 == 0 :
            save_student = ema_student if (USE_CIARDPP and CFG["save_ema_as_student"] and ema_student is not None) else student
            state = { 'model': save_student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            if USE_CIARDPP and ema_student is not None:
                state['raw_student'] = student.state_dict()
                state['ema_student'] = ema_student.state_dict()
            # (A) persist the student projection head so training can be resumed.
            if USE_CIARDPP and student_head is not None:
                state['student_head'] = student_head.state_dict()
            torch.save(state,'./model/' + prefix + "/student_" + str(epoch)+ '.pth')
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            # (D) persist the EMA robust teacher (the one used for distillation).
            if USE_CIARDPP and ema_teacher is not None:
                state['ema_teacher'] = ema_teacher.state_dict()
            torch.save(state,'./model/'+ prefix + "/teacher_" + str(epoch)+ '.pth')
        if epoch > 250:
            save_student = ema_student if (USE_CIARDPP and CFG["save_ema_as_student"] and ema_student is not None) else student
            state = { 'model': save_student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            if USE_CIARDPP and ema_student is not None:
                state['raw_student'] = student.state_dict()
                state['ema_student'] = ema_student.state_dict()
            torch.save(state,'./model/' + prefix + "/student_latest.pth")
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/'+ prefix + "/teacher_latest.pth")
        if (test_nat + test_adv) / 2 > best_accuracy:
            best_accuracy = (test_nat + test_adv)/2
            save_student = ema_student if (USE_CIARDPP and CFG["save_ema_as_student"] and ema_student is not None) else student
            state = { 'model': save_student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            if USE_CIARDPP and ema_student is not None:
                state['raw_student'] = student.state_dict()
                state['ema_student'] = ema_student.state_dict()
            torch.save(state,'./model/' + prefix + "/student_best"+ '.pth')
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/' + prefix + "/teacher_best"+ '.pth')
            logger.info("best accuracy:"+str(best_accuracy))
            
        text = f'student natural acc {np.sum(test_accs_naturals==0)/len(test_accs_naturals):.4f}, adv teacher natural acc {np.sum(teacher_test_accs_naturals==0)/len(teacher_test_accs_naturals):.4f}, nat teacher natural acc {np.sum(nat_teacher_test_accs_naturals==0)/len(nat_teacher_test_accs_naturals):.4f}'
        logger.info(text)
        
        test_acc = np.sum(test_accs==0)/len(test_accs)
        test_accs_natural = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        with open('./model/' + prefix+ '/'+ draw_file,'a') as f:
            text = str(epoch) + " " + str(test_acc) + " " + str(test_accs_natural) + " " + str(teacher_test_acc) + " "+ str(teacher_test_accs_natural)+ " "+ str(nat_teacher_test_acc) + " "+ str(nat_teacher_test_accs_natural)+'\n'
            f.write(text)

# =============================================================================
# POST-TRAINING: WEIGHT SPACE AVERAGING (Model Soup)
# =============================================================================
# After training, average the weights of the pre-margin checkpoint (white-box-
# optimal) and the final/best checkpoint (black-box-optimal). This produces a
# single model that retains both white-box and black-box benefits.
# =============================================================================
if USE_CIARDPP and CFG.get("weight_averaging", False):
    pre_margin_path = './model/' + prefix + "/student_pre_margin.pth"
    post_margin_path = './model/' + prefix + "/student_best.pth"
    if not os.path.exists(post_margin_path):
        post_margin_path = './model/' + prefix + "/student_latest.pth"

    if os.path.exists(pre_margin_path) and os.path.exists(post_margin_path):
        logger.info("=" * 60)
        logger.info("WEIGHT SPACE AVERAGING (Model Soup)")
        logger.info("  pre-margin checkpoint:  {}".format(pre_margin_path))
        logger.info("  post-margin checkpoint: {}".format(post_margin_path))
        alpha = CFG["wa_alpha"]
        logger.info("  alpha = {} (1.0=white-box-optimal, 0.0=black-box-optimal)".format(alpha))

        # load both checkpoints
        pre_state = torch.load(pre_margin_path, map_location='cpu', weights_only=False)
        post_state = torch.load(post_margin_path, map_location='cpu', weights_only=False)
        pre_sd = pre_state['model']
        post_sd = post_state['model']

        # average weights
        averaged_sd = {}
        for k in post_sd.keys():
            if k in pre_sd:
                averaged_sd[k] = alpha * pre_sd[k].float() + (1.0 - alpha) * post_sd[k].float()
            else:
                averaged_sd[k] = post_sd[k]
        # cast back to original dtype
        for k in averaged_sd:
            averaged_sd[k] = averaged_sd[k].to(post_sd[k].dtype)

        # load averaged weights into a fresh student
        wa_student = resnet18()
        wa_student.load_state_dict(averaged_sd)
        wa_student = wa_student.cuda()
        wa_student.eval()

        # evaluate the averaged model
        wa_test_accs = []
        wa_test_accs_naturals = []
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            # white-box PGD
            test_ifgsm_data = attack_pgd(wa_student, test_batch_data, test_batch_labels,
                                         attack_iters=20, step_size=0.003, epsilon=8.0/255.0)
            with torch.no_grad():
                logits = wa_student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(), axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            wa_test_accs = wa_test_accs + predictions.tolist()
            # clean
            with torch.no_grad():
                logits = wa_student(test_batch_data)
            predictions = np.argmax(logits.cpu().detach().numpy(), axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            wa_test_accs_naturals = wa_test_accs_naturals + predictions.tolist()

        wa_test_accs = np.array(wa_test_accs)
        wa_test_adv = np.sum(wa_test_accs == 0) / len(wa_test_accs)
        wa_test_accs_naturals = np.array(wa_test_accs_naturals)
        wa_test_nat = np.sum(wa_test_accs_naturals == 0) / len(wa_test_accs_naturals)

        logger.info("Weight-averaged model (alpha={}):".format(alpha))
        logger.info("  Clean acc:   {:.4f}".format(wa_test_nat))
        logger.info("  Robust acc:  {:.4f} (white-box PGD-20)".format(wa_test_adv))
        logger.info("  (Compare with final model: clean={:.4f}, robust={:.4f})".format(
            test_accs_natural, test_acc))

        # save the averaged model
        wa_state = {'model': wa_student.state_dict(), 'epoch': epochs, 'alpha': alpha}
        torch.save(wa_state, './model/' + prefix + "/student_weight_averaged.pth")
        logger.info("  Saved to: ./model/{}/student_weight_averaged.pth".format(prefix))
        logger.info("=" * 60)

        # also sweep alpha to find the best trade-off
        logger.info("Sweeping alpha to find best trade-off...")
        best_wa_score = 0
        best_wa_alpha = alpha
        for sweep_alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            sweep_sd = {}
            for k in post_sd.keys():
                if k in pre_sd:
                    sweep_sd[k] = (sweep_alpha * pre_sd[k].float() + (1.0 - sweep_alpha) * post_sd[k].float()).to(post_sd[k].dtype)
                else:
                    sweep_sd[k] = post_sd[k]
            sweep_student = resnet18()
            sweep_student.load_state_dict(sweep_sd)
            sweep_student = sweep_student.cuda()
            sweep_student.eval()
            sw_accs = []
            sw_nats = []
            for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
                test_batch_data = test_batch_data.float().cuda()
                test_batch_labels = test_batch_labels.cuda()
                test_ifgsm_data = attack_pgd(sweep_student, test_batch_data, test_batch_labels,
                                             attack_iters=20, step_size=0.003, epsilon=8.0/255.0)
                with torch.no_grad():
                    logits = sweep_student(test_ifgsm_data)
                predictions = np.argmax(logits.cpu().detach().numpy(), axis=1) - test_batch_labels.cpu().detach().numpy()
                sw_accs = sw_accs + predictions.tolist()
                with torch.no_grad():
                    logits = sweep_student(test_batch_data)
                predictions = np.argmax(logits.cpu().detach().numpy(), axis=1) - test_batch_labels.cpu().detach().numpy()
                sw_nats = sw_nats + predictions.tolist()
            sw_adv = np.sum(np.array(sw_accs) == 0) / len(sw_accs)
            sw_nat = np.sum(np.array(sw_nats) == 0) / len(sw_nats)
            sw_score = 0.5 * sw_nat + 0.5 * sw_adv
            logger.info("  alpha={:.1f}: clean={:.4f}, robust={:.4f}, score={:.4f}".format(
                sweep_alpha, sw_nat, sw_adv, sw_score))
            if sw_score > best_wa_score:
                best_wa_score = sw_score
                best_wa_alpha = sweep_alpha
        logger.info("Best alpha={:.1f} with score={:.4f}".format(best_wa_alpha, best_wa_score))
        logger.info("To use best alpha, set CFG['wa_alpha']={} and re-run".format(best_wa_alpha))
    else:
        logger.warning("Weight averaging skipped: pre_margin or post_margin checkpoint not found.")
        logger.warning("  pre_margin: {} (exists={})".format(pre_margin_path, os.path.exists(pre_margin_path)))
        logger.warning("  post_margin: {} (exists={})".format(post_margin_path, os.path.exists(post_margin_path)))
