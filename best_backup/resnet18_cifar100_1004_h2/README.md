# CIFAR-100 / ResNet-18 — 1004 H2 冻结源码备份

2026-10-05：按用户选择同步已测H2。1004表示实验批次 `1004-cifar100-binary-v1`；实际源码同步及备份日期为2026-10-05。此入口替代旧0914v1 R2，历史结果和冻结来源继续保留。

**本次seed0、同一固定第190轮EMA的八项指标均严格超过原CIARD论文。** Clean65.96%，最小余量FGSM为+0.05个百分点；AA24.22%单列，不设置CIFAR-100论文AA门槛。这是用户选定的当前达标方法，不代表逐项超过所有历史模型或已经证明多seed稳定。

## 固定配置与方法

- 来源：`run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0`。
- VARIANT_NAME：`resnet18_cifar100_nb_l2_w40_s0`；prefix：`Cifar100_ResNet18_1004_binary_v1_H2`；方法：`natural_binary_outer`。
- 从头训练，seed0、50,000张训练图片、batch128；原300轮学习率公式截断至190轮，唯一正式评测权重为最后更新后的EMA `student_epoch190.pth`。训练不读取test、不进行test选模或权重平均扫描。
- 新增自然教师真类／其余类二元KL：weight=2、start120/warmup40，仅学生外层；自然教师正确、学生Clean错误且教师真类概率更高时生效；全batch均值再除100类，使用原自然教师温度、学生T1、无T²。第121轮开始非零，第160轮满强度。AWP内层保留原自然KD及含target mix/split的对抗KD，不加入该额外二元项。
- 继承双视图、KD-AWP gamma=.003、一致性=.50、target mix=.20、split target=.25/non-target=0、push=.081740、门控Clean CE=.036067/start158/warmup116/τ=.682852、teacher margin=.011316/start120/warmup89及margin PCGrad、student EMA=.999642。
- CE预算转移、CE损失保持、CE梯度投影及ELS均关闭。`teacher_margin_conflict_gate`保留历史配置值，但当前runner未应用该额外门控；不能仅据配置名声称启用。
- 完整CFG保存在[同步清单](../../SYNC_MANIFEST.json)的H2条目，并附于下方；Python逐字保留已测实现，包括历史注释，实际运行行为以代码及本说明为准。

自然教师为原始包WRN-22-6 `models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth`，SHA256 `c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935`。鲁棒教师为循环更新的WRN-70-16 `models/cifar100_linf_wrn70-16_without.pt`，SHA256 `3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c`。输入raw [0,1]、模型内部CIFAR-100归一化，未新增外部Normalize。

## 已完成测试结果

单位为%；论文差为百分点。论文来源和全部历史列保留在[总README](../../README.md)。每项来自同一个固定checkpoint的完整10,000张测试，均值不含AA。

| Metric | Paper CIARD baseline | H2 | Δ vs paper |
| --- | ---: | ---: | ---: |
| Clean | 65.73 | 65.96 | +0.23 |
| White-box FGSM | 34.47 | 34.52 | +0.05 |
| White-box PGDsat | 28.05 | 29.76 | +1.71 |
| White-box PGDtrades | 29.45 | 30.78 | +1.33 |
| White-box CW | 24.43 | 26.66 | +2.23 |
| Black-box PGDtrades | 42.29 | 43.52 | +1.23 |
| Square (query-based) | 49.76 | 51.90 | +2.14 |
| Black-box CW | 41.44 | 42.29 | +0.85 |
| AutoAttack (separate) | — | 24.22 | — |
| Seven-attack mean | 35.70 | 37.06 | +1.36 |
| Eight-metric mean | 39.45 | 40.67 | +1.22 |

FGSM3452张正确，比论文报告值对应的3447张多5张。AA比非同期历史1002 G4低0.80个百分点，八项均值低0.23个百分点；H2达成的是逐项超过原论文的目标。

保留L∞8/255、训练PGD10/步长2/255、评测PGDsat20/2/255、PGDtrades20/.003、CW30/2/255、Square100 queries及AA；完整评测每指标攻击seed0。PGDtrades步长.003继承官方/历史实现，与论文文字2/255存在差异；Square100 queries与论文所述预算一致。方法设计和190轮终点曾参考历史测试结果，不能把项目test称为从未参与研究决策，也不能由本次单seed结果推出统计稳定。

## 完成证据

- 训练／评测job：137534／137729，均COMPLETED/0:0，实际节点均为aias-compute-6、每作业1×A800。
- 原checkpoint：`/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/model/Cifar100_ResNet18_1004_binary_v1_H2/student_epoch190.pth`；SHA256：`e32461425fc112b763c6650c65756979e8ca75c3342119acbe4363075be0d13b`。
- 原评测器SHA256：`e730fef8329fd3352c6e7c9884988ccb66f37c3ed0492ca84b0633ea44c6de1f`；原批次manifest SHA256：`620d6281ff10c18da66ddda2870cac445a190dee0f0faa803ab5574b2c9e0b11`。
- [训练日志](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/logs/train_stdout_137534.log)、[评测日志](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/logs/eval_stdout_137729.log)、[结果JSON](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/model/Cifar100_ResNet18_1004_binary_v1_H2/eval_137729.json)；结果JSON SHA256：`b44db407d901618a76cd4e682b52472d0fed864d217ef1564148e3a761923162`。
- [正式结果分析](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/结果分析/1004-cifar100-binary-v1_结果分析.md)、[原结果核验](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/run/1004-cifar100-binary-v1/preparation/results_20261005/verification.json)、[本次源码同步核验](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/1004/sync_h2_20261005/verification.json)。原结果核验包含作业成功、权重／源码／脚本／manifest身份及九项完整计数；本次不重复训练或GPU评测。
- 旧R2的26个轻量文件与[0917冻结入口](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/0917v1-cifar10/IJCV_KD/CIARD_Expansion_resnet18_cifar100/README.md)逐字一致，迁移映射保存在[old_r2_mapping.json](/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/1004/sync_h2_20261005/old_r2_mapping.json)。不能把旧教师0911 R1备份当作R2。

## 源码模板与复跑

本目录包含50份Python（25份当前代码及25份`frozen_source`），全部与已测H2相同；依赖保留原requirements.txt。该文件不是独立重建环境的完整锁文件：实际实现还依赖已验证环境中的robustbench，run_checks同时绑定其模型实现源码哈希，复跑时须保留对应环境约束。训练／评测入口为CIARD.py和attack_eval.py，配套train_a800.sbatch与eval_a800.sbatch。模板仅适配本目录工作／日志路径和job-name，保留a800/urgent、1×A800、4CPU、16GB、动态节点及无脚本时限。

**这是轻量源码快照，不能直接提交。** 未打包data/models资源链接、权重、model/logs或缓存。Python保留原身份/prefix、原评测权重绝对路径，以及run_checks对原组名、父级experiment_manifest.json和preparation校验文件的绑定；`frozen_source`参与源码哈希，不能省略。适配模板路径不等于重新建立这些绑定，不通过删除检查或复制虚假manifest使快照运行。

未来复跑须另建独立run，明确新的身份/prefix、评测路径、manifest和冻结哈希，建立公共资源链接及独立输出，再由用户手动提交。原已完成H2源码、日志和权重保持冻结；用户已授权将本次核验后的源码更新发布至GitHub；此前本地同步记录按历史时点保留，本次不提交训练或评测作业。

## 完整固定CFG

```json
{
  "clean_ce_transfer_ratio": 0.0,
  "clean_ce_preserve_loss": false,
  "natural_binary_weight": 2.0,
  "natural_binary_awp": false,
  "natural_binary_start": 120,
  "natural_binary_warmup": 40,
  "robust_kd_error_swap": false,
  "clean_ce_robust_projection": false,
  "training_views": 2,
  "awp_gamma": 0.003,
  "consistency_weight": 0.5,
  "method_start": 120,
  "method_warmup": 40,
  "consistency_temperature": 0.5,
  "target_mix_alpha": 0.2,
  "target_mix_start": 120,
  "target_mix_warmup": 40,
  "split_target_mix": true,
  "split_target_alpha": 0.25,
  "split_nontarget_alpha": 0.0,
  "push_soft": true,
  "push_feature": false,
  "push_gamma": 4.0,
  "push_T": 5.0,
  "push_eta": 0.3,
  "proj_dim": 128,
  "push_lambda": 0.08174,
  "push_warmup": 80,
  "push_require_robust_correct": true,
  "adaptive_weight": false,
  "adv_weight_floor": 0.35,
  "clean_ce_weight": 0.036067,
  "adv_ce_weight": 0.0,
  "ce_start": 158,
  "ce_warmup": 116,
  "clean_ce_robust_gate": true,
  "clean_ce_gate_tau": 0.682852,
  "clean_ce_gate_floor": 0.0,
  "adv_margin_weight": 0.0,
  "adv_margin_kappa": 0.0,
  "adv_margin_start": 120,
  "adv_margin_warmup": 80,
  "teacher_margin_weight": 0.011316,
  "teacher_margin_start": 120,
  "teacher_margin_warmup": 89,
  "teacher_margin_tau": 1.124788,
  "teacher_margin_cap": 1.428932,
  "teacher_margin_clean_gate": false,
  "teacher_margin_clean_tau": 1.0,
  "teacher_margin_clean_floor": 0.0,
  "teacher_margin_adv_gate": false,
  "teacher_margin_adv_tau": 1.0,
  "teacher_margin_adv_floor": 0.0,
  "teacher_margin_relative": false,
  "teacher_margin_relative_eta": 0.5,
  "teacher_margin_conflict_gate": true,
  "teacher_margin_conflict_threshold": 0.0,
  "teacher_margin_conflict_floor": 0.211797,
  "student_arch_adaptive_margin": false,
  "resnet_teacher_margin_weight": 0.005,
  "resnet_teacher_margin_start": 180,
  "resnet_teacher_margin_warmup": 80,
  "teacher_margin_per_sample_conflict": false,
  "teacher_margin_per_sample_tau": 1.0,
  "teacher_margin_per_sample_floor": 0.0,
  "robust_kd_reliable": false,
  "robust_kd_floor": 0.5,
  "capacity_aware": false,
  "capacity_xi": 1.0,
  "capacity_floor": 0.5,
  "capacity_start": 60,
  "ema_itt": false,
  "ema_decay": 0.99,
  "teacher_warmup": 50,
  "ema_use_start": 70,
  "attack_teacher_alpha": 0.0,
  "student_ema": true,
  "student_ema_decay": 0.999642,
  "eval_student_ema": true,
  "save_ema_as_student": true,
  "weight_averaging": false,
  "wa_alpha": 0.5,
  "pcgrad_teacher_margin": true,
  "pcgrad_start": 120
}
```
