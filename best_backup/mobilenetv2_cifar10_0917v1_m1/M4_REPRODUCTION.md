# CIFAR-10 MobileNet-V2：从M1复现0917v1 M4

本目录默认保存M1；M4为另一组已完成测试的参数配置，不需要另外保存重复源码。M4的23份Python与M1相比，仅CIARD.py和attack_eval.py不同。完整M4 CFG、23份源码SHA256和原始结果记录在仓库SYNC_MANIFEST.json的cifar10_batch_results中。

M1与M4各自同一个固定checkpoint的八项均严格高于论文，AA分别47.01%与46.70%；两者黑盒CW均66.14%，仅高论文.02pp。按用户选择以M1作为默认源码：M1的FGSM、PGDtrades、白盒CW和AA更高；M4的Clean、PGDsat、黑盒PGDtrades和Square更高，八项均值64.89%高于M1的64.83%（M1精确64.825%）。M4−M1九项依次为+.41/−.16/+.08/−.02/−.25/+.31/+.14/0/−.31pp。两者均未全面超过历史0909v1 M2，M1/M4本次过线不等于稳定全胜。

## 必要修改

从本目录复制到新的独立run目录，保留其余所有方法、模型、教师和评测文件。不要改写现有run/0917v1或本冻结备份。CIARD.py中仅修改以下配置和身份字段；下面列的是原已测M4身份，用于逐字核对：

```python
VARIANT_NAME = 'mobilenetv2_cifar10_v2_awp0p002_cr0p75'
prefix = 'Cifar10_MobileNetV2_0917v1_v2_awp0p002_cr0p75'
# CFG中的两项（不是替换整个CFG）：
'awp_gamma': 0.002,
'consistency_weight': 0.75,
```

attack_eval.py中同步修改：

```python
variant_name = 'mobilenetv2_cifar10_v2_awp0p002_cr0p75'
path = 'model/Cifar10_MobileNetV2_0917v1_v2_awp0p002_cr0p75/student_best.pth'
```

相对M1，只有awp_gamma从.001变为.002、consistency_weight从.50变为.75；两处variant与两处prefix/权重路径只是实验身份绑定。其余CFG全部保持M1，尤其training_views=2、method_start=120、method_warmup=40、consistency_temperature=.5、batch128、300epochs、seed0、push=.05、CE=.05、teacher margin=.01和EMA=.999。实际CFG仍由原训练入口打印。

真正重新训练时再为新实验设置唯一VARIANT_NAME/prefix（例如为上述两者追加_repro1），并在attack_eval.py中作同样替换，使path始终为model/<新prefix>/student_best.pth。训练的draw_file/model_dir随prefix自动派生。修改train_4090.sbatch和eval_4090_best.sbatch的cd、SBATCH output/error为新run的完整绝对路径，job-name可标注M4；保留原单4090资源配置，除非另行确认设备适配。

本源码包不带运行资源。新run需建立指向公共data、models的有效链接，建立model与logs/slurm目录，确认新prefix输出尚不存在。依赖沿用requirements.txt与ciard环境。用户手动训练成功后，再手动评测该新run固定EMA student_best；本说明不自动创建、提交或覆盖实验。

## M4原始证据与预期对照

- 已测来源：`/home/lixidong25/mycode/CIARD_Expansion/run/0917v1/mobilenetv2_cifar10_v2_awp0p002_cr0p75`。
- 训练/评测：`134639 / 135056`，均为`COMPLETED / 0:0`；固定EMA best epoch=254，完整10000张测试集。
- 权重：`model/Cifar10_MobileNetV2_0917v1_v2_awp0p002_cr0p75/student_best.pth`；checkpoint SHA256：`e4f5903f9f64596aaf79e16c0cb1dcf0957a9be62e49f7e1040b063aa890321c`。
- evaluator SHA256：`51384f95b849e148df9991a1f74c3d90fbd4e0e8635f9c31d4a2160aa382ca78`。
- 结果：`run/0917v1/mobilenetv2_cifar10_v2_awp0p002_cr0p75/model/Cifar10_MobileNetV2_0917v1_v2_awp0p002_cr0p75/eval_best_0909v1_135056.json`；历史JSON文件名保留0909v1，不代表来源批次。
- 日志：`run/0917v1/mobilenetv2_cifar10_v2_awp0p002_cr0p75/logs/train_stdout_134639.log`、`run/0917v1/mobilenetv2_cifar10_v2_awp0p002_cr0p75/logs/eval_best_stdout_135056.log`。
- 已核验CPU严格加载、完成标记、源码/权重/评测器哈希、九项正确数及日志/JSON一致性；原始权重和日志保留在原run。

M4完整九项（Clean、四白盒、三黑盒、AA）为90.02/60.71/50.75/53.34/48.76/67.98/81.41/66.14/46.70%，八项均值64.89%，七项攻击均值61.30%。这些数值来自M4自己的固定checkpoint；不能将M1权重改名为M4，也不能把M1的AA与M4的Clean拼接。复跑是否再次达到这些值需要实际完整评测确认。

沿用test-loader选模、未完全固定的攻击随机种子及PGDtrades步长.003；两者均为单seed结果。原M4 CIARD.py SHA256为`d8e1deae8807f61cbe8817e178fbd0b3dc8eff0b628c7295667bf6d42cd9893a`，attack_eval.py SHA256为`51384f95b849e148df9991a1f74c3d90fbd4e0e8635f9c31d4a2160aa382ca78`；按上述六项还原原身份后应与原23份Python逐字一致，新增复跑身份会相应改变这两个文件哈希。
