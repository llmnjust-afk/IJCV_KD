# CIARD 扩展方法改进说明

中文论文讨论稿，2026 年 10 月 8 日。

本项工作是在 ICCV 2025 CIARD 的双教师循环对抗鲁棒蒸馏基础上，进一步改善学生对可靠知识的利用、自然准确率与鲁棒性的协调，以及蒸馏训练的稳定性。当前选定的四份代码覆盖 MobileNet-V2、ResNet-18 与 CIFAR-10、CIFAR-100 的四种组合；它们共享同一扩展框架，但存在明确的模块开关和训练配方差异。

**适合与师兄交流的结论是：四份代码属于同一个 CIARD 扩展框架，不能表述为只替换网络和数据集的完全相同算法。** 四份共同使用可靠 logit push、鲁棒门控自然监督、教师 margin 匹配、蒸馏目标驱动的对抗权重扰动、双视图对抗一致性和学生 EMA。两个 ResNet 配方还启用分解式蒸馏目标修正与 margin 梯度投影；CIFAR-100 ResNet 的 H2 配方进一步加入只作用于学生外层的自然二元监督。下面分别解释原方法、共同改进和这些差异。

当前成果可用于组织论文的方法和实验讨论，但四份结果支持的具体说法应保留边界：三个模型与数据集组合在本次完整测试的八项数值上全部超过论文公布值；CIFAR-10 ResNet 的 FGSM 为 61.80%，低于论文 61.88% 共 0.08 个百分点，其余七项及 AA 更高。接受这一小幅回落作为研究取舍，与宣称全部指标提升是两件不同的事。完整数值见后文，不将不同 checkpoint 的单项最好值拼接为一行。

## 代码范围与四份配方

| 本文简称 | 模型与数据集 | 当前来源 | 本目录源码 |
| --- | --- | --- | --- |
| M10 | MobileNet-V2，CIFAR-10 | 0917v1 M1 | [MobileNet CIFAR-10](CIARD_Expansion_mobilenetv2_cifar10/README.md) |
| R10 | ResNet-18，CIFAR-10 | 0917v1 R4 | [ResNet CIFAR-10](CIARD_Expansion_resnet18_cifar10/README.md) |
| M100 | MobileNet-V2，CIFAR-100 | 0914v1 M1 | [MobileNet CIFAR-100](CIARD_Expansion_mobilenetv2_cifar100/README.md) |
| R100 | ResNet-18，CIFAR-100 | 1004 H2 | [ResNet CIFAR-100](CIARD_Expansion_resnet18_cifar100/README.md) |

这里的“当前最好”指当前选定、已有完整结果的论文候选配方，不表示每个指标都达到全部历史实验中的最大值。例如 R10 相比历史 R5 更接近 FGSM 论文值，但历史 R5 的部分鲁棒指标及八项均值更高；R100 H2 补齐了相对论文的 Clean 缺口，也付出了相对部分历史配方的强攻击准确率代价。当前选择与历史最优单项应分开理解。

[CIARD](CIARD/README.md) 保存原会议版本；[process](process/README.md) 保存此前整理的历史代码与记录。`process/IMPROVEMENTS_SUMMARY.md` 是历史设计说明，其中的 Label Smoothing、Adaptive Temperature 叙事不能直接用来描述本次四份已测配方。本文以实际启用的配置和训练调用为准。

## 原 CIARD 已经做了什么

设学生为 S，自然教师为 N，鲁棒教师为 R。自然教师在干净增强图像上提供分类知识，鲁棒教师在学生生成的对抗样本上提供鲁棒知识。原代码已经包含两条蒸馏分支：学生 clean 输出学习自然教师，学生 adversarial 输出学习鲁棒教师。原版实际总目标对这两项采用 1:1 系数；代码虽然更新了名为 `weight` 的统计量，最终损失没有使用这些动态权重。

原方法也已经包含对错误自然教师的排斥。在对抗样本上，如果自然教师预测错误，原代码计算学生与该教师的 KL 项，按两条 KD 损失的数量级缩放后，从总损失中减去它。最小化这个目标，相当于鼓励学生远离错误的自然教师预测。因而，本次拓展的变化是重构 push 的选择条件、方向和强度控制，而不是首次引入“远离错误教师”的思想。

两教师输出的温度也不是这次新增的机制。原代码依据两条教师分布的熵比，按批次调节自然与鲁棒温度，并将温度限制在 1 至 10。学生侧 KD 使用温度 1，教师侧使用各自当前温度；实现按 batch 和类别取均值，保留 `log(q+1e-5)` 的写法，没有额外乘常见的温度平方因子。本文后续公式在解释思想时使用 KL 记号，复现时仍应遵循源码的数值表达和归约。

原 CIARD 的“循环”还体现在鲁棒教师更新上。训练先用学生生成对抗样本，随后在这些对抗样本上计算鲁棒教师的真实标签 CE；第 50 轮之后更新鲁棒教师参数，自然教师则保持固定。扩展继续使用这个 live 鲁棒教师，而不是重新定义一套教师训练算法。训练攻击也继续采用学生 CE 驱动的 PGD-10，预算为 8/255、步长为 2/255，起点加标准差 0.001 的 Gaussian 噪声。

这些机制可在原版 [CIARD.py](CIARD/CIARD.py) 的双 KD、温度更新、push 和教师更新部分，以及 [mtard_loss.py](CIARD/mtard_loss.py) 的 `robust_inner_loss_push` 中对应。因此论文的拓展贡献应从这些已有基础之上开始陈述。

## 四份代码共享的扩展框架

共同思路是先判断监督信号是否值得采用，再限制附加监督对原蒸馏目标的干扰，同时在输入增强和权重邻域中约束学生。这里“值得采用”通过教师是否正确、预测概率和分类 margin 等可计算条件表达；它是方法设计依据，不等于已经证明这些条件能保证每个样本或每项攻击的收益。

### 可靠且有界的错误类 push

扩展保留“自然教师在对抗输入上出错时应避免跟随它”的动机，但增加一个判断：鲁棒教师在同一对抗输入上必须预测正确。若两位教师都错，当前 push 不施加排斥信号，以减少缺少可靠参照时的任意移动。

设自然教师在对抗输入上的预测概率为 p_N=softmax(z_N)，错误 top-1 类别为 k_N，真实类别为 y。这里计算置信度的自然教师温度为 1，不使用自然 KD 的动态温度。实际权重为

\[
w_i=\mathbf{1}[k_N\ne y_i]\,
\mathbf{1}[\arg\max R(x_i^a)=y_i]\,
\sigma\!\left(\gamma_p(p_{N,\max}-p_{N,y_i})\right).
\]

四份代码均取 gamma_p=4。自然教师错得越有把握，sigmoid 权重越大；自然教师正确，或鲁棒教师不正确时，权重为零。上述判断和权重全部停止梯度。

新的 push 直接惩罚学生给自然教师错误 top-1 类别的概率：

\[
L_{\mathrm{push}}=\frac{1}{B}\sum_i w_i\,
\operatorname{softmax}(z_S(x_i^a)/T_p)_{k_N},\qquad T_p=5.
\]

它是加入总损失的非负、有界罚项，目标是降低特定错误类别的学生概率，而非通过减去整分布 KL 持续扩大距离。权重乘以 `min(epoch/80,1)` 逐步引入；MobileNet 最终系数为 0.05，ResNet 为 0.081740。代码函数虽然仍叫 `soft_feature_push_loss`，四份当前配置的 `push_feature=False`，实际只有上述 logit 项，没有特征投影或特征对比学习。

该修改可以解释为对原 CIARD 排斥机制的可靠性约束与幅度控制。它提供更受限制的纠正方向，但不能单凭这一公式就断言 Clean 或黑盒鲁棒性必然提升。实现见 [soft_push_weight 与 soft_feature_push_loss](CIARD_Expansion_mobilenetv2_cifar10/mtard_loss.py)。

### 由对抗 margin 调节的 clean CE

仅通过教师软标签学习时，学生与真实标签之间可能仍有偏差。扩展增加一项较小的 clean CE，并利用学生当前对抗分类间隔控制它的权重。定义学生 adversarial margin 为真实类 logit 减去最大其他类 logit：

\[
m_S^a=z_{S,y}(x^a)-\max_{k\ne y}z_{S,k}(x^a),\quad
g_i=\sigma\!\left(\operatorname{stopgrad}(m_{S,i}^a)/\tau_c\right),
\]

\[
L_{\mathrm{clean}}=\frac{1}{B}\sum_i g_i\,\mathrm{CE}(S(x_i),y_i).
\]

对抗 margin 较大的样本获得更强的 clean 标签监督；margin 较小或为负时，clean CE 被软性减弱。这样设计的动机是在细化自然决策边界时，尽量减少对尚不稳定的鲁棒边界的推动。这里并不是硬性筛选“只有对抗正确样本参与”，因为 sigmoid 在负 margin 上仍可能非零。

门控值 detach，因此 clean CE 不会通过改变 gate 反向操纵对抗 margin。四份配方都保留原自然 KD，再附加这项监督；系数、开始轮次和温度按模型配置不同。实现见 [base_losses](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)。

### 由可靠鲁棒教师指导的 margin 匹配

概率分布 KD 之外，扩展显式要求学生接近鲁棒教师的正分类间隔。设鲁棒教师在对抗样本上的 margin 为 m_R，取

\[
\bar m_R=\operatorname{clip}(m_R,0,c),\qquad
h_i=\mathbf{1}[m_{R,i}>0]\,\sigma(m_{R,i}/\tau_m),
\]

\[
L_{\mathrm{margin}}=\frac{1}{B}\sum_i h_i\,
\max(0,\bar m_{R,i}-m_{S,i}^a).
\]

只有鲁棒教师 margin 为正时才提供信号，目标上限 c 限制了所要求的间隔；学生达到目标后不继续推动。教师目标和门控均停止梯度，损失梯度作用于学生对抗 logits。它意在补充普通 KD 对显式决策间隔的约束，尤其适合讨论对 margin 的控制，但现有最终结果不单独证明某一种攻击的增益由它造成。

四份配置都启用这一项，两个 MobileNet 使用普通反传，两个 ResNet 还采用后文的梯度投影。源码中存在额外的 clean/adv 双稳定门控、相对 margin 等实验分支，当前四份均未启用；不要把这些分支写成共同方法。实现仍见 [base_losses](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)。

### 蒸馏目标驱动的对抗权重扰动

对抗权重扰动 AWP 把训练约束从输入邻域扩展到模型权重邻域。当前实现先复制学生作为代理模型，在已生成的对抗输入及对应 clean 输入上，计算两条 KD 的梯度；随后沿着使 KD 增大的方向，临时扰动学生权重。它的目标是让学生在附近较不利的权重位置上仍能完成蒸馏。

对每个可扰动的参数张量 W，扰动为

\[
v_W=\gamma(e)\frac{\lVert W\rVert_2}{\lVert g_W\rVert_2+10^{-12}}g_W,
\qquad
g_W=\nabla_W\operatorname{AvgViews}(L_{\mathrm{KD}}^a+L_{\mathrm{KD}}^n).
\]

只扰动维度大于 1 的权重张量，bias 和 BN 的一维参数不在 AWP 扰动范围内。这里实现的是归一化的一步梯度上升方向，没有额外进行多轮权重攻击优化。

学生随后在 W+v_W 上计算完整外层损失并反传，恢复原始 W 后才执行 SGD。因此 SGD 的 momentum 和 weight decay 使用原始权重，而不是临时扰动后的权重。原始教师 targets 与输入对抗样本在外层作为固定信号使用，不通过它们反传。

需要明确区分内外层：**AWP 内层仅使用 adversarial KD 与 natural KD，不包含 push、clean CE、teacher margin 或 JS。** 两个 ResNet 的 KD 内层包含其分解式目标修正；H2 的自然二元项仍不进入 AWP 内层。对应代码是 [perturb_weights 和 MethodRunner.step](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)，H2 的内层开关见 [awp_kd_loss](CIARD_Expansion_resnet18_cifar100/awp_consistency.py)。

AWP 是已有优化思想。本项工作的表述重点应放在它如何与双教师蒸馏目标、后续一致性和特定纠正项结合，不能将 AWP 本身写成本工作的原创。

### 两个独立增强视图的对抗一致性

从同一训练图像独立采样两次随机 crop 和 horizontal flip，得到两个 clean 视图，再分别生成学生 PGD 对抗输入。学生需要在这两个对抗视图上维持相近预测，从而对数据增强和对抗扰动共同造成的变化进行约束。

令两个学生对抗输出经过温度 T_c=0.5 后为 p_1、p_2，q=(p_1+p_2)/2，则

\[
L_{\mathrm{JS}}=\frac{1}{2}\mathrm{KL}(p_1\Vert q)
+\frac{1}{2}\mathrm{KL}(p_2\Vert q).
\]

两端均参与反传，不将某一个视图固定为 teacher。实现先对样本求均值，再将 JS 除以类别数 C 后乘一致性系数，与原 KD 的类别均值量级对应；没有额外温度平方缩放。四份都从第 121 轮改用两个视图，AWP 与一致性强度在第 121 至 160 轮线性增加，第 160 轮达到设定值。

两个视图的基本损失取平均，鲁棒教师的对抗 CE 也按视图平均；每个 batch 仍只做一次学生优化、一次符合日程的鲁棒教师优化，以及一次动态温度更新。这样可以区分“每个样本多视图”与“每个 batch 多次更新”，避免误述训练量。实现见 [EpochViews、js_divergence 和 MethodRunner.step](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)。

### 学生 EMA 与推理模型

学生每次 SGD 后更新其指数滑动平均 EMA 副本。学生参数与浮点 BN buffers 均做指数平均，整数 buffers 直接复制。EMA 不添加学生训练损失，主要改变用于评测和保存的学生状态。MobileNet 的 decay 为 0.999，ResNet 为 0.999642。

四份报告结果都来自单个 EMA 学生 checkpoint，不需要在推理时保留教师、AWP 代理或两视图训练流程，也不需要多模型集成。这里是**学生 EMA**；当前配置的 `ema_itt=False`，没有使用 EMA 鲁棒教师提供蒸馏标签。实现复用了名为 `ema_update_teacher` 的函数，因此判断作用对象应看调用参数，不能只看函数名。实现见 [EMA 更新函数](CIARD_Expansion_mobilenetv2_cifar10/mtard_loss.py) 和 [学生更新调用](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)。

## 共同目标与训练次序

用省略动态系数细节的形式，四份的外层目标可统一写为

\[
L_{\mathrm{outer}}=\operatorname{AvgViews}\left[
L_{\mathrm{KD}}^a+L_{\mathrm{KD}}^n
+\lambda_p(e)L_{\mathrm{push}}
+\lambda_c(e)L_{\mathrm{clean}}
+\lambda_m(e)L_{\mathrm{margin}}
+\lambda_b(e)L_{\mathrm{binary}}
\right]+\frac{\lambda_{\mathrm{JS}}(e)}{C}L_{\mathrm{JS}}.
\]

只有 R100 H2 的 lambda_b 非零；两个 ResNet 的 adversarial KD 使用后文分解式修正，且实际更新还包含梯度投影。因此这一表达是共同损失框架，不意味着四个实例的完整梯度完全相同。

训练过程可以按以下次序理解。前 120 轮使用单视图基础扩展训练，push 已按自身日程启用；第 121 轮以后，每个 batch 进入双视图 AWP 与一致性分支。对两个增强视图分别生成 PGD 输入并取得两教师输出，固定这些输入与 targets；用 KD 计算 AWP 方向；在受扰权重下计算两视图基本损失、JS 和适用的附加项；执行普通反传或 ResNet 的 margin 投影；恢复学生权重，再更新学生、学生 EMA 和鲁棒教师。各损失自己的启用轮次仍生效，例如 ResNet clean CE 从第 159 轮起才非零。

## 两个 ResNet 配方的额外方法

### 分解式蒸馏目标修正

鲁棒教师在 clean 输入与 adversarial 输入上的分布可能不同。两个 ResNet 配方只在鲁棒教师对 clean 输入预测正确时，利用其 clean 分布修正 adversarial KD。它使用的是**同一个鲁棒教师的两个输入输出**，不是将自然教师与鲁棒教师做额外混合。

修正把分布知识拆成两部分：真实类与其余类的二元概率关系，以及其余类内部的条件分布。令 q_a、q_c 为鲁棒教师在 adversarial/clean 输入上的温度概率，设

\[
q_{\alpha}=(1-\alpha)q_a+\alpha q_c,
\quad b(q)=(q_y,1-q_y),
\quad n(q)=\left\{\frac{q_k}{1-q_y}\right\}_{k\ne y}.
\]

当前配方对二元目标采用 alpha_t=0.25，对非目标类条件分布采用 alpha_n=0。前者允许 clean 教师信息修正真类质量，后者继续保留 adversarial 教师对错误类别关系的判断。alpha 都乘以 start120/warmup40 的预热系数；若鲁棒教师 clean 预测不正确，则保留原 adversarial KD。

实现不是简单把整个 target 换成 `q_0.25`。它从参考整体混合 `q_ref=q_0.20` 的原始 KD 项出发，用二元 KL 差与条件 KL 差进行校正：

\[
L_a=L_{\mathrm{ref}}+\frac{1}{C}\left[
B(q_{\alpha_t})-B(q_{\alpha_{ref}})
+(1-q_{\alpha_{ref},y})\{N(q_{\alpha_n})-N(q_{\alpha_{ref}})\}
\right],
\]

其中 B(q)=KL(b(q)||b(p_S))，N(q)=KL(n(q)||n(p_S))，公式省略 batch 均值。参考项沿用原 `log(q+1e-5)` 数值形式，校正项使用标准 KL；非目标条件项由参考分布的其他类总质量加权。所有 teacher target、混合资格和权重停止梯度。实际实现用 log-space 计算其他类质量，避免接近概率 1 时直接相减的数值问题。

该方法把“真类是否值得加强”和“其余类别之间如何区分”分开控制，是两个 ResNet 已测配方的组成部分。两个 MobileNet 当前没有启用，尽管其目录也保留 helper 文件。实现见 [split_target_mix_loss](CIARD_Expansion_resnet18_cifar10/split_target_mix.py) 及 [kd_losses 调用](CIARD_Expansion_resnet18_cifar10/awp_consistency.py)。

### 对 teacher margin 的单边梯度投影

附加 margin 监督与其余目标可能要求不同更新方向。两个 ResNet 将 teacher-margin 的梯度单独取出，把它与其余外层目标梯度比较。对每个参数张量，记其余目标梯度为 g_b，margin 梯度为 g_m；当两者内积小于零时，执行

\[
g_m'=g_m-\frac{\langle g_m,g_b\rangle}
{\lVert g_b\rVert_2^2+10^{-12}}g_b,
\qquad g=g_b+g_m'.
\]

没有冲突时直接相加。它只修正 margin 分支，属于逐参数张量的单边 PCGrad 风格投影，并非全模型统一投影，也不是对两个梯度做对称处理。这里的 base 包含其余实际启用的外层损失，包括 JS；H2 中还包含自然二元监督。

该操作意在削弱 margin 信号在局部梯度层面与原目标的冲突，但不保证 SGD 含 momentum、weight decay 后每一步都降低 base loss，也不保证长期测试性能单调改善。PCGrad 本身属于已有梯度协调思想，本工作应说明这里的作用对象与集成方式。

另一个容易误读之处是 ResNet 配置中的 `teacher_margin_conflict_gate=True`。历史代码计算了 `conflict_scale`，却没有把它乘进有效损失或梯度；当前双视图分支也明确保留这个语义。因此不能将该标志写成独立有效贡献。真正启用的冲突处理是上面的 margin PCGrad。实现见 [backward_student](CIARD_Expansion_resnet18_cifar10/awp_consistency.py)。

## CIFAR-100 ResNet H2 的自然二元监督

H2 进一步关注一个与鲁棒 margin 门控不同的问题：自然教师已经能正确分类，而学生在 clean 输入上仍然出错。对这类样本，保留全部原损失，再增加真实类与其余类之间的二元 KL，以加强自然教师可提供的真类纠正信号。

设 q_N=softmax(z_N/T_nat)，p_S=softmax(z_S)，当前自然 KD 温度为 T_nat。样本资格必须同时满足

\[
a_i=\mathbf1[\arg\max z_N=y_i]\,
\mathbf1[\arg\max z_S\ne y_i]\,
\mathbf1[q_{N,y_i}>p_{S,y_i}].
\]

自然教师正确还不够：学生必须确实 clean 分类错误，而且教师真类概率必须高于学生。二元目标、资格 mask 和比较中的学生概率全部 detach；损失本身仍通过学生二元概率反传。

\[
L_{\mathrm{binary}}=\frac{1}{BC}\sum_{i=1}^{B}a_i
\mathrm{KL}\!\left((q_{N,y_i},1-q_{N,y_i})\,\Vert\,
(p_{S,y_i},1-p_{S,y_i})\right).
\]

这里 B 是完整 batch 的样本数，C=100。没有按命中的样本数重新归一化，因此命中较少时该项整体影响自然减小。新增项不乘原 clean CE 的鲁棒 margin 门控，使满足上述条件的自然错例能够获得独立的纠正信号。实现使用真类 logit 与其他类 logsumexp 构造二元 log-probabilities；教师使用自然 KD 当前温度，学生使用温度 1，没有 T² 乘子。

H2 的最终权重为 2，start=120、warmup=40，第 121 轮首次非零，第 160 轮满强度。它只进入学生外层优化，`natural_binary_awp=False`；不替换原自然 KD、不改变循环教师 CE、不加入动态温度和旧损失统计的更新依据，也不加入 AWP 内层。它发生在每个增强视图的 clean 输出上，随后随外层损失按视图平均。mask 所用学生 clean 输出与该外层前向一致，因此在 AWP 阶段来自临时受扰权重下的学生。

这与 ResNet split KD 虽然都使用“真类／其余类”分解，但监督来源与目的不同：split KD 调节鲁棒教师 adversarial KD 的目标结构；H2 附加项利用自然教师修正学生 clean 错例。它是当前 R100 H2 的专用附加模块，不能据现有四行结果声称四个组合都从该模块获益。实现见 [natural_binary_loss](CIARD_Expansion_resnet18_cifar100/clean_supervision.py) 和 [外层接入与 AWP 开关](CIARD_Expansion_resnet18_cifar100/awp_consistency.py)。

## 四份实际配置与复现差异

### 模块与关键系数

表中的 start/warmup 表示从 `epoch>start` 开始非零，在 `start+warmup` 轮达到完整强度。所有配方均使用双视图、method_start=120、method_warmup=40、JS 温度 0.5、push 温度 5、push 置信度系数 4、push warmup=80。

| 配置项 | M10 | R10 | M100 | R100 H2 |
| --- | ---: | ---: | ---: | ---: |
| AWP gamma | 0.001 | 0.002 | 0.002 | 0.003 |
| JS 权重 | 0.50 | 0.75 | 0.50 | 0.50 |
| push 权重 | 0.05 | 0.081740 | 0.05 | 0.081740 |
| clean CE 权重 | 0.05 | 0.036067 | 0.05 | 0.036067 |
| clean CE start/warmup | 120/80 | 158/116 | 120/80 | 158/116 |
| clean gate tau | 2 | 0.682852 | 2 | 0.682852 |
| teacher margin 权重 | 0.010 | 0.011316 | 0.010 | 0.011316 |
| teacher margin start/warmup | 140/80 | 120/89 | 140/80 | 120/89 |
| teacher margin tau | 2 | 1.124788 | 2 | 1.124788 |
| teacher margin cap | 2 | 1.428932 | 2 | 1.428932 |
| split KD 真类/非目标 alpha | 关闭 | 0.25/0 | 关闭 | 0.25/0 |
| split KD 参考 alpha | 不适用 | 0.20 | 不适用 | 0.20 |
| teacher-margin PCGrad | 关闭 | 开启 | 关闭 | 开启 |
| 自然二元外层权重 | 0 | 0 | 0 | 2 |
| 学生 EMA decay | 0.999 | 0.999642 | 0.999 | 0.999642 |

四份的特征 push、GradNorm 风格自适应 KD 权重、容量门控、EMA 鲁棒教师、混合教师梯度攻击、独立 adversarial CE、普通 adversarial margin、相对 teacher margin 和逐样本冲突分支都未启用。H2 中的 error swap、clean CE 梯度投影、CE 门控转移也都关闭。保留这些函数不表示当前结果使用了它们。

### 教师与输入接口

| 数据集 | 鲁棒教师 | 自然教师 | 输入约定 |
| --- | --- | --- | --- |
| CIFAR-10 | WRN-34-10，`models/model_cifar_wrn.pt` | ResNet-56，`models/nat_teacher_checkpoint/cifar10_resnnet56.pth` | raw [0,1]；这两份教师权重不使用额外外部 Normalize |
| CIFAR-100 | WRN-70-16，`models/cifar100_linf_wrn70-16_without.pt` | 原始包 WRN-22-6，`models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth` | raw [0,1]；保留教师模型内部 CIFAR-100 归一化 |

同一数据集的两种学生配方采用相同的教师架构与初始权重；各独立实验的鲁棒教师分别循环更新，自然教师保持固定。学生架构仍各自保持 MobileNet-V2 或 ResNet-18。教师文件的名称、形状、hash 和输入接口必须配套；不能只根据通用文件名换用其他网络定义。

CIFAR-100 自然教师曾从较早实验使用的微调权重换为原始包权重。已有[教师诊断](../../../结果分析/0914v1_教师诊断_4090与3090对比.md)在现有输入配置下记录旧/新自然教师 Clean 为 56.62%/76.49%。这解释了为何当前两份 CIFAR-100 代码采用原始包教师，但不等于证明旧权重退化的根因，也不能把教师更换的收益全部归给某个新损失。鲁棒教师诊断与论文之间的差距同样应与方法贡献分开讨论。

### 训练终点与 checkpoint 选择

四份均使用 seed0、50,000 张训练图像、batch size 128、PGD-10 和同一类原始学习率日程。M10/R10 历史训练使用单张 4090；M100 历史训练使用两张 4090，全局 batch128、每卡64；H2 使用单张 A800。硬件与每卡 batch 会影响 BN 和数值轨迹，后续跨配置比较不能只归因为模型结构。

| 配方 | 总训练轮数 | 报告 checkpoint | 选模规则 | 评测随机种子记录 |
| --- | ---: | --- | --- | --- |
| M10 | 300 | epoch250 EMA `student_best.pth` | 历史 test-loader 上 `(Clean+PGD proxy)/2` 选 best | 未全部显式固定，JSON 为 null |
| R10 | 300 | epoch252 EMA `student_best.pth` | 同上 | 未全部显式固定，JSON 为 null |
| M100 | 300 | epoch230 EMA `student_best.pth` | 同上 | 未全部显式固定，JSON 为 null |
| R100 H2 | 190 | 固定 epoch190 EMA `student_epoch190.pth` | 本次训练不读取 test 选 checkpoint | 各指标 seed0 |

前三份的 250、252、230 是完整 300 轮训练中选出的 best 来源轮次，不是预先规定的训练终点。H2 则沿原 300 轮学习率函数训练至 190 轮，不把学习率日程压缩为 190 轮；它不存在同一训练轨迹内的 test-loader 选 best。H2 的配方和终点来自已有研究过程，因此也不能扩大为“整个研究从未参考过测试结果”。

R10 源码仍保留训练后的权重平均 WA 分支，但本表评测的是 EMA `student_best.pth`，没有采用 WA 权重。H2 的 WA 关闭。不能将权重平均或模型集成作为这四行成绩的来源。

## 四份完整测试结果

每行来自一个固定 checkpoint，在完整 10,000 张测试图像上评测。单位为准确率百分比；括号内是相对同模型、同数据集论文 CIARD 公布值的百分点差。指标顺序保持项目原有八项：Clean、四项白盒、三项黑盒。Square 沿用原表的黑盒归类，实际是直接查询学生模型的攻击，与教师生成后迁移的黑盒 PGDtrades/CW 应区分。

### CIFAR-10

| 配方 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 MobileNet-V2 | 89.51 | 59.10 | 47.67 | 50.71 | 46.88 | 66.66 | 80.01 | 66.12 |
| M10 0917 M1 | 89.61 (+0.10) | 60.87 (+1.77) | 50.67 (+3.00) | 53.36 (+2.65) | 49.01 (+2.13) | 67.67 (+1.01) | 81.27 (+1.26) | 66.14 (+0.02) |
| 论文 ResNet-18 | 88.87 | 61.88 | 51.70 | 54.46 | 50.61 | 66.28 | 80.03 | 64.79 |
| R10 0917 R4 | 89.10 (+0.23) | 61.80 (-0.08) | 52.06 (+0.36) | 54.86 (+0.40) | 51.24 (+0.63) | 67.28 (+1.00) | 80.77 (+0.74) | 65.62 (+0.83) |

M10 的八项均高于论文，最小余量是黑盒 CW 的 +0.02pp；R10 的七项更高，FGSM 低 0.08pp。完整 10k 上 0.08pp 对应 8 个样本的正确数差；它可以如实作为方法的一个小幅回落报告，不能在保留该结果时改写为八项全胜，也不能在没有重复实验时断言它必然属于随机噪声。来源为 [0917v1 结果分析](../../../结果分析/0917v1_结果分析.md)。

### CIFAR-100

| 配方 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 MobileNet-V2 | 66.72 | 33.56 | 27.02 | 28.95 | 25.54 | 42.70 | 50.85 | 42.85 |
| M100 0914 M1 | 66.91 (+0.19) | 34.57 (+1.01) | 28.61 (+1.59) | 30.06 (+1.11) | 27.28 (+1.74) | 44.82 (+2.12) | 53.29 (+2.44) | 43.67 (+0.82) |
| 论文 ResNet-18 | 65.73 | 34.47 | 28.05 | 29.45 | 24.43 | 42.29 | 49.76 | 41.44 |
| R100 1004 H2 | 65.96 (+0.23) | 34.52 (+0.05) | 29.76 (+1.71) | 30.78 (+1.33) | 26.66 (+2.23) | 43.52 (+1.23) | 51.90 (+2.14) | 42.29 (+0.85) |

这两个配方在本次结果中均为八项数值高于论文。H2 最小余量是 FGSM 的 +0.05pp；它的“八项过线”并不等于全面超过历史 R2 或其他强攻击更高的配方。M100 使用原始 4090 评测 134582，不混入之后同权重 A800 复评或重新训练的数值。来源为 [0914v1 CIFAR-100 结果分析](../../../结果分析/0914v1_cifar100_结果分析.md) 和 [1004 H2 结果分析](../../../结果分析/1004-cifar100-binary-v1_结果分析.md)。

### AutoAttack 与均值

七项攻击均值排除 Clean；八项均值包含 Clean；两者都不包含 AA，也都不是多种攻击联合后的最坏情况准确率。均值从原始整数正确数计算，最终按十进制四舍五入保留两位；差值在舍入前相减，因此有时不能由表内两个已舍入值直接相减得到。M10 的八项精确均值为 64.825%，显示为 64.83%。

| 配方 | AA | AA 论文差 | 七项攻击均值 | 七项均值论文差 | 八项均值 | 八项均值论文差 | 八项严格更高 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M10 | 47.01 | +0.70 | 61.28 | +1.69 | 64.83 | +1.49 | 8/8 |
| R10 | 49.10 | +0.22 | 61.95 | +0.55 | 65.34 | +0.51 | 7/8 |
| M100 | 24.95 | 不适用 | 37.47 | +1.55 | 41.15 | +1.38 | 8/8 |
| R100 H2 | 24.22 | 不适用 | 37.06 | +1.36 | 40.67 | +1.22 | 8/8 |

CIFAR-10 论文 AA 为 MobileNet-V2 46.31%、ResNet-18 48.88%。论文补充材料的该 AA 表针对 CIFAR-10，不能把其阈值用于 CIFAR-100，因此后者只报告 AA 绝对值。论文 Clean/白盒分别取[补充材料](../../../ICCV_CIARD_Supplementary.pdf)的对应模型 Table 1/2、黑盒取[主文](../../../Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf) Table 5/6 的 Robust 列；MobileNet CIFAR-10 Clean 使用主比较表的 89.51%，不改用 AA 表里的较低 Clean。

## 论文贡献可以如何概括

方法主线可以概括为：**在 CIARD 的双教师循环蒸馏上，根据监督可靠性约束知识传递，并结合蒸馏目标驱动的权重邻域训练与对抗视图一致性，提高轻量学生对自然和鲁棒知识的协调利用。** 这条主线覆盖四份代码的共同部分，也解释了各附加模块为何围绕教师信息、学生边界和优化冲突展开。

展开时可以分成三个层次。第一层是可靠监督：将原错误教师 push 改为有界错误类惩罚，配合对抗 margin 门控的 clean 标签监督与有上限的教师 margin 匹配。第二层是训练稳定性：将 KD 条件 AWP、独立增强视图的对抗一致性以及学生 EMA 接入循环蒸馏。第三层是配方中的进一步纠正：两个 ResNet 的分解式目标修正与 margin 梯度协调，以及 H2 对自然错例的二元监督。这三个层次应对应实际配置表，不把只在某个实例启用的模块伪装成所有实验都使用的统一完整算法。

AWP、JS 一致性、EMA、PCGrad 以及 KL 的二元分解都不是仅凭本次组合就可宣称原创的基本工具。论文可以讨论本工作针对 CIARD 问题的具体组合、监督条件、作用位置和实证表现；有关与已有研究的差异与新颖性，需要在正式相关工作中逐项核对。本文不把历史代码注释中的参考标签或失败原因推测直接当作已完成的文献论证。

现有四份最终结果证明的是对应完整配方在本次评测中的整体表现。它们不足以单独量化每个模块的因果收益，也不足以证明某种模块组合对所有网络都有效。例如 H2 增强自然监督后补齐本次八项论文比较，不代表二元项单独造成全部提升；同批强度对比与历史控制之间也须区分。这些限制决定论文措辞的强度，不改变当前可以先整理方法、已有实验与结论的工作范围。

## 结果解释与使用边界

所有数值均是单训练 seed 的已完成结果，小于 0.1pp 的差异应按观测值陈述，不称为统计显著、稳定复现或必然噪声。前三份历史 test-loader 参与选 checkpoint，存在测试集选择偏差；H2 固定终点改善了该次训练内的选模方式，但四份整体不能统一标记为严格相同的 test-only 流程。

评测沿用本项目冻结的历史攻击实现：L-infinity 预算 8/255，PGDsat 为 20 步、步长 2/255，PGDtrades 为 20 步、步长 0.003，CW 为 30 步、步长 2/255，Square 为 100 queries，AA 单列。名为 PGDtrades 的历史入口使用 CE 攻击目标，不应根据名称写成 KL-TRADES 攻击；0.003 与论文文字的 2/255 存在差异，因此“高于论文公布数值”不等于已经完成严格同协议复现。前三份随机攻击未全部显式定 seed，H2 记录每指标 seed0，这一差异也需保留。

八项均值只帮助概括趋势，不替代单项结果或 AA。R10 的 FGSM 回落、M10 的黑盒 CW 小余量与 H2 的 FGSM 小余量都应留在主结果讨论中。用“总体取得改善，同时存在少数指标权衡”描述当前证据，比“所有设置全面提升”更准确。

本目录是与既有结果一一对应的源码归档，源码已于2026-10-08发布至GitHub（145bfb8）。现补充四组成功训练与评测的24份原始日志，共7,924,782字节（约7.9 MB），保留stdout及对应Slurm `.out/.err`；stdout与对应`.out`逐字节一致，8份`.err`均为空。7个锁文件和CIFAR-100 MobileNet失败作业133782的3份日志未纳入，原文件仍在原run。教师、数据、checkpoint、结果JSON与完整报告仍在仓库外，相关链接仅供本地追溯；本文训练和评测日志链接改为仓库内副本，原日志正文与绝对路径保持不变。源码中的运行身份、prefix、旧脚本路径及 H2 manifest 绑定应按各入口 README 理解，不能因复制到新目录就认为可直接重提；后续若需要复跑，应另行准备独立运行目录并由用户手动提交。本次方法说明没有引入新的训练或评测前置要求。

## checkpoint 与原始证据

下列训练与评测在对应结果归档中均已核验 `COMPLETED/0:0`，每个评测包含完整 10k 九项结果。此处训练与评测日志可在仓库中阅读；固定权重、结果JSON及其他仓库外证据链接仅在本地工作区有效。数字采用指定评测记录，不用后续复评替换。

| 配方 | 固定权重（仅本地） | 训练日志 | 评测日志 | 结果 JSON（仅本地） |
| --- | --- | --- | --- | --- |
| M10 | [epoch250 EMA](../../../run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/model/Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50/student_best.pth) | [134636](CIARD_Expansion_mobilenetv2_cifar10/logs/train_stdout_134636.log) | [135053](CIARD_Expansion_mobilenetv2_cifar10/logs/eval_best_stdout_135053.log) | [整数计数与身份](../../../run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/model/Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50/eval_best_0909v1_135053.json) |
| R10 | [epoch252 EMA](../../../run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/model/Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75/student_best.pth) | [134643](CIARD_Expansion_resnet18_cifar10/logs/train_stdout_134643.log) | [134905](CIARD_Expansion_resnet18_cifar10/logs/eval_best_stdout_134905.log) | [整数计数与身份](../../../run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/model/Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75/eval_best_0909v1_134905.json) |
| M100 | [epoch230 EMA](../../../run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50/model/Cifar100_MobileNetV2_0914v1_natorig_awp0p002_cr0p50/student_best.pth) | [133792](CIARD_Expansion_mobilenetv2_cifar100/logs/train_stdout_133792.log) | [134582](CIARD_Expansion_mobilenetv2_cifar100/logs/eval_best_stdout_134582.log) | [整数计数与身份](../../../run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50/model/Cifar100_MobileNetV2_0914v1_natorig_awp0p002_cr0p50/eval_best_0909v1_134582.json) |
| R100 H2 | [固定 epoch190 EMA](../../../run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/model/Cifar100_ResNet18_1004_binary_v1_H2/student_epoch190.pth) | [137534](CIARD_Expansion_resnet18_cifar100/logs/train_stdout_137534.log) | [137729](CIARD_Expansion_resnet18_cifar100/logs/eval_stdout_137729.log) | [整数计数与身份](../../../run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/model/Cifar100_ResNet18_1004_binary_v1_H2/eval_137729.json) |

| 配方 | checkpoint SHA256 |
| --- | --- |
| M10 | `3956b121747007143f5a9ddd561dafe080b197a4670f95d20ee831551dc8bc7e` |
| R10 | `208cc7dbe1a4158c8c6d3a41f0f442e3952dd92cf011f3419e00edb9845694e8` |
| M100 | `5df58522edfee576f61ecd102be396d9e893d447347c92d89f04bf183f8ece1e` |
| R100 H2 | `e32461425fc112b763c6650c65756979e8ca75c3342119acbe4363075be0d13b` |

阶段性汇总中的“当时尚未同步”等旧文字按记录日期解释；当前四入口选择以[本包 README](README.md)和[跨版本台账](../../../LOCAL_VERSION_MANAGEMENT.md)为准。原始结果报告和 JSON 保留原实验身份，用于追踪数值来源。

## 源码阅读索引

先从各入口 `CIARD.py` 的 CFG 与 epoch 循环阅读，尤其注意第 121 轮以后通过 `MethodRunner.step` 执行并 `continue`，后面保留的单视图代码不再是这个阶段的有效训练路径。不能仅搜索到一个函数或开关名就判断它参与了当前结果。

| 阅读目的 | 文件与位置 |
| --- | --- |
| 原会议方法的 KD 温度 push 循环教师 | [原版 CIARD.py](CIARD/CIARD.py)，重点看 200 行起的训练损失与教师更新；[原版 mtard_loss.py](CIARD/mtard_loss.py) 的 `robust_inner_loss_push` |
| 四份固定参数与入口 | [M10 CFG](CIARD_Expansion_mobilenetv2_cifar10/CIARD.py)、[R10 CFG](CIARD_Expansion_resnet18_cifar10/CIARD.py)、[M100 CFG](CIARD_Expansion_mobilenetv2_cifar100/CIARD.py)、[R100 CFG](CIARD_Expansion_resnet18_cifar100/CIARD.py) |
| 可靠 push 与 EMA 数值实现 | [mtard_loss.py](CIARD_Expansion_mobilenetv2_cifar10/mtard_loss.py)，`soft_push_weight`、`soft_feature_push_loss`、`ema_update_teacher` |
| 共同 KD 门控 CE margin AWP JS 更新次序 | [awp_consistency.py](CIARD_Expansion_mobilenetv2_cifar10/awp_consistency.py)，`kd_losses`、`base_losses`、`perturb_weights`、`js_divergence`、`MethodRunner.step`；M10/R10/M100 此文件相同 |
| ResNet 分解式目标修正 | [split_target_mix.py](CIARD_Expansion_resnet18_cifar10/split_target_mix.py)，`split_target_mix_loss`；由 `kd_losses` 根据 CFG 调用 |
| ResNet 逐参数 margin 投影 | [awp_consistency.py](CIARD_Expansion_resnet18_cifar10/awp_consistency.py)，`backward_student` |
| H2 自然二元监督 | [clean_supervision.py](CIARD_Expansion_resnet18_cifar100/clean_supervision.py)，`natural_binary_loss`；[awp_consistency.py](CIARD_Expansion_resnet18_cifar100/awp_consistency.py) 的 `base_losses` 与 `awp_kd_loss` 分别说明外层加入和内层关闭 |
| H2 固定终点与训练状态 | [CIARD.py](CIARD_Expansion_resnet18_cifar100/CIARD.py) 的 `SELECTION_PROTOCOL`、`epochs=190` 与保存部分；[training_state.py](CIARD_Expansion_resnet18_cifar100/training_state.py) 保存状态身份 |
| 最终评测身份与固定权重 | [M10 evaluator](CIARD_Expansion_mobilenetv2_cifar10/attack_eval.py)、[R10 evaluator](CIARD_Expansion_resnet18_cifar10/attack_eval.py)、[M100 evaluator](CIARD_Expansion_mobilenetv2_cifar100/attack_eval.py)、[R100 evaluator](CIARD_Expansion_resnet18_cifar100/attack_eval.py) |
