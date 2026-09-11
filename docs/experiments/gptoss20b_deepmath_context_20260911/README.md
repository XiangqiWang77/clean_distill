# GPT-OSS-20B：DeepMath privileged context 完整实验报告

实验日期：2026-09-11。范围：DeepMath 上已经完成的全部六组 context 对比。

**结果：`verify` 小幅提高最终答案准确率；`current` 和 `concise` 总准确率持平，但都损失了一部分 vanilla 原本答对的样本；`high` 和 `high_verify` 明显降低准确率，并大幅增加生成长度及截断。** 五种非 vanilla 设置中，一种提高、两种持平、两种降低。

下文的“之前 → 之后”指**同一 base model 不加额外 context → 使用指定 context / reasoning effort**。这些是为 OPSD 选择 teacher context 的生成实验；本实验还没有训练使用这些新 context 的 OPSD student，因此表中的数字不是训练前后 student 的成绩。

## 1. 我们测试了哪些设置

| 名称 | 相对 vanilla 的改动 | Reasoning effort | 具体要求 | 是否提供答案或参考解 |
|---|---|---|---|---|
| `base` / vanilla | 对照，无额外 system instruction | medium | 题目 + 所有组共用的逐步推理、最终答案格式要求 | 否 |
| `current` | 加入原来使用的 teacher 方法指令 | medium | 分解子目标、跟踪约束与不变量、检查边界情况、尽可能用另一条路线验证 | 否 |
| `verify` | 换成验证导向的指令 | medium | 检查关键计算、代回原约束、检查符号/整数条件/增根/漏计；验证完成后结束，不重启解题 | 否 |
| `concise` | 换成简洁解题指令 | medium | 选择一条路线完成精确计算，减少反复；卡住时换一次方法，预留检查和输出答案的空间 | 否 |
| `high` | 不加额外指令，只提高 reasoning effort | high | 使用模型原生 high reasoning 设置 | 否 |
| `high_verify` | 验证指令 + high reasoning | high | 与 `verify` 完全相同的额外指令，提高 reasoning effort | 否 |

`current` 是原有 context 对照；`verify`、`concise`、`high`、`high_verify` 是本轮四种替代设置。`high` 本身是推理设置变化，不是新增的题目信息。所有组都只看到题目；本报告不包含参考解题前缀实验。

## 2. 加入 context 之前和之后：准确率主表

每组使用相同的 **43 道题 × 2 个 decoding seeds = 86 次回答**。百分点（pp）按未四舍五入的准确率计算。

| 设置 | 之前：vanilla | 之后：该设置 | 答对数变化 | 准确率变化 | 判断 |
|---|---:|---:|---:|---:|---|
| `base` / vanilla | 96.51%（83/86） | 96.51%（83/86） | 0 | 0.00 pp | 对照 |
| `current` | 96.51%（83/86） | 96.51%（83/86） | 0 | 0.00 pp | **总分持平，没有净增益** |
| `verify` | 96.51%（83/86） | **97.67%（84/86）** | **+1** | **+1.16 pp** | **本轮最好，有小幅正向收益** |
| `concise` | 96.51%（83/86） | 96.51%（83/86） | 0 | 0.00 pp | **总分持平，平均回答更短** |
| `high` | 96.51%（83/86） | **69.77%（60/86）** | **−23** | **−26.74 pp** | **明显降低** |
| `high_verify` | 96.51%（83/86） | **72.09%（62/86）** | **−21** | **−24.42 pp** | **明显降低** |

从原有 `current` 改为 `verify`，也是 **96.51% → 97.67%，+1.16 pp**。在 high reasoning 下加验证指令，`high → high_verify` 为 **69.77% → 72.09%，+2.33 pp**，但仍远低于 vanilla。反过来，同一条验证指令从 medium 提到 high，`verify → high_verify` 为 **97.67% → 72.09%，−25.58 pp**。

以上“提高/持平/降低”描述这次完成实验的观测结果。`verify` 的收益只有一个回答，不能据此写成已经验证的训练后 OPSD 提升。

## 3. 哪些原来错的被救回，哪些原来对的反而变错

每次比较都匹配**同一题目、同一 decoding seed**。“救回”是 vanilla 错而 context 对；“损失”是 vanilla 对而 context 错。“错”包括答案不符和未输出可接受的最终答案。

| 设置 | 错 → 对：救回 | 对 → 错：损失 | 对 → 对：保留 | 错 → 错：仍错 | 净增加正确回答 |
|---|---:|---:|---:|---:|---:|
| `base` | 0 | 0 | 83 | 3 | 0 |
| `current` | 3 | 3 | 80 | 0 | 0 |
| `verify` | 2 | 1 | 82 | 1 | **+1** |
| `concise` | 3 | 3 | 80 | 0 | 0 |
| `high` | 1 | 24 | 59 | 2 | **−23** |
| `high_verify` | 1 | 22 | 61 | 2 | **−21** |

**总分持平不代表每道题的表现没变。** `current` 和 `concise` 都救回了 vanilla 的三次错误，同时又各自引入三次错误。`verify` 也有负向案例，只是两次救回超过了一次损失。`high` 损失了 vanilla 原本正确回答中的 **24/83 = 28.92%**；`high_verify` 损失 **22/83 = 26.51%**。

将同一题的两个 seed 合在一起，看该题答对次数是增加、减少还是不变：

| 设置 | 改善的题数 | 变差的题数 | 不变的题数 | 总题数 |
|---|---:|---:|---:|---:|
| `base` | 0 | 0 | 43 | 43 |
| `current` | 2 | 1 | 40 | 43 |
| `verify` | 2 | 1 | 40 | 43 |
| `concise` | 2 | 1 | 40 | 43 |
| `high` | 0 | 16 | 27 | 43 |
| `high_verify` | 1 | 16 | 26 | 43 |

两个表的统计单位不同。例如同一题可以在一个 seed 被救回、另一个 seed 被损失，合计后该题仍为“不变”。完整逐次对比见 [paired_scores.csv](paired_scores.csv)。

## 4. 输出长度与失败：哪些设置更耗预算、反而完成不了

生成上限固定为 **10,240 tokens**。平均 tokens 包括推理和最终回答，按全部 86 次回答计算，不只统计答对的回答。

| 设置 | 平均生成 tokens | 相对 vanilla 的长度变化 | 达到上限的回答 | 上限命中率 | 错误/未完成回答 |
|---|---:|---:|---:|---:|---:|
| `base` | 2,307.3 | 0.00% | 2/86 | 2.33% | 3/86 |
| `current` | 2,193.0 | −4.95% | 2/86 | 2.33% | 3/86 |
| `verify` | 2,679.7 | +16.14% | 2/86 | 2.33% | 2/86 |
| `concise` | **1,859.6** | **−19.40%** | 3/86 | 3.49% | 3/86 |
| `high` | 6,590.5 | **+185.64%** | 26/86 | **30.23%** | 26/86 |
| `high_verify` | 6,816.5 | **+195.44%** | 25/86 | **29.07%** | 24/86 |

`concise` 在相同总准确率下减少了约 19.4% 的平均生成 tokens，是本轮最清楚的长度收益；但其截断次数从 2 次升到 3 次，不能说它在所有样本上都更容易完成。`verify` 用约 16.1% 更多的平均 tokens 换来一次额外正确回答。这里测量的是生成长度，不是端到端时延或实际费用。

失败原因按已保存输出分类如下；每行加总等于该设置的错误/未完成次数。

| 设置 | 未进入 final channel | 已进入 final，但未给出可接受的最终结果 | 最终答案与标签不符 | 合计 |
|---|---:|---:|---:|---:|
| `base` | 2 | 0 | 1 | 3 |
| `current` | 2 | 0 | 1 | 3 |
| `verify` | 2 | 0 | 0 | 2 |
| `concise` | 3 | 0 | 0 | 3 |
| `high` | 25 | 1 | 0 | 26 |
| `high_verify` | 23 | 1 | 0 | 24 |

`high` 的 26 次失败、`high_verify` 的 24 次失败全部达到 token 上限，主要停留在 analysis 阶段。`high_verify` 另有一次达到上限的回答已经给出了正确最终结果，所以被计为正确：**截断率和错误率不是同一个指标**。

这些输出支持“在当前 10,240-token 预算下，high 两组完成答案的表现更差”。本实验没有测试更大预算，不能直接把结果推广到不限预算的 high reasoning。

## 5. 两个 decoding seeds 分别怎么样

每个 seed 为 43 道题。括号内是相对该 seed 的 vanilla 准确率变化。

| 设置 | Seed 20260911 | Seed 20260912 | 观察 |
|---|---:|---:|---|
| `base` | 95.35%（41/43） | 97.67%（42/43） | 对照 |
| `current` | 97.67%（+2.33 pp） | 95.35%（−2.33 pp） | 一个 seed 提升，另一个降低 |
| `verify` | 97.67%（+2.33 pp） | 97.67%（0.00 pp） | 一个 seed 提升，另一个持平 |
| `concise` | 95.35%（0.00 pp） | 97.67%（0.00 pp） | 两个 seed 总准确率均持平 |
| `high` | 65.12%（−30.23 pp） | 74.42%（−23.26 pp） | 两个 seed 均明显降低 |
| `high_verify` | 69.77%（−25.58 pp） | 74.42%（−23.26 pp） | 两个 seed 均明显降低 |

`verify − vanilla` 的配对题目 bootstrap 95% 区间为 **[−2.33, +4.65] pp**。因此可以称 `verify` 为本轮观测最好的候选，但收益仍小；high 两组的下降则在两个 seed 上都出现。这里的 seed 是生成采样 seed，不是不同的训练运行。

## 6. 具体案例：之前的回答与之后的回答

表中的“对/错”沿用本次最终答案与数据标签的一致性评分。短 ID 对应完整 query ID 的前八位；完整题目、答案、输出片段及来源哈希见 [cases.json](cases.json)。

| 案例 / seed | 题目类型 | 之前：vanilla | 之后：指定 context | 变化 |
|---|---|---|---|---|
| `0c2330f1` / 20260912 | 偏序集双向保序单射是否保证保序双射 | 10,240 tokens，未进入 final，计错 | `verify`：5,773 tokens，最终回答 No，与标签一致 | **救回一次未完成回答** |
| `0cff2780` / 20260911 | 单射算子有右逆时是否也有左逆 | 4,437 tokens，最终回答 No，与 Yes 标签不符 | `verify`：1,396 tokens，最终肯定结论，与标签一致 | **救回一次答案不符** |
| `0a534095` / 20260912 | 圆内接四边形构造的面积比 | 7,163 tokens，最终结果 1，与标签一致 | `verify`：10,240 tokens，未进入 final | **原本答对，加入验证后失败** |
| `0dc98910` / 20260912 | 含 Legendre symbol 的模素数求和 | 2,906 tokens，最终结果 −1 mod p，与标签一致 | `current`：4,074 tokens，最终结果 0 | **引入一次实质答案错误** |
| `0dc98910` / 20260912 | 同一模素数求和题 | 原本答对 | `concise`：10,240 tokens，未进入 final | **简洁指令仍可能引发未完成回答** |
| `0dc98910` / 20260911 | 同一模素数求和题，另一 seed | 5,314 tokens，最终结果 −1 mod p，与标签一致 | `high`：10,240 tokens；analysis 中出现 −1 / p−1，但未进入 final | **分析中出现正确值，最终回答仍未完成** |

这六个案例同时展示了正向和负向变化。特别是 `verify` 的全部三次正确性变化都已列出：两次救回、一次损失。算子题的“adjoint”表述存在模型对题意的不同解读；本表记录相对原标签的得失，不把该案例当成对题目或整段证明的独立数学认证。其他被接受的最终答案同样不代表整段推理已被证明正确。

## 7. 完整 prompt 与实验设置

所有组的 user message 都是原题后附相同的一句：

```text
Please reason step by step, and put your final answer within \boxed{}.
```

`base` 和 `high` 不添加额外 system instruction。其余设置使用以下原文。

### 原有 context：`current`

```text
Private reasoning-method instruction for the teacher: decompose the problem into explicit subgoals, track constraints and invariants, check boundary cases, and verify the chosen route against an independent alternative when possible. Use only the problem statement.
```

### 验证 context：`verify` / `high_verify`

```text
Solve the problem using exact mathematical reasoning. Before committing to the final answer, check the decisive calculation and substitute the candidate into the original constraints. Check signs, integer restrictions, extraneous roots, and whether all cases were counted. Correct any error you find. Once the answer is verified, finish; do not restart a completed solution. Use only the problem statement.
```

### 简洁 context：`concise`

```text
Find the shortest complete mathematical solution. Choose one promising method and carry it through with exact arithmetic. Avoid repeatedly reconsidering steps that are already established. If a method stalls, change it once and continue. Reserve enough space to check the calculation and state the final answer. Use only the problem statement.
```

| 实验项 | 设置 |
|---|---|
| 模型 | `openai/gpt-oss-20b` |
| 模型 revision | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| 数据来源 | `Leyiii/RLCSD`，`deepmath_filtered_level7_10/train.parquet` |
| 数据 revision | `33d7de919af5b03257ff92c30303fddf9afdda4a` |
| 原始 screen | 48 题，每组 96 次回答，共六组 576 次回答 |
| 主表使用的子集 | 原有审计排除 5 题后，43 题，每组 86 次回答，共 516 次 |
| 数据隔离 | 保留原有训练/开发使用的前 1,200 条 canonical records 不参与抽样，并检查已有训练流和 AMC/AIME query 排除项 |
| Sampling | temperature 0.6，top-p 0.95，top-k 20 |
| 配对 seed | 20260911、20260912；实际 sample seed 按题目身份生成，各 context 一致 |
| 生成上限 | 10,240 tokens |
| 模型权重 | 六组均为同一初始模型，没有 context 专属训练 |
| GPU | 已完成的 12 个 screen shards 均记录为 NVIDIA A40 |
| 最终评分 | `gptoss-final-answer-v2`，最终答案与标签一致性 |

可读取的全部 context 定义保存在 [context.json](context.json)，原始冻结设置保存在 [evidence/manifest.json](evidence/manifest.json)。

## 8. 评分审计、旧结果与当前结果的区别

主表使用完成后的 `SCREEN_REPORT_FINAL_V2`，不使用 provisional progress 表。原始生成内容没有因本报告而重跑或修改。

早期评分直接对整段输出运行 Math-Verify，会漏掉语义正确的 Yes/No 或带等号的答案，也可能把未完成 analysis 中的值当作已提交答案。已有审计改为检查 final channel，保留 44 条与 response hash 绑定的人工语义判断，并对全部六组统一排除五道标签错误或题意不清的题。这些审计发生在生成之后，所以这是一次诊断性 screen。

<details>
<summary>查看同一 43 题上的旧评分与最终审计评分（这是评分修正，不是模型提升）</summary>

下面将旧 scorer 的结果也限制在主表的同一 86 次回答上，保证分母一致。它们不是最初 48 题进度表的原始汇总值。

| 设置 | 同一批回答：旧全文评分 | 同一批回答：最终审计评分 |
|---|---:|---:|
| `base` | 56.98%（49/86） | 96.51%（83/86） |
| `current` | 55.81%（48/86） | 96.51%（83/86） |
| `verify` | 52.33%（45/86） | 97.67%（84/86） |
| `concise` | 59.30%（51/86） | 96.51%（83/86） |
| `high` | 55.81%（48/86） | 69.77%（60/86） |
| `high_verify` | 61.63%（53/86） | 72.09%（62/86） |

这张表只解释旧数字为什么不能与主表混用。不能把左右差值解释成 context 或 OPSD 的训练收益。逐条旧/新评分均保留在 [per_response_scores.csv](per_response_scores.csv)。

</details>

五个排除项为两道标签错误题、两道信息不足题和一道含义歧义题，完整 ID 与原有审计理由见 [excluded_queries.json](evidence/excluded_queries.json)。本报告沿用该固定排除集合，不新增按模型表现选择的排除项。

## 9. 对后续 OPSD 使用 context 的判断

| 设置 | 当前可支持的结论 | 后续使用判断 |
|---|---|---|
| `verify` | 本轮最终答案准确率最高，净多答对一次 | **优先作为 OPSD 的候选 teacher context** |
| `current` | 相对 vanilla 无净准确率提升；有三次负向变化 | 保留为原有 context 对照 |
| `concise` | 相同总准确率，平均生成长度下降 19.40% | 若目标包含减少生成 tokens，可作为候选；不能称为准确率提升 |
| `high` | 准确率下降 26.74 pp，输出明显变长 | 在当前 10,240-token 设置下不优先使用 |
| `high_verify` | 验证指令稍缓解 high 的下降，但仍低于 vanilla 24.42 pp | 在当前 10,240-token 设置下不优先使用 |

目标仍是比较 **使用该 context 训练后的 OPSD student 与 vanilla**。本次已经完成的是 teacher context screen；新 context 对应的 student 训练后成绩尚不存在。本轮 DeepMath 独立确认在生成前已取消，没有可补入的确认集结果。

## 10. 数据与证据索引

| 文件 | 内容 |
|---|---|
| [results.csv](results.csv) | 六组准确率、相对 vanilla 增益、配对得失、长度、截断与逐题变化 |
| [paired_scores.csv](paired_scores.csv) | 86 个题目 × seed 配对，含全部六组正确性与相对 vanilla 的变化 |
| [per_response_scores.csv](per_response_scores.csv) | 全部 516 次保留回答的评分、长度、失败原因、旧评分和 response hash |
| [cases.json](cases.json) | 第 6 节六个案例的完整题目、标签、最终回答和输出尾部片段 |
| [context.json](context.json) | 全部六组 context 定义、共同生成设置、候选选择用途 |
| [summary.json](summary.json) | 全部六组汇总及源文件 SHA256 |
| [screen_report.json](evidence/screen_report.json) | 原有完成版审计报告快照 |
| [manual_answer_review.json](evidence/manual_answer_review.json) | 原有 44 条人工最终答案语义审计 |
| [scoring_amendment.json](evidence/scoring_amendment.json) | 原有评分修订记录 |
| [verify_comparisons.json](evidence/verify_comparisons.json) | verify 的配对区间及分 seed 对比 |

本报告所有主表指标均从已保存的最终评分逐条重新汇总，并与原有完成版报告核对。
