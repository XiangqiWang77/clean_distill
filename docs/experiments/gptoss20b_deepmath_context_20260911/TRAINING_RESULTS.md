# GPT-OSS-20B：verify privileged context 训练结果

四组 DeepMath 训练均已完成，Slurm 作业 `25974303` 正常退出（`COMPLETED`, `0:0`）。两张 B200 总分配时间 **3 小时 9 分 22 秒**，共完成 **160 次有效参数更新**。

**当前还没有这四组 checkpoint 与 vanilla 的统一准确率评测结果，不能宣称 OPSD 或 LGSD 提升。** 下表记录训练过程；长度截断不是准确率指标，蒸馏 loss 也不能直接跨方法比较。

| 方法 | Episodes | 累计 episode 耗时 | 平均生成 tokens | 达到长度上限 | 最终蒸馏 loss | 有效更新 |
|---|---:|---:|---:|---:|---:|---:|
| opsd_16 | 16 | 15.74 分钟 | 2819.1 | 0/16 (0.00%) | 0.004742 | 16/16 |
| lgsd_16 | 16 | 17.76 分钟 | 3137.9 | 1/16 (6.25%) | 0.003700 | 16/16 |
| opsd_64 | 64 | 68.94 分钟 | 3075.8 | 2/64 (3.12%) | 0.018181 | 64/64 |
| lgsd_64 | 64 | 83.48 分钟 | 3663.0 | 8/64 (12.50%) | 0.003860 | 64/64 |

## 目前观察到的好与坏

- 四组 loss 都是有限值，每一步均有非零参数更新；未发生 GPU 内存溢出或训练异常。
- LGSD 的 teacher target KL 均不超过配置的 0.004。这个约束针对 teacher target projection，并不等于训练后模型一定更准确。
- 在 64 episode 训练中，LGSD 的长度上限命中率为 12.50%，OPSD 为 3.125%，LGSD 高 9.375 个百分点；两组 16 episode 分别为 6.25% 和 0%。这是当前可确认的生成完成情况差异。
- 所有 160 次生成都启用 repetition_penalty=1.1。仍有长输出达到上限；仅凭截断不能判定发生了重复循环。

## 使用的 context 和训练设置

`verify` 是前次 DeepMath context screen 中观察到正向变化的 teacher 指令：84/86（97.67%），vanilla 为 83/86（96.51%），差值 +1.16 个百分点。该旧结果是 teacher 提示筛选结果，不能当作本次训练后 student 的成绩。完整前后明细见 [context 报告](README.md)。

该指令只出现在 teacher prompt，student 只看到普通题目。模型固定为 GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`；四组均从同一 base 初始化，64 episode 组不续训另一组 16 episode 的 checkpoint。

学习率 2e-5；attention q/k/v/o LoRA rank 8 / alpha 16；训练 seed 0；temperature 0.6、top-p 0.95、top-k 20、repetition penalty 1.1；总序列长度 10,240，实际 rollout cap 扣除 prompt；LGSD KL budget 0.004。训练使用 BF16 解量化权重。

## 统一评测

已提交评测作业 `25987248`：vanilla、OPSD-16、LGSD-16、OPSD-64、LGSD-64 的五组统一 student-only 评测。评测集为事先准备的 143 道 AMC/AIME 题，与 DeepMath 训练题无交集；每题一次配对采样，seed 20260809，生成上限 10,240，所有组启用 repetition penalty 1.1。

评测使用同一 vLLM MXFP4 推理配置，固定 medium reasoning 和日期，采用 full-response Math-Verify 0.9.0 离线评分。vanilla 会重新运行，避免把旧解码设置的成绩混进本轮比较。将保留相对 vanilla 救回／损失的题目和截断率；单 seed 对比不作为显著性证明。

## 证据

- [训练汇总 JSON](training_results.json)
- [160 个 episode 明细 CSV](training_episodes.csv)
- [评测协议与输入 / 代码 hashes](evaluation_protocol.json)
- [评测提交记录](evaluation_submission.json)
- 本地完整运行目录：`runs/deepmath-context-retrain-20260911/`
- 评测运行目录：`runs/deepmath-context-retrain-20260911/evaluation/`
