# RoBERTa-base + RTE 在 A10 上的公平多随机种子总对比报告

## 一、报告目的

本报告汇总 `LoRA`、`LoRA-GA` 与固定主配置 `GADI-R` 在 `RTE` 任务上的公平对比结果，用于直接纳入论文实验章节。

当前 GADI-R 主配置为：

- `step120 + gradient_mix=0.5 + query_only + topk_layers=1`

## 二、统一实验设置

- 模型：`roberta-base`
- 数据集：`GLUE/rte`
- 硬件环境：`Alibaba Cloud ecs.gn7i-c8g1.2xlarge (NVIDIA A10 x1)`
- 随机种子：`11, 42, 123, 3407, 2026`
- 训练轮数：`1`

对应原始结果目录如下：

- `LoRA`：`results/2026-04-12/roberta-base_RTE_LoRA多随机种子验证_A10`
- `LoRA-GA`：`results/2026-04-12/roberta-base_RTE_LoRA-GA多随机种子验证_A10`
- `GADI-R`：`results/2026-04-12/roberta-base_RTE_GADI-R多随机种子验证_A10`

## 三、总体结果对比

| 方法 | 最终 Accuracy mean ± std | 最终 Loss mean ± std | 总时间 mean ± std (s) | 初始化 mean ± std (s) | 重基化 mean ± std (s) | 峰值显存 mean ± std (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA | `0.504693 ± 0.030675` | `0.692552 ± 0.005813` | `29.99 ± 15.32` | `0.00 ± 0.00` | `0.00 ± 0.00` | `1384.96 ± 222.46` |
| LoRA-GA | `0.548014 ± 0.074135` | `0.682174 ± 0.013098` | `35.22 ± 9.86` | `1.63 ± 0.10` | `0.00 ± 0.00` | `1595.47 ± 267.21` |
| GADI-R | `0.549458 ± 0.073339` | `0.681934 ± 0.013843` | `27.95 ± 4.10` | `1.66 ± 0.13` | `0.49 ± 0.01` | `1613.38 ± 267.53` |

## 四、逐随机种子对比

| seed | LoRA | LoRA-GA | GADI-R | 当个 seed 最优方法 | GADI 刷新层 |
| ---: | ---: | ---: | ---: | --- | --- |
| 11 | `0.527076` | `0.552347` | `0.555957` | `GADI-R` | `base_model.model.roberta.encoder.layer.0.attention.self.query` |
| 42 | `0.469314` | `0.480144` | `0.480144` | `LoRA-GA` | `base_model.model.roberta.encoder.layer.0.attention.self.query` |
| 123 | `0.527076` | `0.584838` | `0.595668` | `GADI-R` | `base_model.model.roberta.encoder.layer.0.attention.self.query` |
| 2026 | `0.472924` | `0.472924` | `0.472924` | `LoRA` | `base_model.model.roberta.encoder.layer.0.attention.self.query` |
| 3407 | `0.527076` | `0.649819` | `0.642599` | `LoRA-GA` | `base_model.model.roberta.encoder.layer.0.attention.self.query` |

## 五、时间与显存开销解读

- `总时间` 反映一次完整 run 从进入训练流程到保存结果结束的整体开销。
- `额外初始化时间` 对应方法特有初始化步骤。LoRA 近似为 0，LoRA-GA 与 GADI-R 主要来自梯度收集与低秩分解。
- `重基化额外耗时` 主要反映 GADI-R 在训练中进行 drift 检测与重基化的额外开销，LoRA 与 LoRA-GA 为 0。
- `峰值显存` 用于判断 GADI-R 的额外开销是否处于可接受范围。

## 六、结论

- 当前任务上平均 Accuracy 最好的方法是 `GADI-R`，其均值为 `0.549458`。
- 若 GADI-R 的平均 Accuracy 同时高于 LoRA 与 LoRA-GA，且额外初始化时间与重基化耗时相对可控，则可说明其具备实际应用价值。

## 七、目录说明

- 当前报告目录：`results/2026-04-12/roberta-base_RTE_A10_公平多随机种子总对比报告`
- 本目录包含总对比报告、方法统计表、逐 seed 对比表、时间显存汇总表。
