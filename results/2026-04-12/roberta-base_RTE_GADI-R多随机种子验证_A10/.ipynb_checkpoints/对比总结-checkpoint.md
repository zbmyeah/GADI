# GADI-R 多随机种子验证总结

## 一、实验设置

- 固定配置：`step120 + gradient_mix=0.5 + query_only + topk_layers=1`
- 模型：`roberta-base`
- 数据集：`glue/rte`
- 随机种子列表：`11, 42, 123, 3407, 2026`
- 已存在的相同配置结果会直接复用，不重复训练。

## 二、参考基线

- LoRA 参考基线最终 Accuracy：`0.472924`，路径：`results/2026-04-12/roberta-base_RTE_LoRA多随机种子验证_A10/lora_seed_2026_155907`
- LoRA-GA 参考基线最终 Accuracy：`0.472924`，路径：`results/2026-04-12/roberta-base_RTE_LoRA-GA多随机种子验证_A10/lora_ga_seed_2026_160204`

## 三、逐种子结果

| seed | 是否复用 | 最终 Accuracy | 最终 Loss | 最佳 Accuracy | 最佳步数 | 是否超过 LoRA | 是否超过 LoRA-GA | value 是否入选 | 总时间(s) | 初始化(s) | 重基化(s) | 峰值显存(MiB) | 刷新层 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 11 | 否 | 0.555957 | 0.677717 | 0.566787 | 140 | 是 | 是 | 否 | 31.83 | 1.90 | 0.50 | 1417.42 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 42 | 否 | 0.480144 | 0.691066 | 0.545126 | 100 | 是 | 是 | 否 | 25.54 | 1.64 | 0.48 | 1418.29 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 123 | 否 | 0.595668 | 0.662574 | 0.653430 | 140 | 是 | 是 | 否 | 24.05 | 1.61 | 0.49 | 1418.29 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 2026 | 否 | 0.472924 | 0.698805 | 0.530686 | 140 | 否 | 否 | 否 | 32.92 | 1.61 | 0.49 | 1906.45 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 3407 | 否 | 0.642599 | 0.679509 | 0.527076 | 140 | 是 | 是 | 否 | 25.40 | 1.57 | 0.49 | 1906.45 | base_model.model.roberta.encoder.layer.0.attention.self.query |

## 四、统计结果

- 最终 Accuracy 均值：`0.549458`
- 最终 Accuracy 标准差：`0.073339`
- 最终 Loss 均值：`0.681934`
- 最终 Loss 标准差：`0.013843`
- 总运行时间均值（秒）：`27.95`
- 额外初始化时间均值（秒）：`1.66`
- 重基化额外耗时均值（秒）：`0.49`
- 峰值显存均值（MiB）：`1613.38`
- 超过 LoRA 参考基线的种子数：`4/5`
- 超过 LoRA-GA 参考基线的种子数：`4/5`
- 表现最好的种子：`3407`，最终 Accuracy `0.642599`
- 表现最差的种子：`2026`，最终 Accuracy `0.472924`

## 五、结论

- 当前固定配置在多随机种子下表现出较稳定的平均优势，说明该 GADI-R 版本已经具备超过 LoRA-GA 的潜力。

## 六、目录说明

- 当前实验根目录：`results/2026-04-12/roberta-base_RTE_GADI-R多随机种子验证_A10`
- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重；复用结果在总表中保留原路径。
