# GADI-R 多随机种子验证总结

## 一、实验设置

- 固定配置：`step120 + gradient_mix=0.5 + query_only + topk_layers=1`。
- 模型：`roberta-base`
- 数据集：`GLUE/MRPC`
- 随机种子列表：`11, 42, 123, 3407, 2026`
- 已存在的相同配置结果会直接复用，不重复训练。

## 二、参考基线

- LoRA 参考基线最终 Accuracy：`0.781863`，路径：`/mnt/workspace/GADI/results/2026-04-04/roberta-base_MRPC_完整对比实验/lora_131228`
- LoRA-GA 参考基线最终 Accuracy：`0.811275`，路径：`/mnt/workspace/GADI/results/2026-04-04/roberta-base_MRPC_完整对比实验/lora_ga_132857`
- 说明：当前参考基线来自已完成的 `seed=42` 实验，因此这里的结论主要用于判断 GADI 配置本身的稳定性和超越潜力。

## 三、逐种子结果

| seed | 是否复用 | 最终 Accuracy | 最终 Loss | 最佳 Accuracy | 最佳步数 | 是否超过 LoRA | 是否超过 LoRA-GA | value 是否入选 | 刷新层 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 11 | 否 | 0.835784 | 0.389550 | 0.813725 | 200 | 是 | 是 | 否 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 42 | 否 | 0.830882 | 0.365337 | 0.791667 | 200 | 是 | 是 | 否 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 123 | 否 | 0.833333 | 0.409344 | 0.713235 | 200 | 是 | 是 | 否 | base_model.model.roberta.encoder.layer.1.attention.self.query |
| 2026 | 否 | 0.772059 | 0.482209 | 0.764706 | 160 | 否 | 否 | 否 | base_model.model.roberta.encoder.layer.0.attention.self.query |
| 3407 | 否 | 0.767157 | 0.455777 | 0.750000 | 200 | 否 | 否 | 否 | base_model.model.roberta.encoder.layer.0.attention.self.query |

## 四、统计结果

- 最终 Accuracy 均值：`0.807843`
- 最终 Accuracy 标准差：`0.034990`
- 最终 Accuracy 最小值：`0.767157`
- 最终 Accuracy 最大值：`0.835784`
- 最终 Loss 均值：`0.420444`
- 最终 Loss 标准差：`0.047901`
- 超过 LoRA 参考基线的种子数：`3/5`
- 超过 LoRA-GA 参考基线的种子数：`3/5`
- 表现最好的种子：`11`，最终 Accuracy `0.835784`
- 表现最差的种子：`3407`，最终 Accuracy `0.767157`

## 五、结论

- 当前固定配置已经在部分随机种子上超过 LoRA-GA，但多种子均值或多数种子优势还不够稳定，说明它有潜力但仍需继续打磨。
- 该配置对标准 LoRA 的优势仍需结合更多种子继续确认。

## 六、目录说明

- 当前实验根目录：`/mnt/workspace/GADI/results/2026-04-12/roberta-base_MRPC_GADI-R多随机种子验证_A10_query_only_top1_step120`
- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重；复用结果在总表中保留原路径。
