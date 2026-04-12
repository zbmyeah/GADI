# 实验说明

## 一、实验目的

本实验用于验证固定配置 `step120 + gradient_mix=0.5 + query_only` 的 GADI-R 在不同随机种子下是否稳定，并观察它相对当前 LoRA / LoRA-GA 参考基线是否仍有优势。

## 二、实验配置

- 实验方法：GADI-R
- 随机种子：11
- 模型：roberta-base
- 数据集：glue/mrpc
- LoRA Rank：8
- LoRA Alpha：16
- 训练轮数：1
- 训练批大小：16
- 重基化步数：120
- 选择策略：query_only
- topk_layers：2
- gradient_mix：0.5
- 校准 batch 数：4

## 三、实验结果

- 最终验证集 Loss：0.394588
- 最终验证集 Accuracy：0.838235
- 训练过程最好验证集 Loss：0.413964
- 训练过程最好验证集 Accuracy：0.808824
- 最好结果出现步数：200
- 相对 LoRA 参考基线 Accuracy 差值：+0.056373
- 相对 LoRA-GA 参考基线 Accuracy 差值：+0.026961
- 总耗时（秒）：22.05

## 四、重基化记录

- step 120: base_model.model.roberta.encoder.layer.0.attention.self.query, base_model.model.roberta.encoder.layer.1.attention.self.query

是否有 value 层进入 top-k：否

本次被刷新的层：base_model.model.roberta.encoder.layer.0.attention.self.query；base_model.model.roberta.encoder.layer.1.attention.self.query

## 五、结果文件说明

- 当前实验目录：`/mnt/workspace/GADI/results/2026-04-12/roberta-base_MRPC_GADI-R多随机种子验证_A10_query_only/gadi_seed_11_145136`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`
