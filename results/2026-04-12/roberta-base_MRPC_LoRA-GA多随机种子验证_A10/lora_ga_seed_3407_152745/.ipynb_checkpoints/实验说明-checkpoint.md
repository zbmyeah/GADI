# 实验说明

## 一、实验目的

本实验用于在阿里云 A10 环境下，对 `LoRA-GA` 进行多随机种子复现实验，为后续与 GADI-R 的 `mean ± std` 公平比较提供基线结果。

## 二、实验配置

- 方法：`LoRA-GA`
- 随机种子：`3407`
- 模型：`roberta-base`
- 数据集：`glue/mrpc`
- LoRA Rank：`8`
- LoRA Alpha：`16`
- 训练轮数：`1`
- 训练 batch size：`16`
- 评估 batch size：`32`
- AMP：`True`
- AMP dtype：`bfloat16`
- TF32：`True`

## 三、实验结果

- 最终验证集 Loss：`0.477734`
- 最终验证集 Accuracy：`0.745098`
- 训练过程中最佳验证集 Loss：`0.474544`
- 训练过程中最佳验证集 Accuracy：`0.750000`
- 最佳结果出现步数：`200`
- 总耗时（秒）：`22.39`

## 四、结果文件说明

- 当前实验目录：`results/2026-04-12/roberta-base_MRPC_LoRA-GA多随机种子验证_A10/lora_ga_seed_3407_152745`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`
