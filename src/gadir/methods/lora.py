from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

from gadir.methods.base import BaseAdaptationMethod

TASK_TYPE_MAP = {
    "SEQ_CLS": TaskType.SEQ_CLS,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "CAUSAL_LM": TaskType.CAUSAL_LM,
}


class VanillaLoraMethod(BaseAdaptationMethod):
    def wrap_model(self, model):
        peft_config = LoraConfig(
            task_type=TASK_TYPE_MAP[self.config.model.task_type.upper()],
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=self.config.model.target_modules,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            use_rslora=self.config.lora.use_rslora,
            init_lora_weights=True,
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        return peft_model
