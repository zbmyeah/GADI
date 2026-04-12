from __future__ import annotations

from gadir.config import RebaseConfig


class IntervalRebaseScheduler:
    def __init__(self, config: RebaseConfig) -> None:
        self.config = config

    def should_rebase(self, global_step: int, rebase_count: int) -> bool:
        if not self.config.enabled:
            return False
        if global_step <= 0:
            return False
        if global_step < self.config.warmup_steps:
            return False
        if self.config.max_rebases is not None and rebase_count >= self.config.max_rebases:
            return False
        return global_step % self.config.interval_steps == 0
