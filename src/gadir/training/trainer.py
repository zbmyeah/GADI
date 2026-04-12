from __future__ import annotations

from contextlib import nullcontext
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from gadir.config import ExperimentConfig
from gadir.data.loaders import build_data_bundle
from gadir.evaluation.metrics import evaluate_sequence_classification
from gadir.methods import build_method
from gadir.models.factory import load_model_and_tokenizer
from gadir.utils.logging import get_logger
from gadir.utils.seed import seed_everything


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _take_calibration_batches(
    calibration_batches_cpu: list[dict[str, torch.Tensor]],
    state: dict[str, int],
    batch_count: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(batch_count):
        index = state["index"] % len(calibration_batches_cpu)
        cpu_batch = calibration_batches_cpu[index]
        batches.append(_move_batch_to_device(cpu_batch, device))
        state["index"] += 1
    return batches


def _resolve_total_steps(config: ExperimentConfig, train_loader_length: int) -> int:
    if config.training.max_steps is not None:
        return config.training.max_steps
    steps_per_epoch = math.ceil(train_loader_length / config.training.gradient_accumulation_steps)
    return max(1, steps_per_epoch * config.training.num_epochs)


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported AMP dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported AMP dtype: {dtype_name}")
    return dtype


def _shutdown_dataloader_workers(dataloader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def run_training(config: ExperimentConfig) -> dict[str, float]:
    logger = get_logger("trainer")
    seed_everything(config.seed)
    run_start = time.perf_counter()

    logger.info(
        "Preparing run | method=%s | seed=%s | model=%s | dataset=%s/%s",
        config.method,
        config.seed,
        config.model.model_name_or_path,
        config.data.dataset_name,
        config.data.dataset_config_name,
    )
    logger.info("Loading model and tokenizer...")
    model, tokenizer, target_modules = load_model_and_tokenizer(config.model)
    config.model.target_modules = target_modules
    logger.info("Loaded model. Building datasets and dataloaders...")
    data_bundle = build_data_bundle(config, tokenizer)
    logger.info(
        "Data ready | train_batches=%s | eval_batches=%s | calibration_batches=%s",
        len(data_bundle.train_loader),
        len(data_bundle.eval_loader),
        len(data_bundle.calibration_loader),
    )
    method = build_method(config)
    model = method.wrap_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    logger.info("Model wrapped with method=%s and moved to device=%s.", config.method, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
    amp_enabled = device.type == "cuda" and config.training.use_amp
    amp_dtype = _resolve_amp_dtype(config.training.amp_dtype) if amp_enabled else None
    grad_scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=grad_scaler_enabled) if device.type == "cuda" else None
    logger.info(
        "Precision setup | amp_enabled=%s | amp_dtype=%s | allow_tf32=%s",
        amp_enabled,
        amp_dtype,
        config.training.allow_tf32 if device.type == "cuda" else False,
    )

    calibration_batches_cpu = list(data_bundle.calibration_loader)
    if not calibration_batches_cpu:
        raise ValueError("Calibration dataloader produced no batches.")
    calibration_state = {"index": 0}
    logger.info("Cached %s calibration batches in host memory.", len(calibration_batches_cpu))
    calibration_batch_provider = lambda batch_count: _take_calibration_batches(
        calibration_batches_cpu,
        calibration_state,
        batch_count,
        device,
    )

    init_start = time.perf_counter()
    logger.info("Initializing method-specific state...")
    method.initialize(
        model=model,
        calibration_batch_provider=calibration_batch_provider,
        device=device,
    )
    initialization_time = time.perf_counter() - init_start
    method.set_initialization_time(initialization_time)
    logger.info("Initialization finished in %.2f seconds.", initialization_time)

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    total_steps = _resolve_total_steps(config, len(data_bundle.train_loader))
    logger.info("Starting training for %s optimizer steps on device=%s.", total_steps, device)

    global_step = 0
    running_loss = 0.0
    train_history: list[dict[str, float | int]] = []
    eval_history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)
    training_start = time.perf_counter()
    last_heartbeat = training_start
    heartbeat_seconds = 30.0

    try:
        for _ in range(config.training.num_epochs):
            model.train()
            for batch_index, batch in enumerate(data_bundle.train_loader, start=1):
                batch = _move_batch_to_device(batch, device)
                autocast_context = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_enabled and amp_dtype is not None
                    else nullcontext()
                )
                with autocast_context:
                    outputs = model(**batch)
                    loss = outputs.loss / config.training.gradient_accumulation_steps
                if grad_scaler_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                current_loss = float(outputs.loss.detach())
                running_loss += current_loss

                if batch_index % config.training.gradient_accumulation_steps != 0:
                    continue

                if grad_scaler_enabled and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                method.after_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    calibration_batch_provider=calibration_batch_provider,
                    device=device,
                )

                elapsed = time.perf_counter() - training_start
                if global_step == 1 or time.perf_counter() - last_heartbeat >= heartbeat_seconds:
                    logger.info(
                        "heartbeat step=%s/%s | latest_loss=%.4f | elapsed=%.1fs",
                        global_step,
                        total_steps,
                        current_loss,
                        elapsed,
                    )
                    last_heartbeat = time.perf_counter()

                if global_step % config.training.log_every_steps == 0:
                    average_train_loss = running_loss / config.training.log_every_steps
                    logger.info(
                        "step=%s/%s | train_loss=%.4f",
                        global_step,
                        total_steps,
                        average_train_loss,
                    )
                    train_history.append(
                        {
                            "step": global_step,
                            "train_loss": average_train_loss,
                        }
                    )
                    running_loss = 0.0

                if global_step % config.training.eval_every_steps == 0:
                    logger.info("Starting evaluation at step=%s...", global_step)
                    metrics = evaluate_sequence_classification(model, data_bundle.eval_loader, device)
                    logger.info("Completed evaluation at step=%s | eval=%s", global_step, metrics)
                    eval_history.append(
                        {
                            "step": global_step,
                            **metrics,
                        }
                    )

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        logger.info("Starting final evaluation...")
        final_metrics = evaluate_sequence_classification(model, data_bundle.eval_loader, device)
        logger.info("final_eval=%s", final_metrics)

        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model artifacts to %s...", output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model artifacts to %s", output_dir)
        method_artifacts = method.get_artifacts()
        peak_memory_allocated_mb = 0.0
        peak_memory_reserved_mb = 0.0
        if device.type == "cuda":
            peak_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        runtime = {
            "total_wall_time_seconds": time.perf_counter() - run_start,
            "method_initialization_time_seconds": initialization_time,
            "rebase_overhead_time_seconds": float(
                method_artifacts.get("profiling", {}).get("rebase_time_seconds", 0.0)
            ),
            "peak_memory_allocated_mb": peak_memory_allocated_mb,
            "peak_memory_reserved_mb": peak_memory_reserved_mb,
            "device": device.type,
        }
        return {
            **final_metrics,
            "train_history": train_history,
            "eval_history": eval_history,
            "total_steps": total_steps,
            "method_artifacts": method_artifacts,
            "runtime": runtime,
        }
    finally:
        _shutdown_dataloader_workers(data_bundle.train_loader)
        _shutdown_dataloader_workers(data_bundle.eval_loader)
        _shutdown_dataloader_workers(data_bundle.calibration_loader)
