#!/usr/bin/env python3
"""Full tiny-Qwen proof of DDP plus gradient-accumulation target normalization."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    scalar_comparison_metrics,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--accumulation-steps", type=int, required=True)
    return parser.parse_args()


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
    )


def make_case(global_index: int, supervised: int, device: torch.device) -> dict[str, torch.Tensor]:
    sequence_length = 12
    generator = torch.Generator(device="cpu").manual_seed(50_000 + global_index)
    input_ids = torch.randint(1, 255, (1, sequence_length), generator=generator).to(device)
    labels = torch.full_like(input_ids, -100)
    if supervised:
        labels[:, 1 : supervised + 1] = input_ids[:, 1 : supervised + 1]
        selected_positions = torch.arange(supervised, device=device, dtype=torch.long)
        shift_labels = labels[:, 1 : supervised + 1].reshape(-1).contiguous()
    else:
        # The real collator's graph-connected all-masked sentinel contract.
        selected_positions = torch.tensor([0], device=device, dtype=torch.long)
        shift_labels = torch.tensor([-100], device=device, dtype=torch.long)
    selective = {
        "input_ids": input_ids,
        "labels": labels,
        "logits_to_keep": selected_positions,
        "shift_labels": shift_labels,
        "use_cache": False,
    }
    return selective


def concatenate_named_tensors(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1) for name in sorted(values)])


def optimizer_from_manifest(parameters, training: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        betas=(training["adam_beta1"], training["adam_beta2"]),
        eps=training["adam_epsilon"],
        weight_decay=training["weight_decay"],
    )


def optimizer_moment_vector(optimizer: torch.optim.AdamW, name: str) -> torch.Tensor:
    values = [state[name].detach().reshape(-1) for state in optimizer.state.values() if name in state]
    if not values:
        raise RuntimeError(f"optimizer has no initialized {name} moments")
    return torch.cat(values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-model DDP/GA qualification requires CUDA")
    if args.accumulation_steps <= 0:
        raise ValueError("accumulation steps must be positive")
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        raise RuntimeError("launch with torchrun and at least two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    numerical = qualification["numerical_acceptance"]
    training = qualification["training_unit"]

    torch.manual_seed(20260718)
    reference = Qwen3_5ForCausalLM(tiny_config()).to(device=device, dtype=torch.float32).train()
    batched_reference = Qwen3_5ForCausalLM(tiny_config()).to(device=device, dtype=torch.float32).train()
    selective = Qwen3_5ForCausalLM(tiny_config()).to(device=device, dtype=torch.float32).train()
    batched_reference.load_state_dict(reference.state_dict(), strict=True)
    selective.load_state_dict(reference.state_dict(), strict=True)
    apply_liger_kernel_to_qwen3_5(
        rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=False, swiglu=False, model=selective
    )
    if "liger_kernel" not in selective.forward.__module__:
        raise RuntimeError("Liger did not patch the Qwen3.5 selective forward")
    ddp = torch.nn.parallel.DistributedDataParallel(selective, device_ids=[local_rank], find_unused_parameters=False)

    global_case_count = world_size * args.accumulation_steps
    counts = [
        0 if global_index % world_size == 0 else 1 + global_index % 7 for global_index in range(global_case_count)
    ]
    global_divisor = sum(counts)
    per_rank_counts = [sum(counts[rank::world_size]) for rank in range(world_size)]
    if global_divisor <= 0 or per_rank_counts[0] != 0 or any(count <= 0 for count in per_rank_counts[1:]):
        raise RuntimeError("adversarial target-count construction failed")
    cases = [make_case(index, counts[index], device) for index in range(global_case_count)]

    reference_initial = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    batched_initial = {name: parameter.detach().clone() for name, parameter in batched_reference.named_parameters()}
    selective_initial = {name: parameter.detach().clone() for name, parameter in selective.named_parameters()}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        batched_output = batched_reference(
            input_ids=torch.cat([inputs["input_ids"] for inputs in cases]),
            labels=torch.cat([inputs["labels"] for inputs in cases]),
            num_items_in_batch=global_divisor,
            use_cache=False,
        )
    batched_output.loss.backward()

    reference_loss_total = torch.zeros((), device=device, dtype=torch.float32)
    for inputs in cases:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = reference(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                num_items_in_batch=global_divisor,
                use_cache=False,
            )
        reference_loss_total = reference_loss_total + output.loss.detach()
        output.loss.backward()

    local_loss_total = torch.zeros((), device=device, dtype=torch.float32)
    for accumulation_slot in range(args.accumulation_steps):
        global_index = accumulation_slot * world_size + rank
        inputs = cases[global_index]
        sync_context = contextlib.nullcontext() if accumulation_slot == args.accumulation_steps - 1 else ddp.no_sync()
        with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**inputs, num_items_in_batch=global_divisor)
            # Mirrors Trainer average_tokens_across_devices: the model returns
            # local_sum/global_divisor and DDP later averages gradients by W.
            loss = output.loss * world_size
        local_loss_total = local_loss_total + output.loss.detach()
        loss.backward()

    global_observed_loss = local_loss_total.detach().clone()
    torch.distributed.all_reduce(global_observed_loss, op=torch.distributed.ReduceOp.SUM)
    loss_metrics = scalar_comparison_metrics(float(global_observed_loss), float(reference_loss_total))
    validate_comparison_metrics(loss_metrics, numerical, kind="loss", context="full-model DDP/GA loss")
    batched_loss_metrics = scalar_comparison_metrics(float(batched_output.loss), float(reference_loss_total))
    validate_comparison_metrics(
        batched_loss_metrics, numerical, kind="loss", context="batched versus accumulated single-process loss"
    )

    reference_gradients = {}
    batched_gradients = {}
    selective_gradients = {}
    parameter_gradient_metrics = {}
    batched_parameter_gradient_metrics = {}
    for (reference_name, reference_parameter), (batched_name, batched_parameter), (
        selective_name,
        selective_parameter,
    ) in zip(
        reference.named_parameters(), batched_reference.named_parameters(), selective.named_parameters(), strict=True
    ):
        if (
            reference_name != batched_name
            or reference_name != selective_name
            or reference_parameter.grad is None
            or batched_parameter.grad is None
            or selective_parameter.grad is None
        ):
            raise RuntimeError(f"gradient structure drift at {reference_name!r}/{batched_name!r}/{selective_name!r}")
        reference_gradients[reference_name] = reference_parameter.grad.detach().clone()
        batched_gradients[batched_name] = batched_parameter.grad.detach().clone()
        selective_gradients[selective_name] = selective_parameter.grad.detach().clone()
        metrics = tensor_comparison_metrics(selective_parameter.grad, reference_parameter.grad)
        validate_comparison_metrics(metrics, numerical, kind="gradient", context=f"DDP/GA {reference_name}")
        parameter_gradient_metrics[reference_name] = metrics
        batched_metrics = tensor_comparison_metrics(batched_parameter.grad, reference_parameter.grad)
        validate_comparison_metrics(
            batched_metrics, numerical, kind="gradient", context=f"batched/single-process {reference_name}"
        )
        batched_parameter_gradient_metrics[reference_name] = batched_metrics
    aggregate_gradient_metrics = tensor_comparison_metrics(
        concatenate_named_tensors(selective_gradients), concatenate_named_tensors(reference_gradients)
    )
    validate_comparison_metrics(
        aggregate_gradient_metrics, numerical, kind="gradient", context="full-model DDP/GA aggregate gradient"
    )
    batched_aggregate_gradient_metrics = tensor_comparison_metrics(
        concatenate_named_tensors(batched_gradients), concatenate_named_tensors(reference_gradients)
    )
    validate_comparison_metrics(
        batched_aggregate_gradient_metrics,
        numerical,
        kind="gradient",
        context="batched versus accumulated single-process aggregate gradient",
    )

    reference_preclip_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), training["max_gradient_norm"])
    batched_preclip_norm = torch.nn.utils.clip_grad_norm_(
        batched_reference.parameters(), training["max_gradient_norm"]
    )
    selective_preclip_norm = torch.nn.utils.clip_grad_norm_(selective.parameters(), training["max_gradient_norm"])
    clip_norm_metrics = scalar_comparison_metrics(float(selective_preclip_norm), float(reference_preclip_norm))
    validate_comparison_metrics(clip_norm_metrics, numerical, kind="loss", context="DDP/GA pre-clip norm")
    batched_clip_norm_metrics = scalar_comparison_metrics(float(batched_preclip_norm), float(reference_preclip_norm))
    validate_comparison_metrics(
        batched_clip_norm_metrics, numerical, kind="loss", context="batched/single-process pre-clip norm"
    )

    reference_optimizer = optimizer_from_manifest(reference.parameters(), training)
    batched_optimizer = optimizer_from_manifest(batched_reference.parameters(), training)
    selective_optimizer = optimizer_from_manifest(selective.parameters(), training)
    reference_optimizer.step()
    batched_optimizer.step()
    selective_optimizer.step()
    reference_updates = {
        name: parameter.detach() - reference_initial[name] for name, parameter in reference.named_parameters()
    }
    selective_updates = {
        name: parameter.detach() - selective_initial[name] for name, parameter in selective.named_parameters()
    }
    batched_updates = {
        name: parameter.detach() - batched_initial[name] for name, parameter in batched_reference.named_parameters()
    }
    aggregate_update_metrics = tensor_comparison_metrics(
        concatenate_named_tensors(selective_updates), concatenate_named_tensors(reference_updates)
    )
    validate_comparison_metrics(
        aggregate_update_metrics, numerical, kind="update", context="full-model DDP/GA aggregate update"
    )
    batched_aggregate_update_metrics = tensor_comparison_metrics(
        concatenate_named_tensors(batched_updates), concatenate_named_tensors(reference_updates)
    )
    validate_comparison_metrics(
        batched_aggregate_update_metrics,
        numerical,
        kind="update",
        context="batched versus accumulated single-process aggregate update",
    )
    optimizer_moment_metrics = {}
    batched_optimizer_moment_metrics = {}
    for moment_name in ("exp_avg", "exp_avg_sq"):
        observed_metrics = tensor_comparison_metrics(
            optimizer_moment_vector(selective_optimizer, moment_name),
            optimizer_moment_vector(reference_optimizer, moment_name),
        )
        validate_comparison_metrics(observed_metrics, numerical, kind="update", context=f"DDP/GA AdamW {moment_name}")
        optimizer_moment_metrics[moment_name] = observed_metrics
        batched_metrics = tensor_comparison_metrics(
            optimizer_moment_vector(batched_optimizer, moment_name),
            optimizer_moment_vector(reference_optimizer, moment_name),
        )
        validate_comparison_metrics(
            batched_metrics,
            numerical,
            kind="update",
            context=f"batched versus accumulated single-process AdamW {moment_name}",
        )
        batched_optimizer_moment_metrics[moment_name] = batched_metrics
    optimizer_step_values = {
        label: sorted({float(state["step"]) for state in optimizer.state.values() if "step" in state})
        for label, optimizer in (
            ("accumulated_single_process", reference_optimizer),
            ("batched_single_process", batched_optimizer),
            ("ddp_gradient_accumulation", selective_optimizer),
        )
    }
    if any(values != [1.0] for values in optimizer_step_values.values()):
        raise AssertionError(f"AdamW step-counter drift: {optimizer_step_values}")

    optimizer_dtypes = {
        str(value.dtype)
        for optimizer in (reference_optimizer, batched_optimizer, selective_optimizer)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    if optimizer_dtypes != {"torch.float32"}:
        raise AssertionError(f"optimizer state dtype drift: {optimizer_dtypes}")

    if rank == 0:
        report = {
            "artifact": "qwen35_full_model_ddp_gradient_accumulation_qualification",
            "schema_version": 1,
            "status": "passed",
            "qualification_protocol_id": qualification["protocol_id"],
            "qualification_manifest_sha256": qualification_sha256,
            "world_size": world_size,
            "gradient_accumulation_steps": args.accumulation_steps,
            "global_case_count": global_case_count,
            "per_case_assistant_targets": counts,
            "per_rank_assistant_targets": per_rank_counts,
            "includes_zero_target_case": counts[0] == 0,
            "includes_zero_target_rank": per_rank_counts[0] == 0,
            "global_assistant_target_divisor": global_divisor,
            "loss_comparison": loss_metrics,
            "batched_vs_accumulated_loss_comparison": batched_loss_metrics,
            "preclip_gradient_norm_comparison": clip_norm_metrics,
            "batched_vs_accumulated_preclip_gradient_norm_comparison": batched_clip_norm_metrics,
            "aggregate_gradient_comparison": aggregate_gradient_metrics,
            "batched_vs_accumulated_aggregate_gradient_comparison": batched_aggregate_gradient_metrics,
            "parameter_gradient_comparisons": parameter_gradient_metrics,
            "batched_vs_accumulated_parameter_gradient_comparisons": batched_parameter_gradient_metrics,
            "aggregate_adamw_update_comparison": aggregate_update_metrics,
            "batched_vs_accumulated_adamw_update_comparison": batched_aggregate_update_metrics,
            "adamw_moment_comparisons": optimizer_moment_metrics,
            "batched_vs_accumulated_adamw_moment_comparisons": batched_optimizer_moment_metrics,
            "adamw_step_counters": optimizer_step_values,
            "optimizer_floating_dtypes": sorted(optimizer_dtypes),
            "cuda_device": torch.cuda.get_device_name(),
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report_output.with_name(f".{args.report_output.name}.incomplete.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report_output)
        print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
