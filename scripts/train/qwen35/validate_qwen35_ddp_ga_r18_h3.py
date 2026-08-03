#!/usr/bin/env python3
"""Produce one R18 H3 scenario/chunk distributed-normalization artifact."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID, install_qwen35_checkpointed_chunked_loss
from open_instruct.qwen35_qualification import (
    scalar_comparison_metrics,
    sha256_file,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)
from open_instruct.qwen35_qualification_r18_h3 import (
    COMPARISON_FAMILIES,
    COMPARISON_PATHS,
    GRADIENT_FAMILIES,
    H3_ARTIFACT,
    H3_PROTOCOL_ID,
    PATHS,
    STORED_FAMILIES,
    expected_case_records,
    load_h3_harness_amendment,
    load_h3_manifest,
    prepare_distributed_output_directory,
    scenario_by_id,
    tensor_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-manifest", type=Path, required=True)
    parser.add_argument("--r18-manifest", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--harness-amendment", type=Path, required=True)
    parser.add_argument("--harness-amendment-human-protocol", type=Path, required=True)
    parser.add_argument("--harness-amendment-preregistration-closure", type=Path, required=True)
    parser.add_argument("--attempt01-failure-closure", type=Path, required=True)
    parser.add_argument("--scenario-id", choices=("P4x2", "B4x4"), required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=source_root(), check=True, text=True, capture_output=True).stdout.strip()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def named_state_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def tiny_config(config: dict[str, Any]) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        linear_conv_kernel_dim=config["linear_conv_kernel_dim"],
        linear_key_head_dim=config["linear_key_head_dim"],
        linear_value_head_dim=config["linear_value_head_dim"],
        linear_num_key_heads=config["linear_num_key_heads"],
        linear_num_value_heads=config["linear_num_value_heads"],
        layer_types=config["layer_types"],
        tie_word_embeddings=config["tie_word_embeddings"],
        attention_dropout=config["attention_dropout"],
        initializer_range=config["initializer_range"],
        use_cache=config["use_cache"],
    )


def make_model(config: dict[str, Any], *, seed: int, chunk_size: int, device: torch.device) -> Qwen3_5ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(tiny_config(config)).to(device=device, dtype=torch.float32).train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    install_qwen35_checkpointed_chunked_loss(model, chunk_size=chunk_size)
    if model.forward.__module__ != "open_instruct.qwen35_chunked_loss":
        raise RuntimeError("R18 selected-loss forward was not installed")
    if {parameter.dtype for parameter in model.parameters()} != {torch.float32}:
        raise RuntimeError("H3 model parameters are not exclusively FP32")
    return model


def make_case(record: dict[str, Any], model_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(record["seed"])
    input_ids = torch.randint(
        1, model_config["vocab_size"] - 1, (1, model_config["sequence_length"]), generator=generator, dtype=torch.long
    )
    labels = torch.full_like(input_ids, -100)
    count = record["target_count"]
    if count:
        labels[:, 1 : count + 1] = input_ids[:, 1 : count + 1]
        positions = torch.arange(count, dtype=torch.long)
        shifted = labels[:, 1 : count + 1].reshape(-1).contiguous()
    else:
        positions = torch.tensor([0], dtype=torch.long)
        shifted = torch.tensor([-100], dtype=torch.long)
    report_record = {
        **record,
        "input_ids_sha256": tensor_sha256(input_ids),
        "labels_sha256": tensor_sha256(labels),
        "logits_to_keep_sha256": tensor_sha256(positions),
        "shift_labels_sha256": tensor_sha256(shifted),
    }
    return {
        "record": report_record,
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "logits_to_keep": positions.to(device),
        "shift_labels": shifted.to(device),
    }


def forward_case(model: torch.nn.Module, case: dict[str, Any], global_target_count: int) -> tuple[torch.Tensor, dict]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise RuntimeError("H3 CUDA BF16 autocast was not active")
        output = model(
            input_ids=case["input_ids"],
            labels=case["labels"],
            logits_to_keep=case["logits_to_keep"],
            shift_labels=case["shift_labels"],
            num_items_in_batch=global_target_count,
            use_cache=False,
        )
    if output.logits is not None:
        raise RuntimeError("H3 selected-loss forward returned dense logits")
    if output.loss is None or output.loss.dtype != torch.float32 or not bool(torch.isfinite(output.loss)):
        raise RuntimeError("H3 selected-loss forward returned an invalid loss")
    audit_owner = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    audit = getattr(audit_owner, "_qwen35_last_loss_audit", None)
    if not isinstance(audit, dict):
        raise RuntimeError("H3 selected-loss forward did not emit its audit")
    return output.loss, json.loads(json.dumps(audit))


def clone_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def clone_named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"H3 disconnected parameter gradient: {name}")
        if parameter.grad.dtype != torch.float32 or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"H3 invalid FP32 finite gradient: {name}")
        values[name] = parameter.grad.detach().clone()
    return values


def make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
        weight_decay=config["weight_decay"],
        foreach=config["foreach"],
        fused=config["fused"],
    )
    if any(group.get("fused") is not True or group.get("foreach") is not False for group in optimizer.param_groups):
        raise RuntimeError("H3 optimizer did not retain fused=True and foreach=False")
    return optimizer


def optimizer_state(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int], set[str]]:
    exp_avg = {}
    exp_avg_sq = {}
    steps = set()
    dtypes = set()
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"H3 optimizer state structure drift for {name}")
        steps.add(int(state["step"].item()))
        for family, destination in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            value = state[family]
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"H3 optimizer {family} is not finite FP32 for {name}")
            dtypes.add(str(value.dtype))
            destination[name] = value.detach().clone()
    return exp_avg, exp_avg_sq, sorted(steps), dtypes


def aggregate(values: dict[str, torch.Tensor], parameter_names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1) for name in parameter_names])


def build_path_comparisons(
    state: dict[str, dict[str, dict[str, torch.Tensor]]], *, parameter_names: list[str], numerical: dict[str, Any]
) -> dict[str, Any]:
    comparisons = {}
    for observed_path in COMPARISON_PATHS:
        comparisons[observed_path] = {}
        for family in COMPARISON_FAMILIES:
            if family == "raw_adamw_update":
                reference = {
                    name: state[PATHS[0]]["post_step_parameter"][name] - state[PATHS[0]]["initial_parameter"][name]
                    for name in parameter_names
                }
                observed = {
                    name: state[observed_path]["post_step_parameter"][name]
                    - state[observed_path]["initial_parameter"][name]
                    for name in parameter_names
                }
            else:
                reference = state[PATHS[0]][family]
                observed = state[observed_path][family]
            kind = "gradient" if family in GRADIENT_FAMILIES else "update"
            named = {}
            for name in parameter_names:
                metrics = tensor_comparison_metrics(observed[name], reference[name])
                validate_comparison_metrics(
                    metrics, numerical, kind=kind, context=f"H3 {observed_path} {family} {name}"
                )
                named[name] = metrics
            aggregate_metrics = tensor_comparison_metrics(
                aggregate(observed, parameter_names), aggregate(reference, parameter_names)
            )
            validate_comparison_metrics(
                aggregate_metrics, numerical, kind=kind, context=f"H3 {observed_path} aggregate {family}"
            )
            comparisons[observed_path][family] = {"aggregate": aggregate_metrics, "named": named}
    return comparisons


def execute_path(
    model: Qwen3_5ForCausalLM,
    cases: list[dict[str, Any]],
    *,
    global_target_count: int,
    mode: str,
    optimizer_config: dict[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if mode not in {"graph_sum", "sequential_backward"}:
        raise ValueError(f"unsupported central H3 path {mode}")
    initial = clone_named_parameters(model)
    losses = []
    audits = []
    graph_loss = None
    for case in cases:
        loss, audit = forward_case(model, case, global_target_count)
        losses.append(float(loss.detach().item()))
        audits.append({"case_id": case["record"]["case_id"], "audit": audit})
        if mode == "graph_sum":
            graph_loss = loss if graph_loss is None else graph_loss + loss
        else:
            loss.backward()
    if mode == "graph_sum":
        if graph_loss is None:
            raise RuntimeError("H3 graph-sum path saw no cases")
        graph_loss.backward()
        global_loss = float(graph_loss.detach().item())
    else:
        global_loss = float(torch.tensor(losses, dtype=torch.float32, device=next(model.parameters()).device).sum())
    preclip = clone_named_gradients(model)
    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["maximum_gradient_norm"])
    clipped = clone_named_gradients(model)
    optimizer = make_optimizer(model, optimizer_config)
    optimizer.step()
    post = clone_named_parameters(model)
    exp_avg, exp_avg_sq, steps, dtypes = optimizer_state(optimizer, model)
    return (
        {
            "initial_parameter": initial,
            "preclip_gradient": preclip,
            "clipped_gradient": clipped,
            "post_step_parameter": post,
            "optimizer_exp_avg": exp_avg,
            "optimizer_exp_avg_sq": exp_avg_sq,
        },
        {
            "audits": audits,
            "clip_grad_norm_return": float(clip_return.item()),
            "global_loss": global_loss,
            "optimizer_state_dtypes": sorted(dtypes),
            "optimizer_steps": steps,
            "per_case_losses": losses,
        },
    )


def execute_ddp_path(
    ddp: torch.nn.parallel.DistributedDataParallel,
    all_cases: list[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    accumulation_steps: int,
    global_target_count: int,
    optimizer_config: dict[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    model = ddp.module
    initial = clone_named_parameters(model)
    local_rows = []
    local_audits = []
    local_loss_sum = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for slot in range(accumulation_steps):
        case_id = slot * world_size + rank
        case = all_cases[case_id]
        synchronized = slot == accumulation_steps - 1
        sync_context = contextlib.nullcontext() if synchronized else ddp.no_sync()
        with sync_context:
            unscaled_loss, audit = forward_case(ddp, case, global_target_count)
            scaled_loss = unscaled_loss * world_size
            scaled_loss.backward()
        local_loss_sum = local_loss_sum + unscaled_loss.detach()
        local_rows.append(
            {
                "case_id": case_id,
                "global_target_count": global_target_count,
                "rank": rank,
                "scaled_backward_loss": float(scaled_loss.detach().item()),
                "slot": slot,
                "synchronized_backward": synchronized,
                "target_count": case["record"]["target_count"],
                "unscaled_model_loss": float(unscaled_loss.detach().item()),
            }
        )
        local_audits.append({"case_id": case_id, "audit": audit})
    global_loss = local_loss_sum.clone()
    torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.SUM)
    preclip = clone_named_gradients(model)
    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["maximum_gradient_norm"])
    clipped = clone_named_gradients(model)
    optimizer = make_optimizer(model, optimizer_config)
    optimizer.step()
    post = clone_named_parameters(model)
    exp_avg, exp_avg_sq, steps, dtypes = optimizer_state(optimizer, model)
    post_state_hash = named_state_sha256(
        {
            **{f"post::{name}": value for name, value in post.items()},
            **{f"exp_avg::{name}": value for name, value in exp_avg.items()},
            **{f"exp_avg_sq::{name}": value for name, value in exp_avg_sq.items()},
        }
    )
    gathered_rows: list[list[dict] | None] = [None] * world_size
    gathered_audits: list[list[dict] | None] = [None] * world_size
    gathered_hashes: list[str | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered_rows, local_rows)
    torch.distributed.all_gather_object(gathered_audits, local_audits)
    torch.distributed.all_gather_object(gathered_hashes, post_state_hash)
    rows = sorted((item for values in gathered_rows for item in (values or [])), key=lambda item: item["case_id"])
    audits = sorted((item for values in gathered_audits for item in (values or [])), key=lambda item: item["case_id"])
    return (
        {
            "initial_parameter": initial,
            "preclip_gradient": preclip,
            "clipped_gradient": clipped,
            "post_step_parameter": post,
            "optimizer_exp_avg": exp_avg,
            "optimizer_exp_avg_sq": exp_avg_sq,
        },
        {
            "audits": audits,
            "case_losses": rows,
            "clip_grad_norm_return": float(clip_return.item()),
            "global_loss": float(global_loss.item()),
            "optimizer_state_dtypes": sorted(dtypes),
            "optimizer_steps": steps,
            "post_state_hashes": gathered_hashes,
            "per_case_losses": [item["unscaled_model_loss"] for item in rows],
        },
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R18 H3 requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", device_id=device)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != 4:
            raise RuntimeError(f"R18 H3 requires exactly four ranks, found {world_size}")
        if "A100" not in torch.cuda.get_device_name(device):
            raise RuntimeError(f"R18 H3 requires Leonardo A100 hardware, found {torch.cuda.get_device_name(device)!r}")
        h3, h3_sha256, r18 = load_h3_manifest(
            args.h3_manifest, r18_manifest_path=args.r18_manifest, human_protocol_path=args.human_protocol
        )
        harness_amendment, harness_amendment_sha256 = load_h3_harness_amendment(
            args.harness_amendment,
            human_amendment_path=args.harness_amendment_human_protocol,
            attempt01_failure_closure_path=args.attempt01_failure_closure,
            preregistration_closure_path=args.harness_amendment_preregistration_closure,
            h3_manifest_path=args.h3_manifest,
        )
        if args.chunk_size not in h3["candidate_chunk_sizes_in_execution_order"]:
            raise ValueError("requested chunk size is outside the H3 candidate set")
        scenario = scenario_by_id(h3, args.scenario_id)
        output_directory_initialization = prepare_distributed_output_directory(
            args.output_dir, rank=rank, broadcast_object_list=torch.distributed.broadcast_object_list
        )

        imported_liger = sorted(
            name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
        )
        gathered_liger: list[list[str] | None] = [None] * world_size
        gathered_devices: list[str | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_liger, imported_liger)
        torch.distributed.all_gather_object(gathered_devices, torch.cuda.get_device_name(device))
        if any(values for values in gathered_liger):
            raise RuntimeError(f"R18 H3 imported Liger: {gathered_liger}")

        records = expected_case_records(h3, args.scenario_id)
        cases = [make_case(record, h3["model"], device) for record in records]
        ddp_model = make_model(h3["model"], seed=scenario["model_seed"], chunk_size=args.chunk_size, device=device)
        central_model = None
        sequential_model = None
        if rank == 0:
            central_model = make_model(
                h3["model"], seed=scenario["model_seed"], chunk_size=args.chunk_size, device=device
            )
            sequential_model = make_model(
                h3["model"], seed=scenario["model_seed"], chunk_size=args.chunk_size, device=device
            )
            initial_sets = [clone_named_parameters(model) for model in (ddp_model, central_model, sequential_model)]
            if [list(values) for values in initial_sets].count(list(initial_sets[0])) != 3:
                raise RuntimeError("H3 initial parameter-name order drift")
            for name in initial_sets[0]:
                if not all(torch.equal(initial_sets[0][name], values[name]) for values in initial_sets[1:]):
                    raise RuntimeError(f"H3 initial parameter mismatch for {name}")
        ddp = torch.nn.parallel.DistributedDataParallel(
            ddp_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

        state: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        details: dict[str, dict[str, Any]] = {}
        if rank == 0:
            assert central_model is not None and sequential_model is not None
            state[PATHS[0]], details[PATHS[0]] = execute_path(
                central_model,
                cases,
                global_target_count=scenario["global_target_count"],
                mode="graph_sum",
                optimizer_config=h3["optimizer"],
            )
            state[PATHS[1]], details[PATHS[1]] = execute_path(
                sequential_model,
                cases,
                global_target_count=scenario["global_target_count"],
                mode="sequential_backward",
                optimizer_config=h3["optimizer"],
            )
        state[PATHS[2]], details[PATHS[2]] = execute_ddp_path(
            ddp,
            cases,
            rank=rank,
            world_size=world_size,
            accumulation_steps=scenario["accumulation_steps"],
            global_target_count=scenario["global_target_count"],
            optimizer_config=h3["optimizer"],
        )

        outcome: list[dict[str, Any] | None] = [None]
        if rank == 0:
            parameter_names = list(state[PATHS[0]]["initial_parameter"])
            if any(list(state[path]["initial_parameter"]) != parameter_names for path in PATHS):
                raise RuntimeError("H3 path parameter-name order drift")
            for path in COMPARISON_PATHS:
                for name in parameter_names:
                    if not torch.equal(
                        state[path]["initial_parameter"][name], state[PATHS[0]]["initial_parameter"][name]
                    ):
                        raise RuntimeError(f"H3 initial state mismatch for {path}/{name}")
            numerical = h3["numerical_acceptance"]
            comparisons = build_path_comparisons(state, parameter_names=parameter_names, numerical=numerical)
            loss_comparisons = {}
            for path in COMPARISON_PATHS:
                metrics = scalar_comparison_metrics(details[path]["global_loss"], details[PATHS[0]]["global_loss"])
                validate_comparison_metrics(metrics, numerical, kind="loss", context=f"H3 {path} global loss")
                loss_comparisons[path] = metrics

            clipping = {}
            for path in PATHS:
                pre = aggregate(state[path]["preclip_gradient"], parameter_names)
                post = aggregate(state[path]["clipped_gradient"], parameter_names)
                pre_norm = float(torch.linalg.vector_norm(pre.double()))
                post_norm = float(torch.linalg.vector_norm(post.double()))
                if not math.isfinite(pre_norm) or pre_norm <= numerical["preclip_norm_must_exceed"]:
                    raise AssertionError(f"H3 {path} failed to exercise active clipping: {pre_norm}")
                if not math.isfinite(post_norm) or post_norm > numerical["postclip_norm_maximum"]:
                    raise AssertionError(f"H3 {path} postclip norm failed: {post_norm}")
                if torch.equal(pre, post):
                    raise AssertionError(f"H3 {path} gradients were unchanged by clipping")
                clipping[path] = {
                    "clip_grad_norm_return": details[path]["clip_grad_norm_return"],
                    "postclip_norm_from_evidence": post_norm,
                    "preclip_norm_from_evidence": pre_norm,
                }
            for path in COMPARISON_PATHS:
                metrics = scalar_comparison_metrics(
                    clipping[path]["preclip_norm_from_evidence"], clipping[PATHS[0]]["preclip_norm_from_evidence"]
                )
                validate_comparison_metrics(metrics, numerical, kind="loss", context=f"H3 {path} preclip norm")

            for path in PATHS:
                if details[path]["optimizer_steps"] != [1] or details[path]["optimizer_state_dtypes"] != [
                    "torch.float32"
                ]:
                    raise RuntimeError(f"H3 {path} optimizer state/counter drift")
            ddp_rank_hashes = details[PATHS[2]]["post_state_hashes"]
            if len(set(ddp_rank_hashes)) != 1:
                raise AssertionError("H3 DDP ranks ended with non-identical optimizer states")

            evidence = {}
            geometry = []
            for name in parameter_names:
                value = state[PATHS[0]]["initial_parameter"][name]
                geometry.append(
                    {"dtype": str(value.dtype), "elements": value.numel(), "name": name, "shape": list(value.shape)}
                )
            for path in PATHS:
                for family in STORED_FAMILIES:
                    for name in parameter_names:
                        evidence[tensor_key(path, family, name)] = (
                            state[path][family][name].detach().cpu().contiguous()
                        )
            evidence_path = args.output_dir / "h3_evidence.safetensors"
            temporary_evidence = args.output_dir / f".h3_evidence.{os.getpid()}.safetensors"
            save_file(
                evidence,
                str(temporary_evidence),
                metadata={
                    "protocol_id": H3_PROTOCOL_ID,
                    "scenario_id": args.scenario_id,
                    "chunk_size": str(args.chunk_size),
                    "harness_amendment_sha256": harness_amendment_sha256,
                },
            )
            os.replace(temporary_evidence, evidence_path)

            code_root = source_root()
            source_files = (
                "open_instruct/qwen35_chunked_loss.py",
                "open_instruct/qwen35_qualification_r18_h3.py",
                "scripts/train/qwen35/g2_job_guard.sh",
                "scripts/train/qwen35/leonardo_h3_r18.sbatch",
                "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3.json",
                "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3_harness_amendment_r2.json",
                "scripts/train/qwen35/validate_qwen35_ddp_ga_r18_h3.py",
            )
            worktree_clean = git_output("status", "--porcelain") == ""
            if not worktree_clean:
                raise RuntimeError("R18 H3 requires a clean immutable source worktree")
            report = {
                "allowed_conclusion": (
                    "This scenario/candidate passed H3; only completion and independent validation of the full eight-run H3 set may authorize H4."
                ),
                "artifact": H3_ARTIFACT,
                "audits": {path: details[path]["audits"] for path in PATHS},
                "case_records": [case["record"] for case in cases],
                "chunk_size": args.chunk_size,
                "clipping": clipping,
                "comparisons": comparisons,
                "contract": {
                    "accumulation_steps": scenario["accumulation_steps"],
                    "global_target_count": scenario["global_target_count"],
                    "per_rank_target_counts": scenario["per_rank_target_counts"],
                    "target_counts_by_slot_rank": scenario["target_counts_by_slot_rank"],
                    "world_size": world_size,
                },
                "decision": {
                    "all_gating_comparisons_passed": True,
                    "allowed_successor": "H4_only",
                    "scientific_training_authorized": False,
                },
                "environment": {
                    "autocast": {"device_type": "cuda", "dtype": "torch.bfloat16", "enabled": True},
                    "backend": torch.distributed.get_backend(),
                    "cuda_version": torch.version.cuda,
                    "device_names": gathered_devices,
                    "liger_modules_by_rank": gathered_liger,
                    "torch_version": torch.__version__,
                    "world_size": world_size,
                },
                "h3_manifest_sha256": h3_sha256,
                "harness_amendment": {
                    "attempt01_failure_closure_sha256": sha256_file(args.attempt01_failure_closure),
                    "cuda_device_binding": "before_nccl_initialization_with_explicit_device_id",
                    "human_protocol_sha256": sha256_file(args.harness_amendment_human_protocol),
                    "machine_manifest_sha256": harness_amendment_sha256,
                    "output_directory_initialization": output_directory_initialization,
                    "preregistration_closure_sha256": sha256_file(args.harness_amendment_preregistration_closure),
                    "status": harness_amendment["status"],
                },
                "human_protocol_sha256": h3["human_protocol"]["sha256"],
                "losses": {
                    "comparisons": loss_comparisons,
                    "global_unscaled": {path: details[path]["global_loss"] for path in PATHS},
                    "per_case_unscaled": {path: details[path]["per_case_losses"] for path in PATHS},
                },
                "optimizer": {
                    "ddp_rank_post_step_state_sha256": ddp_rank_hashes,
                    "floating_state_dtypes": ["torch.float32"],
                    "foreach": False,
                    "fused": True,
                    "gradient_dtypes": ["torch.float32"],
                    "parameter_dtypes": ["torch.float32"],
                    "step_counters": {path: details[path]["optimizer_steps"] for path in PATHS},
                },
                "protocol_id": H3_PROTOCOL_ID,
                "r18_manifest_sha256": sha256_file(args.r18_manifest),
                "scaling": {
                    "ddp_case_losses": details[PATHS[2]]["case_losses"],
                    "global_target_count": scenario["global_target_count"],
                    "world_size_multiplier": world_size,
                },
                "scenario_id": args.scenario_id,
                "schema_version": 1,
                "source_attestation": {
                    "git_commit": git_output("rev-parse", "HEAD"),
                    "git_worktree_clean": worktree_clean,
                    "implementation_id": IMPLEMENTATION_ID,
                    "liger_modules_imported": imported_liger,
                    "source_files_sha256": {name: sha256_file(code_root / name) for name in source_files},
                },
                "status": "passed",
                "tensor_evidence": {
                    "bytes": evidence_path.stat().st_size,
                    "families": list(STORED_FAMILIES),
                    "file_name": evidence_path.name,
                    "format": "safetensors",
                    "key_count": len(evidence),
                    "parameter_geometry": geometry,
                    "paths": list(PATHS),
                    "sha256": sha256_file(evidence_path),
                },
            }
            write_json_atomic(args.output_dir / "h3_report.json", report)
            outcome[0] = {"status": "passed", "report": str(args.output_dir / "h3_report.json")}
        torch.distributed.broadcast_object_list(outcome, src=0)
        if not outcome[0] or outcome[0].get("status") != "passed":
            raise RuntimeError(f"H3 rank-0 decision failed: {outcome[0]}")
        torch.distributed.barrier()
        if rank == 0:
            print(json.dumps(outcome[0], sort_keys=True))
    except Exception as error:
        failure = {
            "artifact": "qwen35_r18_h3_rank_failure",
            "chunk_size": args.chunk_size,
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "rank": rank,
            "scenario_id": args.scenario_id,
            "schema_version": 1,
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(args.output_dir / f"rank_{rank:02d}_failure.json", failure)
        except Exception:
            pass
        raise
    finally:
        if torch.distributed.is_initialized():
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
