"""
Beaker-free SFT training script for OLMo-3 7B on SLURM clusters.

Adapted from OLMo-core's Olmo-3-7B-SFT.py with all Beaker dependencies removed.
Designed for CINECA Leonardo (4× A100 64GB per node) but works on any SLURM cluster.

Usage (via torchrun, typically called from an sbatch script):
    torchrun --nproc_per_node=4 Olmo-3-7B-SFT-slurm.py train \\
        my-run-name \\
        /path/to/olmocore-checkpoint/model_and_optim \\
        --root_dir /path/to/work/dir \\
        --dataset_path /path/to/tokenized/data \\
        --seq_len 32768 \\
        --num_nodes 2 \\
        --gpus_per_node 4 \\
        --global_batch_size 1048576

See scripts/slurm/sft/train_dolci_instruct_leonardo.sh for a complete example.
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, cast

from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import copy_dir, dir_is_empty, get_parent, join_path
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 16_384
DEFAULT_NUM_NODES = 1
DEFAULT_GPUS_PER_NODE = 4  # Leonardo has 4 A100s per node
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int
    gpu_type: str
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: Optional[int] = None

    def __post_init__(self):
        assert self.global_batch_size_tokens > 0, "global_batch_size_tokens must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert (
            self.sequence_length & (self.sequence_length - 1)
        ) == 0, "sequence_length must be a power of 2"
        assert self.world_size > 0, "world_size must be positive"
        assert (self.world_size & (self.world_size - 1)) == 0, "world_size must be a power of 2"

        max_tokens_per_rank = MAX_RANK_MICROBATCH_SIZE_TOKENS
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        if self.sequence_length > max_tokens_per_rank:
            min_cp_degree = 2
            while (self.sequence_length // min_cp_degree) > max_tokens_per_rank:
                min_cp_degree *= 2

            self.cp_degree = min_cp_degree
            log.info(
                f"Sequence length ({self.sequence_length} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). Setting cp_degree={self.cp_degree}"
            )

        cp_factor = self.cp_degree if self.cp_degree is not None else 1
        dp_world_size = self.world_size // cp_factor
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        if rank_batch_size_tokens > max_tokens_per_rank * cp_factor:
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > (
                max_tokens_per_rank * cp_factor
            ):
                self.grad_accum_steps *= 2

            self.rank_microbatch_size_tokens = rank_batch_size_tokens // self.grad_accum_steps
            log.info(
                f"Rank batch size ({rank_batch_size_tokens} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). "
                f"Using grad_accum_steps={self.grad_accum_steps}"
            )
        else:
            self.rank_microbatch_size_tokens = rank_batch_size_tokens
            self.grad_accum_steps = 1

        seq_per_rank = self.sequence_length // (self.cp_degree or 1)
        assert self.rank_microbatch_size_tokens % seq_per_rank == 0, (
            "rank_microbatch_size_tokens must be divisible by sequence_length/cp_degree (got "
            f"{self.rank_microbatch_size_tokens} and {seq_per_rank})"
        )
        self.rank_microbatch_size_sequences = (
            self.rank_microbatch_size_tokens // seq_per_rank
        )

        total_tokens = self.rank_microbatch_size_tokens * dp_world_size * self.grad_accum_steps
        assert self.global_batch_size_tokens == total_tokens, (
            "global_batch_size_tokens must equal "
            "(rank_microbatch_size_tokens * dp_world_size * grad_accum_steps) (got "
            f"{self.global_batch_size_tokens} and {total_tokens})"
        )


def build_sft_dataset(
    work_dir: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    dataset_path: str,
) -> NumpyPackedFSLDatasetConfig:
    clean_path = dataset_path.rstrip("/")
    token_id_paths = [f"{clean_path}/token_ids_part_*.npy"]
    label_mask_paths = [f"{clean_path}/labels_mask_*.npy"]

    dataset = NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        paths=token_id_paths,
        expand_glob=True,
        label_mask_paths=label_mask_paths,
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=sequence_length,
    )

    return dataset


@dataclass
class SFTConfig(Config):
    run_name: str
    model: TransformerConfig
    dataset: Optional[NumpyPackedFSLDatasetConfig]
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int

    @classmethod
    def build(
        cls,
        *,
        run_name: str,
        seq_len: int,
        num_nodes: int,
        gpus_per_node: int,
        global_batch_size: int,
        root_dir: str,
        gpu_type: str,
        overrides: List[str],
        dataset_path: str,
        save_folder: Optional[str] = None,
        lr: float = 8e-05,
        max_epochs: int = 3,
        max_steps: Optional[int] = None,
        init_seed: int = 33333,
    ) -> "SFTConfig":
        user_name = os.environ.get("USER", "unknown")
        work_dir = os.path.join(root_dir, "dataset-cache")

        tokenizer_config = TokenizerConfig.dolma2()
        dataset_config = build_sft_dataset(
            work_dir=work_dir,
            tokenizer_config=tokenizer_config,
            sequence_length=seq_len,
            dataset_path=dataset_path,
        )

        bs_config = BatchSizeConfig(
            sequence_length=seq_len,
            world_size=num_nodes * gpus_per_node,
            global_batch_size_tokens=global_batch_size,
            gpu_type=gpu_type,
        )
        if get_local_rank() == 0:
            print("Batch size config (before overrides):")
            print(bs_config)

        dp_shard_degree = gpus_per_node // (bs_config.cp_degree or 1)
        if not dp_shard_degree > 0:
            raise OLMoConfigurationError(f"dp_shard_degree ({dp_shard_degree}) must be positive.")

        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward", "blocks.*.attention"],
        )

        cp_config = (
            (
                TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
                if dataset_config.generate_doc_lengths
                else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
            )
            if bs_config.cp_degree
            else None
        )

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=gpus_per_node // (bs_config.cp_degree or 1),
        )

        model = TransformerConfig.olmo3_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
        )

        if save_folder is None:
            save_folder = os.path.join(root_dir, "checkpoints", user_name, "olmo-sft", run_name)

        config = SFTConfig(
            run_name=run_name,
            model=model,
            dataset=None,
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=bs_config.global_batch_size_tokens, seed=34521, num_workers=4
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=bs_config.rank_microbatch_size_tokens,
                max_sequence_length=bs_config.sequence_length,
                z_loss_multiplier=None,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=lr,
                    weight_decay=0.0,
                    betas=(0.9, 0.95),
                    compile=False,
                ),
                dp_config=dp_config,
                cp_config=cp_config,
                ac_config=ac_config,
                scheduler=LinearWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=save_folder,
                load_strategy=LoadStrategy.never,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.steps(max_steps) if max_steps is not None else Duration.epochs(max_epochs),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000, ephemeral_save_interval=500, save_async=False
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity=None,
                    project="chatflow_v2",
                    enabled=False,
                    cancel_tags=None,
                ),
            ),
            init_seed=init_seed,
        ).merge(overrides)

        config.dataset = dataset_config

        print(config)

        return config


def train(checkpoint: str, config: SFTConfig, no_save_tokenizer: bool):
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    if config.dataset is not None:
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = config.trainer.build(train_module, data_loader)

        if not no_save_tokenizer and get_rank() == 0:
            tokenizer_path = join_path(get_parent(dataset.paths[0]), "tokenizer")
            if not dir_is_empty(tokenizer_path):
                log.info("Saving tokenizer...")
                destination_path = join_path(trainer.save_folder, "tokenizer")
                if not dir_is_empty(destination_path):
                    log.info(f"Tokenizer already exists: {destination_path}")
                else:
                    log.info(f"Saving tokenizer to {destination_path}")
                    copy_dir(tokenizer_path, destination_path)

        config_dict = config.as_config_dict()
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

        log.info("Loading checkpoint...")
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            log.info(
                f"No checkpoint found in save folder '{trainer.save_folder}', "
                f"attempting to load from pretraining checkpoint '{checkpoint}'"
            )
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        else:
            log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

        trainer.fit()
    else:
        log.error(f"Config dataset is None: {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT training for OLMo-3 7B on SLURM (Beaker-free).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-node, 4 GPUs
  torchrun --nproc_per_node=4 %(prog)s train my-run /path/to/ckpt \\
      --root_dir /path/to/work --dataset_path /path/to/data

  # Multi-node via srun (see train_dolci_instruct_leonardo.sh)
  srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 ... %(prog)s train ...
""",
    )

    parser.add_argument(
        "cmd",
        choices=["train", "dry_run"],
        help="'train' to run training, 'dry_run' to print config and exit.",
    )
    parser.add_argument(
        "run_name",
        help="Name of the run (used for checkpoint dir and W&B).",
    )
    parser.add_argument(
        "pretrain_checkpoint",
        help="Path to the OLMo-core native checkpoint (model_and_optim dir).",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory for dataset cache and checkpoints (should be on shared FS).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the pre-tokenized SFT dataset (containing token_ids_part_*.npy).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=DEFAULT_NUM_NODES,
        help="Number of nodes.",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node (default: 4 for Leonardo A100).",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=64 * DEFAULT_SEQUENCE_LENGTH,
        help="Global batch size in tokens.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA A100 64GB",
        help="GPU type string (used for batch size calculation).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8e-05,
        help="Learning rate (default: 8e-5 for Instruct SFT).",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs (ignored if --max_steps is set).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps. Overrides --max_epochs when set.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="Override checkpoint save folder (default: {root_dir}/checkpoints/{user}/olmo-sft/{run_name}).",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Disable saving the dataset's tokenizer in the model directory.",
    )
    parser.add_argument(
        "--init_seed",
        type=int,
        default=33333,
        help="Seed for model init, optimizer state, and torch RNG (passed to seed_all). "
             "Use distinct seeds for paired replications (e.g., A3/Q2 with K=3).",
    )

    args, overrides = parser.parse_known_args()

    if args.cmd == "train":
        prepare_training_environment(backend="nccl")
    else:
        prepare_cli_environment()

    config = SFTConfig.build(
        run_name=args.run_name,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        global_batch_size=args.global_batch_size,
        root_dir=args.root_dir,
        gpu_type=args.gpu_type,
        overrides=overrides,
        dataset_path=args.dataset_path,
        save_folder=args.save_folder,
        lr=args.lr,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        init_seed=args.init_seed,
    )

    if get_local_rank() == 0:
        print(config)

    if args.cmd == "dry_run":
        pass
    elif args.cmd == "train":
        try:
            train(args.pretrain_checkpoint, config, args.no_save_tokenizer)
        finally:
            teardown_training_environment()
