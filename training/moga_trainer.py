#!/usr/bin/env python3
"""
MoGA (Memory-object-conditioned Gated-rank Adaptation) trainer implementation
Trains only MoGA adapter parameters + image-encoder LayerNorms.
Everything else in SAM 2 is frozen.
"""

import gc
import json
import logging as py_logging  # Rename to avoid conflict with config
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)

CORE_LOSS_KEY = "core_loss"

def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model

@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"

@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)

@dataclass
class DistributedConf:
    backend: Optional[str] = None
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30

@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: Optional[bool] = None
    cudnn_allow_tf32: Optional[bool] = None

@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self

@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False

class MoGATrainer:
    """
    Standalone trainer for fine-tuning SAM 2 with MoGA + image-encoder LayerNorm co-training.
    Trains:
    1. MoGA adapter parameters (memory-attention gating + LoRA)
    2. ALL LayerNorm parameters in Image Encoder
    Does NOT inherit from Trainer. All logic is self-contained.
    """
    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0                       

        self._infer_distributed_backend_if_none(distributed, accelerator)
        self._setup_device(accelerator)
        self._setup_torch_dist_and_backend(cuda, distributed)
        # self.rank is assigned by _setup_ddp_distributed_training above.

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()
        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        # 1. Instantiate the model.
        self.model = instantiate(self.model_conf, _convert_="all")
        self.model.to(self.device)
        
        # 2. Load pretrained SAM 2 checkpoint (strict=False: MoGA adapters are missing on purpose).
        ckpt_path = None
        try:
            ckpt_path = checkpoint.get('model_weight_initializer', {}).get('state_dict', {}).get('checkpoint_path', None)
        except Exception:
            pass
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=False)
            
        # 3. Freeze the whole SAM2 model; re-enable MoGA adapters below, plus all
        #    image-encoder LayerNorms (this is the paper's training setup).
        for p in self.model.parameters():
            p.requires_grad = False

        moga_params = self._get_moga_parameters()
        for p in moga_params:
            p.requires_grad = True

        # ALSO unfreeze ALL LayerNorm parameters in Image Encoder
        py_logging.info("\n" + "="*80)
        py_logging.info("Unfreezing LayerNorm parameters in Image Encoder...")
        py_logging.info("="*80)

        layernorm_count = 0
        layernorm_param_names = []

        for name, module in self.model.named_modules():
            if 'image_encoder' in name and isinstance(module, torch.nn.LayerNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}"
                    param.requires_grad = True
                    layernorm_count += param.numel()
                    layernorm_param_names.append(full_name)

        py_logging.info(f"Found {len(layernorm_param_names)} LayerNorm parameters ({layernorm_count:,} params)")
        for name in layernorm_param_names[:5]:
            py_logging.info(f"  - {name}")
        if len(layernorm_param_names) > 5:
            py_logging.info(f"  ... and {len(layernorm_param_names) - 5} more")

        # Print only the total number of trainable parameters for verification
        trainable_count = 0
        moga_count = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # Skip uninitialized parameters (LazyLinear)
                try:
                    count = p.numel()
                    trainable_count += count
                    if 'memory_attention' in name and any(k in name for k in ['sa_gate', 'ca_gate', 'sa_lora', 'ca_lora']):
                        moga_count += count
                except ValueError as e:
                    if "uninitialized parameter" in str(e):
                        print(f"Skipping uninitialized parameter: {name}")
                        continue
                    else:
                        raise e

        py_logging.info(f"\n{'='*80}")
        py_logging.info(f"[MoGA trainer] Total trainable parameters: {trainable_count:,}")
        py_logging.info(f"  - MoGA parameters:  {moga_count:,}")
        py_logging.info(f"  - LayerNorm parameters: {layernorm_count:,}")
        py_logging.info(f"{'='*80}\n")
        
        # 4. Standard trainer setup from here on.
        self._setup_components()
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")
        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()
        self.load_checkpoint()
        
        # Initialize any lazy parameters before DDP wrapping
        self._initialize_lazy_parameters()
        
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

        # Optional wandb logging. Set MOGA_WANDB_PROJECT (and optionally
        # MOGA_WANDB_RUN) in the environment to enable; unset = no wandb at all.
        self._wandb_enabled = False
        if self.rank == 0 and os.environ.get("MOGA_WANDB_PROJECT"):
            wandb.init(project=os.environ["MOGA_WANDB_PROJECT"],
                       name=os.environ.get("MOGA_WANDB_RUN"))
            self._wandb_enabled = True

    def _get_moga_parameters(self):
        """Collect MoGA adapter parameters from every memory-attention layer.

        These are the only parameters trained here besides the image-encoder
        LayerNorms (unfrozen below). Everything else in SAM 2 stays frozen.
        """
        params = []
        if not hasattr(self.model, 'memory_attention'):
            return params
        memory_attention = self.model.memory_attention
        if not hasattr(memory_attention, 'layers'):
            return params
        for layer in memory_attention.layers:
            # LoRA A/B matrices for self-attention Q/K and cross-attention Q.
            if hasattr(layer, 'sa_lora_A'):
                params.extend([layer.sa_lora_A.weight, layer.sa_lora_B.weight])
            if hasattr(layer, 'ca_lora_A'):
                params.extend([layer.ca_lora_A.weight, layer.ca_lora_B.weight])
            # Gate modules (shared across objects; per-object via batch dim B).
            if hasattr(layer, 'sa_gate'):
                params.extend(list(layer.sa_gate.parameters()))
            if hasattr(layer, 'ca_gate'):
                params.extend(list(layer.ca_gate.parameters()))
        return params

    def _initialize_lazy_parameters(self):
        """Initialize lazy parameters with a dummy forward pass"""
        self.model.eval()
        
        # Create a dummy batch to initialize lazy parameters
        try:
            # Import required classes
            from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
            import torch
            
            # Create minimal dummy data
            batch_size = 1
            num_frames = 2
            height = 64  # Small size for dummy initialization
            width = 64
            num_objects = 1
            
            dummy_images = torch.randn(num_frames, batch_size, 3, height, width).to(self.device)
            dummy_masks = torch.randint(0, 2, (num_frames, num_objects, height, width)).bool().to(self.device)
            dummy_obj_to_frame_idx = torch.zeros(num_frames, num_objects, 2, dtype=torch.int32).to(self.device)
            
            # Create dummy metadata
            dummy_metadata = BatchedVideoMetaData(
                unique_objects_identifier=torch.zeros(batch_size, 3, dtype=torch.long).to(self.device),
                frame_orig_size=torch.tensor([[height, width]], dtype=torch.long).to(self.device)
            )
            
            # Create proper BatchedVideoDatapoint
            dummy_batch = BatchedVideoDatapoint(
                img_batch=dummy_images,
                obj_to_frame_idx=dummy_obj_to_frame_idx,
                masks=dummy_masks,
                metadata=dummy_metadata,
                dict_key='all'
            )
            
            with torch.no_grad():
                try:
                    _ = self.model(dummy_batch)
                    print("Successfully initialized lazy parameters with dummy forward pass")
                except Exception as e:
                    print(f"Warning: Could not run full forward pass for initialization: {e}")
                    # Try to initialize just the memory attention part
                    self._initialize_memory_attention_lazy_params()
                    
        except Exception as e:
            print(f"Warning: Could not initialize lazy parameters: {e}")
            print("Attempting alternative initialization...")
            self._initialize_memory_attention_lazy_params()

    def _initialize_memory_attention_lazy_params(self):
        """Initialize just the memory attention lazy parameters"""
        try:
            import torch
            
            # Initialize masklet_proj in memory attention layers
            if hasattr(self.model, 'memory_attention') and hasattr(self.model.memory_attention, 'layers'):
                for layer in self.model.memory_attention.layers:
                    if hasattr(layer, 'masklet_proj') and isinstance(layer.masklet_proj, torch.nn.LazyLinear):
                        # Create dummy masklet input to initialize LazyLinear (memory dimension is 64)
                        dummy_masklet = torch.randn(1, 64).to(self.device)  # Memory dimension from sam2
                        with torch.no_grad():
                            _ = layer.masklet_proj(dummy_masklet)
                        print(f"Initialized masklet_proj in layer with input dim 64")
                        
        except Exception as e:
            print(f"Could not initialize memory attention lazy params: {e}")

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _setup_timers(self):
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )
        if distributed_conf.comms_dtype is not None:
            from torch.distributed.algorithms import ddp_comm_hooks
            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                py_logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                py_logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        py_logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)
        py_logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def _setup_components(self):
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        self._check_val_key_match(val_keys, phase=val_phase)
        py_logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}
        self.logger = Logger(self.logging_conf)
        # DON'T re-instantiate model! We already did that in __init__ with proper freezing
        # self.model = instantiate(self.model_conf, _convert_="all")
        
        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)
        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")
        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )
        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )
        py_logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        # CRITICAL FIX: Only optimize parameters with requires_grad=True
        trainable_param_names = {
            name for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        py_logging.info(f"\n{'='*80}")
        py_logging.info(f"[FIXED] Constructing optimizer with {len(trainable_param_names)} trainable parameters")
        for name in sorted(trainable_param_names):
            py_logging.info(f"  - {name}")
        py_logging.info(f"{'='*80}\n")

        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=trainable_param_names,  # THIS IS THE FIX!
            validate_param_groups=False,  # Don't validate against ALL params
        )

        # Verify optimizer only contains trainable parameters
        total_params_in_optimizer = sum(
            p.numel() for group in self.optim.optimizer.param_groups
            for p in group['params']
        )
        py_logging.info(f"Optimizer contains {total_params_in_optimizer:,} parameters\n")

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            py_logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, path):
        py_logging.info(f"Resuming training from {path}")
        with g_pathmgr.open(path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
            strict=False
        )
        try:
            self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            py_logging.warning(f"Could not load optimizer state: {e}. Starting with a fresh optimizer.")
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_meter_values = checkpoint.get("best_meter_values", {})
        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        if self.rank == 0 and self._wandb_enabled:
            log_dict = {loss_str: loss.item()}
            log_dict.update({k: v.item() if hasattr(v, 'item') else v for k, v in step_losses.items()})
            wandb.log(log_dict, step=self.steps[phase])

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                py_logging.info(f"Resuming training from epoch: {self.epoch}")
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    py_logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)

            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            data_time.update(time.time() - end)
            batch = batch.to(self.device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        py_logging.info(f"Meters: {out_dict}")
        
        if self.rank == 0:
            epoch_log = {k: v.avg for k, v in loss_mts.items()}
            epoch_log.update({k: v.avg for k, v in extra_loss_mts.items()})
            if self._wandb_enabled:
                wandb.log({f"epoch/{k}": v for k, v in epoch_log.items()}, step=self.epoch)
        
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    py_logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                all_params = []
                for group in self.optim.optimizer.param_groups:
                    all_params.extend(group['params'])

                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        py_logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        if self.rank == 0:
            epoch_log = {k: v.avg for k, v in loss_mts.items()}
            epoch_log.update({k: v.avg for k, v in extra_loss_mts.items()})
            if self._wandb_enabled:
                wandb.log({f"epoch/{k}": v for k, v in epoch_log.items()}, step=self.epoch)
        
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        self.optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            py_logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        self.scaler.scale(loss).backward()
        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        py_logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        py_logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            assert len(val_keys) == len(set(val_keys)), f"Duplicate keys in val datasets, keys: {val_keys}"
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )
            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                assert all([k in loss_keys for k in val_keys]), (
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        if g_pathmgr.exists(checkpoint_path):
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success