"""Training functions for mammography pipeline."""

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from . import config
from .datasets import CBISDet, MILBagsCBIS, collate_det, collate_mil
from .models import (
    FasterRCNNWithAttnMIL,
    Stage1MIL,
    build_faster_rcnn,
    transplant_stage1_to_stage2,
)



@dataclass
class TrainCfg:
    """Training configuration for Stage-1."""

    backbone: str = "convnext_tiny"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    steps: int = 1
    batch_size: int = 2
    wavelet: str = "haar"
    J: int = 1
    h1_pct: float = 0.25


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """Normalize state_dict keys by removing torch.compile() prefix.
    
    When using torch.compile(), state_dict keys have '_orig_mod.' prefix.
    This function strips that prefix to make the state_dict compatible
    with uncompiled models.
    
    Args:
        state_dict: State dict that may have '_orig_mod.' prefixed keys
        
    Returns:
        State dict with normalized keys (prefix removed if present)
    """
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            normalized[k[len("_orig_mod."):]] = v
        else:
            normalized[k] = v
    return normalized


def train_stage1_mil(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: TrainCfg,
    val_limit: Optional[int] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_metric: str = "loss",
    experiment_type: Optional[str] = None,
    experiment_value: Optional[str] = None,
    resume: bool = True,
    gpu_id: Optional[int] = None,
) -> Stage1MIL:
    """Train Stage-1 MIL model with checkpointing and resume support.
    
    Args:
        df_tr: Training dataframe
        df_va: Validation dataframe
        cfg: Training configuration
        val_limit: Maximum validation batches
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for improvement
        early_stopping_metric: Metric to monitor ("loss" or "auc")
        experiment_type: Experiment type for checkpointing (e.g., "wavelet", "J", "h1", "arch")
        experiment_value: Experiment value for checkpointing (e.g., "haar", "1", "0.25")
        resume: Whether to resume from checkpoint if exists
        gpu_id: GPU ID for monitoring (optional, inferred from CUDA_VISIBLE_DEVICES if not provided)
        
    Returns:
        Trained Stage-1 MIL model
    """
    from .checkpointing import (
        load_model_checkpoint,
        save_best_model,
        check_if_training_completed,
        get_model_path,
    )
    
    
    if experiment_type is not None and experiment_value is not None:
        # Infer GPU ID from CUDA_VISIBLE_DEVICES if not provided
        if gpu_id is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                gpu_id = 0
            elif torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
            else:
                gpu_id = 0
        
        experiment_id = f"{experiment_type}_{experiment_value}_s1"
        
    # Check if training was already completed
    if resume and experiment_type is not None and experiment_value is not None:
        if check_if_training_completed(experiment_type, experiment_value, "stage1"):
            print(f"[S1] Training already completed for {experiment_type}={experiment_value}. Loading model...")
            model = Stage1MIL(cfg.backbone, pretrained=True).to(config.DEVICE)
            checkpoint_meta = load_model_checkpoint(experiment_type, experiment_value, "stage1", model)
            if checkpoint_meta is not None:
                return model
    
    model = Stage1MIL(cfg.backbone, pretrained=True).to(config.DEVICE)
    
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        model = torch.compile(model, mode='default')
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas)
    
    # Try to load checkpoint and resume training
    start_step = 0
    best_metric = float("inf") if early_stopping_metric == "loss" else float("-inf")
    patience_counter = 0
    best_model_state = None
    
    if resume and experiment_type is not None and experiment_value is not None:
        checkpoint_meta = load_model_checkpoint(experiment_type, experiment_value, "stage1", model, opt, resume_training=True)
        if checkpoint_meta is not None:
            # Check if training was completed
            if checkpoint_meta.get("training_completed", False):
                print(f"[S1] Training already completed for {experiment_type}={experiment_value}. Loading best model...")
                checkpoint_path = get_model_path(experiment_type, experiment_value, "stage1")
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
                normalized_state = _normalize_state_dict_keys(checkpoint["model_state_dict"])
                final_model = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
                final_model.load_state_dict(normalized_state)
                return final_model
            
            start_step = checkpoint_meta["step"] + 1
            patience_counter = checkpoint_meta["patience_counter"]
            best_metric = checkpoint_meta.get("best_metric", best_metric)
            early_stopping_patience = checkpoint_meta.get("early_stopping_patience", early_stopping_patience)
            
            # Load best model state from checkpoint for tracking
            checkpoint_path = get_model_path(experiment_type, experiment_value, "stage1")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                best_model_state = _normalize_state_dict_keys(
                    {k: v.cpu().clone() for k, v in checkpoint["model_state_dict"].items()}
                )
            print(f"[S1] Resuming training from step {start_step}, patience={patience_counter}/{early_stopping_patience}, best_metric={best_metric}")

    ds_tr = MILBagsCBIS(
        df_tr,
        split_tag="train",
        wavelet=cfg.wavelet,
        J=cfg.J,
        h1_pct=cfg.h1_pct,
    )
    ds_va = MILBagsCBIS(
        df_va,
        split_tag="val",
        wavelet=cfg.wavelet,
        J=cfg.J,
        h1_pct=cfg.h1_pct,
    )
    
    y_arr = df_tr["label"].astype(int).to_numpy()
    pos = max(1, (y_arr == 1).sum())
    neg = max(1, (y_arr == 0).sum())
    weights = np.where(y_arr == 1, 0.5 / pos, 0.5 / neg).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )
    
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    
    num_workers = 8
    persistent_workers = True
    
    loader_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_mil,
        generator=generator,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mil,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )
    criterion = nn.BCEWithLogitsLoss()

    from torch.amp import autocast, GradScaler
    autocast_context = lambda: autocast('cuda')
    scaler = GradScaler('cuda')
    
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    warmup_batch = next(iter(loader_tr))
    warmup_bags = warmup_batch["bags"].to(config.DEVICE, non_blocking=True)
    warmup_y = warmup_batch["y"].to(config.DEVICE, non_blocking=True)
    with autocast_context():
        warmup_logit, _, _ = model(warmup_bags)
        warmup_loss = criterion(warmup_logit, warmup_y)
    scaler.scale(warmup_loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    del warmup_bags, warmup_y, warmup_logit, warmup_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    if best_model_state is None:
        best_model_state = None  # Will be set when we get first improvement
    steps_since_validation = 0
    # Validate frequently enough to detect improvements within patience window
    # Validate every 10 steps to check for improvements
    validation_interval = max(1, min(10, cfg.steps // 20)) if early_stopping else max(1, cfg.steps // 10)
    
    # Skip to start_step if resuming
    step = start_step
    pbar = tqdm(total=cfg.steps, initial=start_step, desc="[S1] Training", unit="step", leave=True)
    
    # Skip batches if resuming
    loader_iter = iter(loader_tr)
    for _ in range(start_step):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader_tr)
            break
    
    step = start_step
    for batch in loader_iter:
        # Add debug output for first step to show progress
        if step == start_step:
            print(f"[S2] Loading first batch (this may take 30-90s for I/O and topology loading)...", flush=True)
        # Add debug output for first step to show progress
        if step == start_step:
            print(f"[S1] Loading first batch (this may take 30-60s for I/O)...", flush=True)
        bags = batch["bags"].to(config.DEVICE, non_blocking=True)  # Non-blocking transfer
        y = batch["y"].to(config.DEVICE, non_blocking=True)  # Non-blocking transfer
        if step == start_step:
            print(f"[S1] First batch loaded, starting training...", flush=True)
        
        # Mixed precision forward pass
        opt.zero_grad(set_to_none=True)
        with autocast_context():
            logit, A, Z = model(bags)
            loss = criterion(logit, y)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        step += 1
        steps_since_validation += 1
        loss_val = loss.detach().item()
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "patience": f"{patience_counter}/{early_stopping_patience}" if early_stopping else ""})
        del bags, y, logit, A, Z, loss
        
        # Break if we've done enough steps
        if step >= cfg.steps:
            break
        
        if early_stopping and (steps_since_validation >= validation_interval or step >= cfg.steps):
            model.eval()
            val_losses = []
            val_ys = []
            val_ps = []
            val_total = len(loader_va) if val_limit is None else min(val_limit, len(loader_va))
            with torch.no_grad():
                val_seen = 0
                for val_batch in loader_va:
                    val_bags = val_batch["bags"].to(config.DEVICE, non_blocking=True)
                    val_y = val_batch["y"].to(config.DEVICE, non_blocking=True)
                    val_y_np = val_y.cpu().numpy()
                    with autocast_context():
                        val_logit, _, _ = model(val_bags)
                        val_loss = criterion(val_logit, val_y)
                    val_p = torch.sigmoid(val_logit).cpu().numpy()
                    val_losses.append(val_loss.detach().item())
                    val_ys.append(val_y_np)
                    val_ps.append(val_p)
                    del val_bags, val_y, val_logit, val_loss
                    val_seen += 1
                    if val_limit is not None and val_seen >= val_total:
                        break
            
            if val_ys and val_losses:
                val_yv = np.concatenate(val_ys).astype(int)
                val_pv = np.concatenate(val_ps).astype(float)
                avg_val_loss = np.mean(val_losses)
                
                val_auc = None
                if len(np.unique(val_yv)) >= 2:
                    val_auc = roc_auc_score(val_yv, val_pv)
                
                if val_auc is not None:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "val_loss": f"{avg_val_loss:.4f}", "val_auc": f"{val_auc:.4f}", "patience": f"{patience_counter}/{early_stopping_patience}"})
                else:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "val_loss": f"{avg_val_loss:.4f}", "patience": f"{patience_counter}/{early_stopping_patience}"})
                
                if early_stopping_metric == "loss":
                    current_metric = avg_val_loss
                    improved = current_metric < (best_metric - early_stopping_min_delta)
                else:
                    if val_auc is None:
                        improved = False
                    else:
                        current_metric = val_auc
                        improved = current_metric > (best_metric + early_stopping_min_delta)
                
                if improved:
                    best_metric = current_metric
                    patience_counter = 0
                    best_model_state = _normalize_state_dict_keys(
                        {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    )
                    
                    if experiment_type is not None and experiment_value is not None:
                        current_state = _normalize_state_dict_keys(
                            {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        )
                        best_model_for_save = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
                        best_model_for_save.load_state_dict(best_model_state)
                        save_best_model(
                            best_model_for_save,
                            experiment_type,
                            experiment_value,
                            "stage1",
                            step,
                            optimizer=opt,
                            patience_counter=patience_counter,
                            early_stopping_patience=early_stopping_patience,
                            best_metric=best_metric,
                            training_completed=False,
                            config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
                            current_model_state=current_state,
                        )
                        del best_model_for_save, current_state
                    
                    if val_auc is not None:
                        print(f"\n[S1] Step {step}: Improved! val_loss={avg_val_loss:.4f}, val_auc={val_auc:.4f}")
                    else:
                        print(f"\n[S1] Step {step}: Improved! val_loss={avg_val_loss:.4f}")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[S1] Early stopping triggered at step {step} (patience: {patience_counter}/{early_stopping_patience})")
                        if experiment_type is not None and experiment_value is not None and best_model_state is not None:
                            best_model_for_save = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
                            best_model_for_save.load_state_dict(best_model_state)
                            save_best_model(
                                best_model_for_save,
                                experiment_type,
                                experiment_value,
                                "stage1",
                                step,
                                optimizer=opt,
                                patience_counter=patience_counter,
                                early_stopping_patience=early_stopping_patience,
                                best_metric=best_metric,
                                training_completed=True,
                                config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
                            )
                            del best_model_for_save
                        break
                    else:
                        if experiment_type is not None and experiment_value is not None:
                            current_state = _normalize_state_dict_keys(
                                {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            )
                            if best_model_state is not None:
                                best_model_for_save = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
                                best_model_for_save.load_state_dict(best_model_state)
                                save_best_model(
                                    best_model_for_save,
                                    experiment_type,
                                    experiment_value,
                                    "stage1",
                                    step,
                                    optimizer=opt,
                                    patience_counter=patience_counter,
                                    early_stopping_patience=early_stopping_patience,
                                    best_metric=best_metric,
                                    training_completed=False,
                                    config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
                                    current_model_state=current_state,
                                )
                                del best_model_for_save
                            else:
                                save_best_model(
                                    model,
                                    experiment_type,
                                    experiment_value,
                                    "stage1",
                                    step,
                                    optimizer=opt,
                                    patience_counter=patience_counter,
                                    early_stopping_patience=early_stopping_patience,
                                    best_metric=best_metric,
                                    training_completed=False,
                                    config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
                                )
                            del current_state
            else:
                print(f"\n[S1] Warning: No validation data collected at step {step}")
            
            steps_since_validation = 0
            model.train()
        
        if step >= cfg.steps:
            break
    
    pbar.close()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if early_stopping and best_model_state is not None:
        print(f"[S1] Best model from step {step} (patience: {patience_counter}/{early_stopping_patience}) will be loaded into fresh model")
    
    if experiment_type is not None and experiment_value is not None:
        training_completed = (patience_counter >= early_stopping_patience) if early_stopping else (step >= cfg.steps)
        if best_model_state is not None:
            best_model_for_save = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
            best_model_for_save.load_state_dict(best_model_state)
            save_best_model(
                best_model_for_save,
                experiment_type,
                experiment_value,
                "stage1",
                step,
                optimizer=opt,
                patience_counter=patience_counter,
                early_stopping_patience=early_stopping_patience,
                best_metric=best_metric,
                training_completed=training_completed,
                config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
            )
            del best_model_for_save
        else:
            save_best_model(
                model,
                experiment_type,
                experiment_value,
                "stage1",
                step,
                optimizer=opt,
                patience_counter=patience_counter,
                early_stopping_patience=early_stopping_patience,
                best_metric=best_metric,
                training_completed=training_completed,
                config_dict={"backbone": cfg.backbone, "wavelet": cfg.wavelet, "J": cfg.J, "h1_pct": cfg.h1_pct},
            )
    
    # Return an uncompiled model for use in Stage-2
    if best_model_state is not None:
        final_model = Stage1MIL(cfg.backbone, pretrained=False).to(config.DEVICE)
        final_model.load_state_dict(best_model_state)
        return final_model
    else:
        return model


def train_stage2_det(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    s1_model: Stage1MIL,
    backbone_name: str,
    wavelet: str = "haar",
    J: int = 1,
    h1_pct: float = 0.25,
    steps: int = 1,
    lr: float = 1e-4,
    betas=(0.9, 0.999),
    use_topo: bool = True,
    val_limit: Optional[int] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_metric: str = "loss",
    experiment_type: Optional[str] = None,
    experiment_value: Optional[str] = None,
    resume: bool = True,
    gpu_id: Optional[int] = None,
    **kwargs,  # For enhanced architecture params: fusion_method, use_convnext_topo
):
    """Train Stage-2 detection model with checkpointing and resume support.
    
    Args:
        df_tr: Training dataframe
        df_va: Validation dataframe
        s1_model: Trained Stage-1 model
        backbone_name: Backbone name
        wavelet: Wavelet type
        J: Wavelet decomposition level
        h1_pct: H1 persistence pair keep percentage
        steps: Number of training steps
        lr: Learning rate
        betas: Adam beta parameters
        use_topo: Whether to use topology features
        val_limit: Maximum validation batches
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for improvement
        early_stopping_metric: Metric to monitor ("loss" or "auc")
        experiment_type: Experiment type for checkpointing (e.g., "wavelet", "J", "h1", "arch")
        experiment_value: Experiment value for checkpointing (e.g., "haar", "1", "0.25")
        resume: Whether to resume from checkpoint if exists
        
    Returns:
        Trained Stage-2 detection model
    """
    from .checkpointing import (
        load_model_checkpoint,
        save_best_model,
        check_if_training_completed,
        get_model_path,
    )
    
    # Check if training was already completed
    if resume and experiment_type is not None and experiment_value is not None:
        if check_if_training_completed(experiment_type, experiment_value, "stage2"):
            print(f"[S2] Training already completed for {experiment_type}={experiment_value}. Loading model...")
            base = build_faster_rcnn(backbone_name, pretrained_backbone=True).to(config.DEVICE)
            in_ch = 9 if use_topo else 1
            base.transform.image_mean = [0.0] * in_ch
            base.transform.image_std = [1.0] * in_ch
            model = FasterRCNNWithAttnMIL(base, attn_dim=128, mil_loss_weight=1.0).to(config.DEVICE)
            checkpoint_meta = load_model_checkpoint(experiment_type, experiment_value, "stage2", model)
            if checkpoint_meta is not None:
                return model
    
    # Support enhanced architectures if fusion_method is provided
    fusion_method = kwargs.get('fusion_method', None)
    use_convnext_topo = kwargs.get('use_convnext_topo', False)
    
    if fusion_method is not None:
        from .models_enhanced import EnhancedTimmBackboneWithFPN
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign
        from .models import FasterRCNN, _is_swin, _swin_size_from_name
        
        # Build enhanced backbone
        bb = EnhancedTimmBackboneWithFPN(
            backbone_name,
            pretrained=True,  # Always use pretrained backbone for Stage-2
            fusion_method=fusion_method,
            use_convnext_topo=use_convnext_topo,
        )
        
        fpn_out_names = ["0", "1", "2", "3"]
        n_lvls = len(fpn_out_names)
        anchor_sizes = tuple((32 * (2 ** i),) for i in range(n_lvls))
        aspect_ratios = tuple((0.5, 1.0, 2.0) for _ in range(n_lvls))
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=fpn_out_names, output_size=7, sampling_ratio=2)
        bb.fpn_names = fpn_out_names
        
        base = FasterRCNN(bb, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
        if _is_swin(backbone_name):
            s = _swin_size_from_name(backbone_name)
            base.transform.min_size = (s,)
            base.transform.max_size = s
        base = base.to(config.DEVICE)
    else:
        base = build_faster_rcnn(backbone_name, pretrained_backbone=True).to(
            config.DEVICE
        )

    in_ch = 9 if use_topo else 1
    base.transform.image_mean = [0.0] * in_ch
    base.transform.image_std = [1.0] * in_ch

    model = FasterRCNNWithAttnMIL(
        base, attn_dim=128, mil_loss_weight=1.0
    ).to(config.DEVICE)
    transplant_stage1_to_stage2(s1_model, model)
    
    
    if experiment_type is not None and experiment_value is not None:
        # Infer GPU ID from CUDA_VISIBLE_DEVICES if not provided
        if gpu_id is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                gpu_id = 0
            elif torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
            else:
                gpu_id = 0
        
        experiment_id = f"{experiment_type}_{experiment_value}_s2"
    
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    
    # Try to load checkpoint and resume training
    start_step = 0
    best_metric = float("inf") if early_stopping_metric == "loss" else float("-inf")
    patience_counter = 0
    best_model_state = None
    
    if resume and experiment_type is not None and experiment_value is not None:
        checkpoint_meta = load_model_checkpoint(experiment_type, experiment_value, "stage2", model, opt, resume_training=True)
        if checkpoint_meta is not None:
            # Check if training was completed
            if checkpoint_meta.get("training_completed", False):
                print(f"[S2] Training already completed for {experiment_type}={experiment_value}. Loading best model...")
                checkpoint_path = get_model_path(experiment_type, experiment_value, "stage2")
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
                normalized_state = _normalize_state_dict_keys(checkpoint["model_state_dict"])
                model.load_state_dict(normalized_state)  # Load best model
                return model
            
            start_step = checkpoint_meta["step"] + 1
            patience_counter = checkpoint_meta["patience_counter"]
            best_metric = checkpoint_meta.get("best_metric", best_metric)
            early_stopping_patience = checkpoint_meta.get("early_stopping_patience", early_stopping_patience)
            
            # Load best model state from checkpoint for tracking
            checkpoint_path = get_model_path(experiment_type, experiment_value, "stage2")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                best_model_state = _normalize_state_dict_keys(
                    {k: v.cpu().clone() for k, v in checkpoint["model_state_dict"].items()}
                )
            print(f"[S2] Resuming training from step {start_step}, patience={patience_counter}/{early_stopping_patience}, best_metric={best_metric}")

    ds_tr = CBISDet(
        df_tr,
        split_tag="train",
        wavelet=wavelet,
        J=J,
        h1_pct=h1_pct,
        use_topo=use_topo,
    )
    ds_va = CBISDet(
        df_va,
        split_tag="val",
        wavelet=wavelet,
        J=J,
        h1_pct=h1_pct,
        use_topo=use_topo,
    )
    # Set generator seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    
    # Use multiple workers for parallel data loading to keep GPU busy
    # This prevents CPU-bound data loading from blocking GPU training
    num_workers = 8
    persistent_workers = True
    loader_tr = DataLoader(
        ds_tr,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_det,
        generator=generator,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_det,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
    )

    from torch.amp import autocast, GradScaler
    autocast_context = lambda: autocast('cuda')
    scaler = GradScaler('cuda')
    
    step = 0
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # This avoids the 25+ second stall on the first real training step
    warmup_imgs, warmup_targets, warmup_metas = next(iter(loader_tr))
    warmup_imgs = [x.to(config.DEVICE, non_blocking=True) for x in warmup_imgs]
    warmup_targets = [{k: v.to(config.DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                      for k, v in t.items()} for t in warmup_targets]
    warmup_labels = [float(m["label"]) for m in warmup_metas]
    with autocast_context():
        warmup_losses, _, _ = model(warmup_imgs, targets=warmup_targets, image_labels=warmup_labels)
        warmup_loss = sum(warmup_losses.values())
    scaler.scale(warmup_loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    del warmup_imgs, warmup_targets, warmup_losses, warmup_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Early stopping setup (best_metric and patience_counter may be loaded from checkpoint)
    steps_since_validation = 0
    # Validate frequently enough to detect improvements within patience window
    # Validate every 10 steps to check for improvements
    validation_interval = max(1, min(10, steps // 20)) if early_stopping else max(1, steps // 10)
    
    # Skip to start_step if resuming
    step = start_step
    pbar = tqdm(total=steps, initial=start_step, desc="[S2] Training", unit="step", leave=True)
    
    # Skip batches if resuming
    loader_iter = iter(loader_tr)
    for _ in range(start_step):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader_tr)
            break
    
    for imgs, targets, metas in loader_iter:
        # Add debug output for first step to show progress
        if step == start_step:
            print(f"[S2] Loading first batch (this may take 30-90s for I/O and topology loading)...", flush=True)
        imgs = [x.to(config.DEVICE, non_blocking=True) for x in imgs]  # Non-blocking transfer
        targets = [{k: v.to(config.DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]  # Non-blocking transfer
        img_labels = torch.tensor(
            [int(m["label"]) for m in metas], dtype=torch.float32, device=config.DEVICE
        )
        if step == start_step:
            print(f"[S2] First batch loaded, starting training...", flush=True)
        
        # Mixed precision forward pass
        opt.zero_grad(set_to_none=True)
        with autocast_context():
            losses, dets, img_probs = model(imgs, targets=targets, image_labels=img_labels)
            loss = sum(losses.values())
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        step += 1
        steps_since_validation += 1
        loss_val = loss.detach().item()
        loss_dict = {k: v.detach().item() for k, v in losses.items()}
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss_val:.4f}", **{k: f"{v:.3f}" for k, v in loss_dict.items()}, "patience": f"{patience_counter}/{early_stopping_patience}" if early_stopping else ""})
        del imgs, targets, img_labels, losses, dets, img_probs, loss
        
        # Periodic validation and early stopping check
        if early_stopping and (steps_since_validation >= validation_interval or step >= steps):
            model.eval()
            val_losses = []
            val_labels = []
            val_probs = []
            val_total = len(loader_va) if val_limit is None else min(val_limit, len(loader_va))
            with torch.no_grad():
                va_seen = 0
                for val_imgs, val_targets, val_metas in loader_va:
                    val_imgs = [x.to(config.DEVICE, non_blocking=True) for x in val_imgs]
                    val_targets = [{k: v.to(config.DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                   for k, v in t.items()} for t in val_targets]
                    val_img_labels = torch.tensor(
                        [int(m["label"]) for m in val_metas], dtype=torch.float32, device=config.DEVICE
                    )
                    with autocast_context():
                                # We're in eval mode, so we need to compute loss manually
                                model_output = model(val_imgs, targets=val_targets, image_labels=val_img_labels)
                                if len(model_output) == 3:
                                    val_losses_dict, _, val_img_probs = model_output
                                    val_loss = sum(val_losses_dict.values())
                                    val_losses_dict_to_del = val_losses_dict
                                else:
                                    _, val_img_probs = model_output
                                    # Compute loss manually for validation by converting probs back to logits
                                    val_img_logits = torch.logit(val_img_probs.clamp(min=1e-7, max=1-1e-7))
                                    val_loss = F.binary_cross_entropy_with_logits(val_img_logits, val_img_labels)
                                    val_img_logits_to_del = val_img_logits
                    val_losses.append(val_loss.detach().item())
                    val_labels.extend([int(m["label"]) for m in val_metas])
                    val_probs.extend(val_img_probs.cpu().numpy().tolist())
                    del val_imgs, val_targets, val_img_labels, val_img_probs, val_loss
                    if len(model_output) == 3:
                        del val_losses_dict_to_del
                    else:
                        del val_img_logits_to_del
                    va_seen += 1
                    if val_limit is not None and va_seen >= val_total:
                        break
            
            if val_losses:
                avg_val_loss = np.mean(val_losses)
                val_yv = np.array(val_labels, dtype=int)
                val_pv = np.array(val_probs, dtype=float)
                val_auc = None
                if len(np.unique(val_yv)) >= 2:
                    val_auc = roc_auc_score(val_yv, val_pv)
                
                if val_auc is not None:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "val_loss": f"{avg_val_loss:.4f}", "val_auc": f"{val_auc:.4f}", "patience": f"{patience_counter}/{early_stopping_patience}"})
                else:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "val_loss": f"{avg_val_loss:.4f}", "patience": f"{patience_counter}/{early_stopping_patience}"})
                
                if early_stopping_metric == "loss":
                    current_metric = avg_val_loss
                    improved = current_metric < (best_metric - early_stopping_min_delta)
                else:
                    if val_auc is None:
                        improved = False
                    else:
                        current_metric = val_auc
                        improved = current_metric > (best_metric + early_stopping_min_delta)
                
                if improved:
                    best_metric = current_metric
                    patience_counter = 0
                    best_model_state = _normalize_state_dict_keys(
                        {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    )
                    
                    if experiment_type is not None and experiment_value is not None:
                        current_state = _normalize_state_dict_keys(
                            {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        )
                        best_model_for_save = FasterRCNNWithAttnMIL(base, attn_dim=128, mil_loss_weight=1.0).to(config.DEVICE)
                        best_model_for_save.load_state_dict(best_model_state)
                        save_best_model(
                            best_model_for_save,
                            experiment_type,
                            experiment_value,
                            "stage2",
                            step,
                            optimizer=opt,
                            patience_counter=patience_counter,
                            early_stopping_patience=early_stopping_patience,
                            best_metric=best_metric,
                            training_completed=False,
                            config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
                            current_model_state=current_state,
                        )
                        del best_model_for_save, current_state
                    
                    if val_auc is not None:
                        print(f"\n[S2] Step {step}: Improved! val_loss={avg_val_loss:.4f}, val_auc={val_auc:.4f}")
                    else:
                        print(f"\n[S2] Step {step}: Improved! val_loss={avg_val_loss:.4f}")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[S2] Early stopping triggered at step {step} (patience: {patience_counter}/{early_stopping_patience})")
                        if experiment_type is not None and experiment_value is not None and best_model_state is not None:
                            best_model_for_save = FasterRCNNWithAttnMIL(base, attn_dim=128, mil_loss_weight=1.0).to(config.DEVICE)
                            best_model_for_save.load_state_dict(best_model_state)
                            save_best_model(
                                best_model_for_save,
                                experiment_type,
                                experiment_value,
                                "stage2",
                                step,
                                optimizer=opt,
                                patience_counter=patience_counter,
                                early_stopping_patience=early_stopping_patience,
                                best_metric=best_metric,
                                training_completed=True,
                                config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
                            )
                            del best_model_for_save
                        break
                    else:
                        if experiment_type is not None and experiment_value is not None:
                            current_state = _normalize_state_dict_keys(
                                {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            )
                            if best_model_state is not None:
                                best_model_for_save = FasterRCNNWithAttnMIL(base, attn_dim=128, mil_loss_weight=1.0).to(config.DEVICE)
                                best_model_for_save.load_state_dict(best_model_state)
                                save_best_model(
                                    best_model_for_save,
                                    experiment_type,
                                    experiment_value,
                                    "stage2",
                                    step,
                                    optimizer=opt,
                                    patience_counter=patience_counter,
                                    early_stopping_patience=early_stopping_patience,
                                    best_metric=best_metric,
                                    training_completed=False,
                                    config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
                                    current_model_state=current_state,
                                )
                                del best_model_for_save
                            else:
                                save_best_model(
                                        model,
                                        experiment_type,
                                        experiment_value,
                                        "stage2",
                                        step,
                                        optimizer=opt,
                                        patience_counter=patience_counter,
                                        early_stopping_patience=early_stopping_patience,
                                        best_metric=best_metric,
                                        training_completed=False,
                                        config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
                                    )
                                del current_state
            else:
                print(f"\n[S2] Warning: No validation data collected at step {step}")
            
            steps_since_validation = 0
            model.train()
        
        if step >= steps:
            break
    
    pbar.close()
    
    # Clean up after training (but less aggressively)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[S2] Restored best model from step {step} (patience: {patience_counter}/{early_stopping_patience})")
    
    if experiment_type is not None and experiment_value is not None:
        training_completed = (patience_counter >= early_stopping_patience) if early_stopping else (step >= steps)
        if best_model_state is not None:
            best_model_for_save = FasterRCNNWithAttnMIL(base, attn_dim=128, mil_loss_weight=1.0).to(config.DEVICE)
            best_model_for_save.load_state_dict(best_model_state)
            save_best_model(
                best_model_for_save,
                experiment_type,
                experiment_value,
                "stage2",
                step,
                optimizer=opt,
                patience_counter=patience_counter,
                early_stopping_patience=early_stopping_patience,
                best_metric=best_metric,
                training_completed=training_completed,
                config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
            )
            del best_model_for_save
        else:
            save_best_model(
                model,
                experiment_type,
                experiment_value,
                "stage2",
                step,
                optimizer=opt,
                patience_counter=patience_counter,
                early_stopping_patience=early_stopping_patience,
                best_metric=best_metric,
                training_completed=training_completed,
                config_dict={"backbone": backbone_name, "wavelet": wavelet, "J": J, "h1_pct": h1_pct, "use_topo": use_topo},
            )
    
    return model

