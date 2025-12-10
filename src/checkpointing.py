"""Model checkpointing utilities for training resume and best model saving."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from . import config


def get_model_path(experiment_type: str, experiment_value: str, stage: str) -> Path:
    """Get path to saved model checkpoint.
    
    Args:
        experiment_type: Experiment type (e.g., "wavelet", "J", "h1", "arch")
        experiment_value: Experiment value (e.g., "haar", "1", "0.25")
        stage: Training stage ("stage1" or "stage2")
        
    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = config.RESULTS_ROOT / "checkpoints" / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{experiment_type}_{experiment_value}.pt"
    return checkpoint_dir / filename


def save_best_model(
    model: torch.nn.Module,
    experiment_type: str,
    experiment_value: str,
    stage: str,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    patience_counter: int = 0,
    early_stopping_patience: int = 10,
    best_metric: float = 0.0,
    training_completed: bool = False,
    config_dict: Optional[Dict[str, Any]] = None,
    current_model_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Save best model checkpoint.
    
    Args:
        model: Model to save
        experiment_type: Experiment type identifier
        experiment_value: Experiment value identifier
        stage: Training stage ("stage1" or "stage2")
        step: Current training step
        optimizer: Optimizer state (optional)
        patience_counter: Current patience counter
        early_stopping_patience: Early stopping patience
        best_metric: Best metric value achieved
        training_completed: Whether training is completed
        config_dict: Configuration dictionary to save
        current_model_state: Current model state (optional, for resume)
    """
    checkpoint_path = get_model_path(experiment_type, experiment_value, stage)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "patience_counter": patience_counter,
        "early_stopping_patience": early_stopping_patience,
        "best_metric": best_metric,
        "training_completed": training_completed,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if config_dict is not None:
        checkpoint["config_dict"] = config_dict
    
    if current_model_state is not None:
        checkpoint["current_model_state"] = current_model_state
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(
    experiment_type: str,
    experiment_value: str,
    stage: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    resume_training: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load model checkpoint.
    
    Args:
        experiment_type: Experiment type identifier
        experiment_value: Experiment value identifier
        stage: Training stage ("stage1" or "stage2")
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        resume_training: If True, load current state; if False, load best state
        
    Returns:
        Checkpoint metadata dictionary or None if checkpoint doesn't exist
    """
    checkpoint_path = get_model_path(experiment_type, experiment_value, stage)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    if resume_training and "current_model_state" in checkpoint:
        model.load_state_dict(checkpoint["current_model_state"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "step": checkpoint.get("step", 0),
        "patience_counter": checkpoint.get("patience_counter", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "early_stopping_patience": checkpoint.get("early_stopping_patience", 10),
        "training_completed": checkpoint.get("training_completed", False),
    }


def check_if_training_completed(experiment_type: str, experiment_value: str, stage: str) -> bool:
    """Check if training was already completed for an experiment.
    
    Args:
        experiment_type: Experiment type identifier
        experiment_value: Experiment value identifier
        stage: Training stage ("stage1" or "stage2")
        
    Returns:
        True if training was completed, False otherwise
    """
    checkpoint_path = get_model_path(experiment_type, experiment_value, stage)
    
    if not checkpoint_path.exists():
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("training_completed", False)


def save_checkpoint(
    model: torch.nn.Module,
    step: int,
    checkpoint_dir: Path,
    name: str,
    epoch: Optional[int] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a simple checkpoint (used by main.py).
    
    Args:
        model: Model to save
        step: Training step
        checkpoint_dir: Directory to save checkpoint
        name: Checkpoint name
        epoch: Epoch number (optional)
        config_dict: Configuration dictionary (optional)
        metrics: Metrics dictionary (optional)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if config_dict is not None:
        checkpoint["config_dict"] = config_dict
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, checkpoint_path)

