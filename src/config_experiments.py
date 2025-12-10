"""Experiment configuration matching the paper exactly."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config


@dataclass
class ExperimentConfig:
    """Configuration for the paper experiment.
    
    Matches the paper exactly:
    - Stage I: 625 iterations, batch size 8, grayscale only
    - Stage II: 100 iterations, batch size 1, with topology (9 channels)
    - ConvNeXt-Tiny backbone only
    - CBIS-DDSM 80/20 train/val split (stratified by malignancy)
    - Test on INbreast and CMMD
    """
    
    # Stage 1 configuration
    s1_backbone: str = "convnext_tiny"
    s1_steps: int = 625
    s1_batch_size: int = 8
    
    # Stage 2 configuration
    s2_steps: int = 100
    s2_batch_size: int = 1
    
    ablate_wavelet_types: List[str] = None
    ablate_J: List[int] = None
    ablate_h1: List[float] = None
    arch_list: List[str] = None
    
    # Validation and testing
    val_limit: Optional[int] = None
    test_patients: Optional[int] = None
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "loss"
    
    # Threshold selection
    use_separate_threshold_set: bool = False
    threshold_set_fraction: float = 0.15
    
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None
    threshold_samples: Optional[int] = None
    
    # Checkpointing
    save_checkpoints: bool = False
    checkpoint_dir: Optional[Path] = None
    
    # Bootstrap samples for confidence intervals
    bootstrap_samples: int = 10000
    
    def __post_init__(self):
        """Set default ablation lists if not provided."""
        if self.ablate_wavelet_types is None:
            self.ablate_wavelet_types = ["haar", "db2", "db4"]
        if self.ablate_J is None:
            self.ablate_J = [1, 2, 3]
        if self.ablate_h1 is None:
            self.ablate_h1 = [0.10, 0.25, 0.50]
        if self.arch_list is None:
            self.arch_list = ["convnext_tiny"]


PRODUCTION = ExperimentConfig(
    s1_backbone="convnext_tiny",
    s1_steps=625,
    s1_batch_size=8,
    s2_steps=100,
    s2_batch_size=1,
    ablate_wavelet_types=["haar", "db2", "db4"],
    ablate_J=[1, 2, 3],
    ablate_h1=[0.10, 0.25, 0.50],
    arch_list=["convnext_tiny"],
    bootstrap_samples=10000,
)

