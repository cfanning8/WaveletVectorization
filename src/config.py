"""Configuration and path settings for the mammography pipeline."""

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 14

rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

BASE_DIR = Path(__file__).parent.parent.resolve()

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VEC_ROOT = PROCESSED_DIR / "wavelet_vectorization"

CBIS_ROOT = RAW_DIR / "CBIS-DDSM"

INBREAST_ROOT_PARENT = RAW_DIR / "INbreast"
CMMD_ROOT_PARENT = RAW_DIR / "TheChineseMammographyDatabase"

RESULTS_ROOT = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "notebooks" / "figures"

for d in [VEC_ROOT, RESULTS_ROOT, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(0)
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

PATCH = 512
STRIDE = 384
KEEP_FRAC = 0.25
PATCH = 512
STRIDE = 384
KEEP_FRAC = 0.25

