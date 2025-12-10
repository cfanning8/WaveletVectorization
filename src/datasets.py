"""PyTorch datasets for mammography pipeline."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import config
from .utils import _load_image01, _win_long
from .wavelet import build_or_load_vec8

try:
    import pydicom
except ImportError:
    pydicom = None


def _otsu_thresh_01(img01: np.ndarray) -> float:
    """Compute Otsu threshold for image."""
    hist, _ = np.histogram((img01 * 255).astype(np.uint8), bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    p /= p.sum() + 1e-12
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    mu0 = np.divide(mu, w0 + 1e-12)
    mu1 = np.divide(mu_t - mu, w1 + 1e-12)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    t = int(np.nanargmax(sigma_b))
    return t / 255.0


def extract_patches(img01: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Extract patches from image using Otsu thresholding.
    
    Args:
        img01: Normalized image in [0, 1] range
        
    Returns:
        List of (y0, y1, x0, x1) patch coordinates
    """
    H, W = img01.shape
    thr = _otsu_thresh_01(img01)
    mask = (img01 >= thr).astype(np.uint8)
    out = []
    for y0 in range(0, max(1, H - config.PATCH + 1), config.STRIDE):
        for x0 in range(0, max(1, W - config.PATCH + 1), config.STRIDE):
            y1, x1 = y0 + config.PATCH, x0 + config.PATCH
            if y1 > H or x1 > W:
                continue
            if mask[y0:y1, x0:x1].mean() >= config.KEEP_FRAC:
                out.append((y0, y1, x0, x1))
    assert len(out) > 0, "No valid patches after Otsu+coverage; image likely tiny or blank."
    return out


class MILBagsCBIS(Dataset):
    """Stage-1 MIL dataset for CBIS-DDSM."""

    def __init__(
        self,
        df: pd.DataFrame,
        split_tag: str,
        wavelet: str = "haar",
        J: int = 1,
        h1_pct: float = 0.25,
    ):
        self.df = df.reset_index(drop=True)
        self.split_tag = split_tag
        self.wavelet = wavelet
        self.J = int(J)
        self.h1_pct = float(h1_pct)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = _load_image01(r["image_path"])
        patches = extract_patches(img)
        
        P = len(patches)
        img_tensor = torch.from_numpy(img).float()
        bag = torch.zeros(P, 1, config.PATCH, config.PATCH, dtype=torch.float32)
        for i, (y0, y1, x0, x1) in enumerate(patches):
            patch = img_tensor[y0:y1, x0:x1]
            if patch.shape != (config.PATCH, config.PATCH):
                patch_t = patch.unsqueeze(0).unsqueeze(0)
                patch_resized = F.interpolate(
                    patch_t, size=(config.PATCH, config.PATCH), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
                bag[i, 0] = patch_resized
            else:
                bag[i, 0] = patch
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        meta = {
            "patient_id": str(r["patient_id"]),
            "image_path": r["image_path"],
            "side": r.get("side", None),
            "view": r.get("view", None),
        }
        return {"bag": bag, "y": y, "meta": meta}


def collate_mil(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for MIL batches."""
    B = len(batch)
    Pmax = max(b["bag"].shape[0] for b in batch)
    C, H, W = batch[0]["bag"].shape[1:]
    bags = torch.zeros(B, Pmax, C, H, W, dtype=torch.float32)
    ys = torch.zeros(B, dtype=torch.float32)
    metas = []
    for i, b in enumerate(batch):
        P = b["bag"].shape[0]
        bags[i, :P] = b["bag"]
        ys[i] = b["y"]
        metas.append(b["meta"])
    return {"bags": bags, "y": ys, "metas": metas}


class CBISDet(Dataset):
    """Stage-2 detection dataset for CBIS-DDSM.
    
    If use_topo=True: returns gray+topo8 as 9-channel input for detector
    (FiLM will consume topo), and also enables RoI-topo branch downstream.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split_tag: str,
        wavelet: str = "haar",
        J: int = 1,
        h1_pct: float = 0.25,
        use_topo: bool = True,
        require_roi_masks: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.split_tag = split_tag
        self.wavelet = wavelet
        self.J = int(J)
        self.h1_pct = float(h1_pct)
        self.use_topo = bool(use_topo)
        self.require_roi_masks = bool(require_roi_masks)
        
        self._topo_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img01 = _load_image01(r["image_path"])
        g = torch.from_numpy(img01).unsqueeze(0)  # [1,H,W]
        
        if self.use_topo:
            H, W = img01.shape
            cache_key = (r["image_path"], self.wavelet, self.J, self.h1_pct, H, W)
            
            if cache_key in self._topo_cache:
                maps8_t = self._topo_cache[cache_key]
            else:
                maps8 = build_or_load_vec8(
                    dataset="CBIS-DDSM",
                    split=self.split_tag,
                    image_path=r["image_path"],
                    wavelet=self.wavelet,
                    J=self.J,
                    h1_pct=self.h1_pct,
                )
                maps8_t = torch.from_numpy(maps8)
                if maps8_t.shape[-2:] != (H, W):
                    maps8_t = F.interpolate(
                        maps8_t.unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                maps8_t = maps8_t.cpu().pin_memory()
                self._topo_cache[cache_key] = maps8_t
            
            maps8_t = self._topo_cache[cache_key]
            
            x = torch.cat([g, maps8_t], dim=0)
        else:
            x = g

        boxes_list = []
        mask_list = []
        if "roi_mask_path" in self.df.columns:
            v = r["roi_mask_path"]
            if isinstance(v, (list, tuple)):
                mask_list = [p for p in v if isinstance(p, str)]
        
        if len(mask_list) == 0:
            if self.require_roi_masks:
                raise ValueError(
                    f"No ROI masks found for image {r['image_path']} (patient_id: {r['patient_id']}). "
                    "Stage-2 training requires ROI masks. Check data loading or skip this image."
                )
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor(idx, dtype=torch.int64),
                }
        else:
            # Process ROI masks
            for p in mask_list:
                sp = _win_long(p)
                ds = pydicom.dcmread(sp, force=True)
                m = ds.pixel_array
                if m.dtype != np.uint8:
                    m = (m > 0).astype(np.uint8)
                ys, xs = np.where(m > 0)
                if ys.size == 0:
                    continue
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                boxes_list.append([x0, y0, x1, y1])
            
            if len(boxes_list) == 0:
                if self.require_roi_masks:
                    raise ValueError(
                        f"No valid boxes extracted from ROI masks for image {r['image_path']} "
                        f"(patient_id: {r['patient_id']}). All masks are empty. Check ROI mask data."
                    )
                else:
                    boxes = torch.empty((0, 4), dtype=torch.float32)
                    labels = torch.empty((0,), dtype=torch.int64)
                    target = {
                        "boxes": boxes,
                        "labels": labels,
                        "image_id": torch.tensor(idx, dtype=torch.int64),
                    }
            else:
                boxes = torch.tensor(boxes_list, dtype=torch.float32)
                labels = torch.zeros((boxes.shape[0],), dtype=torch.int64) + 1  # lesion class=1
                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor(idx, dtype=torch.int64),
                }
        meta = {
            "patient_id": str(r["patient_id"]),
            "label": int(r["label"]),
            "side": r.get("side", None),
            "view": r.get("view", None),
            "image_path": r["image_path"],
        }
        return (x, target, meta)


def collate_det(batch: List[Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collate function for detection batches."""
    imgs = [b[0] for b in batch]
    targs = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return imgs, targs, metas

