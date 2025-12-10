"""Evaluation functions for mammography pipeline."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config
from .datasets import CBISDet, MILBagsCBIS, collate_det, collate_mil
from .models import FasterRCNNWithAttnMIL, Stage1MIL
from torch.utils.data import DataLoader


def aggregate_patient_scores(
    df_img: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate image-level scores to patient-level.
    
    Args:
        df_img: DataFrame with image-level predictions
        
    Returns:
        Tuple of (y, p, pids) where y=labels, p=probabilities, pids=patient_ids
    """
    if "box_prob" in df_img.columns:
        gb = df_img.groupby(["patient_id"], as_index=False).agg(
            patient_prob=("box_prob", "max"), patient_label=("label", "max")
        )
    else:
        gb = df_img.groupby(["patient_id"], as_index=False).agg(
            patient_prob=("image_prob", "max"), patient_label=("label", "max")
        )
    y = gb["patient_label"].astype(int).to_numpy()
    p = gb["patient_prob"].astype(float).to_numpy()
    pids = gb["patient_id"].astype(str).to_numpy()
    return y, p, pids


def youdens_threshold(y_true, y_prob):
    """Compute Youden's J statistic threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i])


def youdens_threshold_from_curve(fpr, tpr, thresholds):
    """Compute Youden's J statistic threshold from pre-computed ROC curve."""
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thresholds[i])


def thr_for_spec(y_true, y_prob, spec_target=0.95):
    """Find threshold for target specificity."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    m = spec >= spec_target
    i = int(np.argmax(tpr[m])) if np.any(m) else int(np.argmax(spec))
    return float(thr[m][i]) if np.any(m) else float(thr[i])


def thr_for_spec_from_curve(fpr, tpr, thresholds, spec_target=0.95):
    """Find threshold for target specificity from pre-computed ROC curve."""
    spec = 1.0 - fpr
    m = spec >= spec_target
    if np.any(m):
        i = int(np.argmax(tpr[m]))
        return float(thresholds[m][i])
    else:
        i = int(np.argmax(spec))
        return float(thresholds[i])


def thr_for_sens(y_true, y_prob, sens_target=0.95):
    """Find threshold for target sensitivity."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    m = tpr >= sens_target
    i = int(np.argmax(m)) if np.any(m) else int(np.argmax(tpr))
    return float(thr[i])


def thr_for_sens_from_curve(fpr, tpr, thresholds, sens_target=0.95):
    """Find threshold for target sensitivity from pre-computed ROC curve."""
    m = tpr >= sens_target
    if np.any(m):
        i = int(np.argmax(m))
        return float(thresholds[i])
    else:
        i = int(np.argmax(tpr))
        return float(thresholds[i])


def metrics_from_thr(y, p, thr):
    """Compute metrics at given threshold."""
    yhat = (p >= thr).astype(int)
    sens = recall_score(y, yhat, zero_division=0)
    spec = (yhat[y == 0] == 0).mean() if (y == 0).any() else np.nan
    prec = precision_score(y, yhat, zero_division=0)
    acc = accuracy_score(y, yhat)
    auc = roc_auc_score(y, p) if len(np.unique(y)) >= 2 else np.nan
    return dict(Precision=prec, Sensitivity=sens, Specificity=spec, Accuracy=acc, AUC=auc)


def spec_at_95sens(y, p, thr_sens95):
    """Compute specificity at 95% sensitivity threshold."""
    yhat = (p >= thr_sens95).astype(int)
    if (y == 0).any():
        return float((yhat[y == 0] == 0).mean())
    return np.nan


def sens_at_95spec(y, p, thr_spec95):
    """Compute sensitivity at 95% specificity threshold."""
    yhat = (p >= thr_spec95).astype(int)
    if (y == 1).any():
        return float((yhat[y == 1] == 1).mean())
    return np.nan


def bootstrap_ci(metric_fn, y, p, B=10, seed=None):
    """Compute bootstrap confidence interval for a metric."""
    if seed is None:
        seed = config.SEED
    rng = np.random.default_rng(seed)
    vals = []
    N = len(y)
    for _ in range(B):
        idx = rng.choice(np.arange(N), size=N, replace=True)
        try:
            vals.append(float(metric_fn(y[idx], p[idx])))
        except Exception:
            vals.append(float("nan"))
    arr = np.array(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def metrics_with_ci(y, p, thr, thr_spec95, thr_sens95, B=10, seed=None):
    """Compute metrics with confidence intervals."""
    if seed is None:
        seed = config.SEED
    base = metrics_from_thr(y, p, thr)
    out = {}
    for k in ["Precision", "Sensitivity", "Specificity", "Accuracy", "AUC"]:
        if k == "AUC":
            fn = lambda y_, p_: roc_auc_score(y_, p_) if len(np.unique(y_)) >= 2 else float("nan")
        else:
            fn = lambda y_, p_, kk=k: metrics_from_thr(y_, p_, thr)[kk]
        lo, hi = bootstrap_ci(fn, y, p, B=B, seed=seed)
        out[k] = base[k]
        out[k + "_CI"] = [lo, hi]
    
    lo, hi = bootstrap_ci(
        lambda y_, p_: spec_at_95sens(y_, p_, thr_sens95), y, p, B=B, seed=seed
    )
    out["SPEC_at_95SENS"] = spec_at_95sens(y, p, thr_sens95)
    out["SPEC_at_95SENS_CI"] = [lo, hi]
    
    lo, hi = bootstrap_ci(
        lambda y_, p_: sens_at_95spec(y_, p_, thr_spec95), y, p, B=B, seed=seed
    )
    out["SENS_at_95SPEC"] = sens_at_95spec(y, p, thr_spec95)
    out["SENS_at_95SPEC_CI"] = [lo, hi]
    
    return out


@torch.no_grad()
def infer_stage1_on_df(
    model: Stage1MIL, df: pd.DataFrame, batch_size: int = 1
) -> pd.DataFrame:
    """Run Stage-1 inference on dataframe.
    
    Args:
        model: Trained Stage-1 model
        df: DataFrame with image paths
        batch_size: Batch size for inference
        
    Returns:
        DataFrame with predictions
    """
    ds = MILBagsCBIS(df, split_tag="eval-no-cache")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mil,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        prefetch_factor=None,  # Disable prefetching
        persistent_workers=False,
    )
    image_path_to_label = dict(zip(df["image_path"], df["label"]))
    
    rows = []
    model.eval()
    for b in loader:
        bags = b["bags"].to(config.DEVICE, non_blocking=False)
        metas = b["metas"]
        logit, _, _ = model(bags)
        probs = torch.sigmoid(logit).cpu().numpy()
        for i, m in enumerate(metas):
            label = image_path_to_label.get(m["image_path"], 0)
            rows.append(
                dict(
                    patient_id=m["patient_id"],
                    label=label,
                    image_path=m["image_path"],
                    image_prob=float(probs[i]),
                )
            )
        # Clear GPU tensors immediately
        del bags, logit, probs
        # Aggressive cleanup every batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


@torch.no_grad()
def infer_stage2_on_df(
    det_mil: FasterRCNNWithAttnMIL,
    df: pd.DataFrame,
    wavelet="haar",
    J=1,
    h1_pct=0.25,
    use_topo: bool = True,
) -> pd.DataFrame:
    """Run Stage-2 inference on dataframe.
    
    Args:
        det_mil: Trained Stage-2 model
        df: DataFrame with image paths
        wavelet: Wavelet type
        J: Wavelet decomposition level
        h1_pct: H1 persistence pair keep percentage
        use_topo: Whether to use topology features
        
    Returns:
        DataFrame with predictions
    """
    # Ensure model is on GPU
    det_mil = det_mil.to(config.DEVICE)
    det_mil.eval()
    
    ds = CBISDet(
        df,
        split_tag="eval",
        wavelet=wavelet,
        J=J,
        h1_pct=h1_pct,
        use_topo=use_topo,
        require_roi_masks=False,  # For inference, ROI masks are not required
    )
    # Clear topology cache periodically to prevent memory accumulation
    # With J=2, topology maps are larger, so cache can grow very large
    max_cache_size = 50  # Limit cache to 50 entries
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_det,
        pin_memory=True,
        prefetch_factor=None,
        persistent_workers=False,
    )
    rows = []
    import gc
    from tqdm.auto import tqdm
    
    # Add progress bar for inference
    total_batches = len(loader)
    
    if total_batches > 0:
        print("  Warming up GPU for inference...", flush=True)
        try:
            dummy_iter = iter(loader)
            dummy_batch = next(dummy_iter)
            dummy_imgs, _, _ = dummy_batch
            dummy_imgs_gpu = [x.to(config.DEVICE, non_blocking=True) for x in dummy_imgs]
            with torch.cuda.amp.autocast():
                _ = det_mil(dummy_imgs_gpu)
            torch.cuda.synchronize()
            del dummy_imgs_gpu, dummy_imgs
            torch.cuda.empty_cache()
            print("  GPU warmup complete!", flush=True)
        except Exception as e:
            print(f"  GPU warmup failed (non-critical): {e}", flush=True)
    
    loader_with_progress = tqdm(enumerate(loader), total=total_batches, desc="Inferencing", unit="batch")
    
    for batch_idx, (imgs, _, metas) in loader_with_progress:
        imgs = [x.to(config.DEVICE, non_blocking=True) for x in imgs]
        with torch.cuda.amp.autocast():
            _, img_probs = det_mil(imgs)
        p_img = float(img_probs.squeeze(0).detach().cpu().numpy())
        m = metas[0]
        rows.append(
            dict(
                patient_id=m["patient_id"],
                label=int(m["label"]),
                image_path=m["image_path"],
                image_prob=p_img,
            )
        )
        # Clear GPU tensors immediately
        del imgs, img_probs
        # Aggressive cleanup every batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Periodic garbage collection and cache clearing every 100 batches to prevent memory accumulation
        if (batch_idx + 1) % 100 == 0:
            if hasattr(ds, '_topo_cache') and len(ds._topo_cache) > max_cache_size:
                keys_to_remove = list(ds._topo_cache.keys())[:-max_cache_size]
                for key in keys_to_remove:
                    del ds._topo_cache[key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # Final cleanup - clear topology cache completely
    if hasattr(ds, '_topo_cache'):
        ds._topo_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def pick_thresholds(
    det_mil: FasterRCNNWithAttnMIL,
    df_threshold: pd.DataFrame,
    wavelet: str,
    J: int,
    h1: float,
    use_topo: bool,
) -> Dict[str, float]:
    """Pick thresholds on threshold selection set.
    
    Args:
        det_mil: Trained Stage-2 model
        df_threshold: Threshold selection dataframe
        wavelet: Wavelet type
        J: Wavelet decomposition level
        h1: H1 persistence pair keep percentage
        use_topo: Whether to use topology features
        
    Returns:
        Dictionary with threshold values
        
    Raises:
        ValueError: If threshold set has only one class
    """
    df_thresh_det = infer_stage2_on_df(
        det_mil, df_threshold, wavelet=wavelet, J=J, h1_pct=h1, use_topo=use_topo
    )
    y_t, p_t, _ = aggregate_patient_scores(df_thresh_det)
    
    if len(np.unique(y_t)) < 2:
        raise ValueError(
            f"Threshold set has only one class (classes: {np.unique(y_t)}). "
            "Cannot select thresholds. Check data split or use fixed threshold."
        )
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_t, p_t)
    
    # Compute thresholds from pre-computed ROC curve
    thr_J = youdens_threshold_from_curve(fpr, tpr, thresholds)
    thr_spec95 = thr_for_spec_from_curve(fpr, tpr, thresholds, spec_target=0.95)
    thr_sens95 = thr_for_sens_from_curve(fpr, tpr, thresholds, sens_target=0.95)
    return {"thr_J": thr_J, "thr_spec95": thr_spec95, "thr_sens95": thr_sens95}


def pick_thresholds_on_val(
    det_mil: FasterRCNNWithAttnMIL,
    df_val: pd.DataFrame,
    wavelet,
    J,
    h1,
    use_topo,
) -> Dict[str, float]:
    """Pick thresholds on validation set (legacy function for backward compatibility).
    
    DEPRECATED: Use pick_thresholds() with separate threshold set instead.
    
    Args:
        det_mil: Trained Stage-2 model
        df_val: Validation dataframe
        wavelet: Wavelet type
        J: Wavelet decomposition level
        h1: H1 persistence pair keep percentage
        use_topo: Whether to use topology features
        
    Returns:
        Dictionary with threshold values
        
    Raises:
        ValueError: If validation set has only one class
    """
    df_val_det = infer_stage2_on_df(
        det_mil, df_val, wavelet=wavelet, J=J, h1_pct=h1, use_topo=use_topo
    )
    y_v, p_v, _ = aggregate_patient_scores(df_val_det)
    
    if len(np.unique(y_v)) < 2:
        unique_classes = np.unique(y_v).tolist()
        print(f"⚠️  WARNING: Validation set has only one class (classes: {unique_classes}). "
              "Cannot select optimal thresholds. Using fallback thresholds (0.5 for all).")
        # Return fallback thresholds instead of crashing
        return {
            "thr_J": 0.5,
            "thr_spec95": 0.5,
            "thr_sens95": 0.5,
            "warning": f"Single class validation set: {unique_classes}. Using fallback thresholds."
        }
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_v, p_v)
    
    # Compute thresholds from pre-computed ROC curve
    thr_J = youdens_threshold_from_curve(fpr, tpr, thresholds)
    thr_spec95 = thr_for_spec_from_curve(fpr, tpr, thresholds, spec_target=0.95)
    thr_sens95 = thr_for_sens_from_curve(fpr, tpr, thresholds, sens_target=0.95)
    return {"thr_J": thr_J, "thr_spec95": thr_spec95, "thr_sens95": thr_sens95}

