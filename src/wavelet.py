"""Wavelet decomposition and persistence homology vectorization."""

import io
import math
import zipfile
from pathlib import Path
from typing import Dict, Optional

import gudhi
import numpy as np
import pywt
from PIL import Image

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

from . import config
from .utils import (
    _exists_long,
    _fingerprint,
    _load_image01,
    _open_read_long,
    _open_write_long,
)


def _npz_path_pd(
    dataset: str,
    split: str,
    image_path: str,
) -> Path:
    """Get path for cached persistence diagram pairs."""
    fid = _fingerprint(image_path)
    sub = config.VEC_ROOT / dataset / split / "pd_pairs"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{fid}.npz"


def _npz_path_wavelet(
    dataset: str,
    split: str,
    image_path: str,
    wavelet: str,
    J: int,
) -> Path:
    """Get path for cached wavelet decomposition."""
    fid = _fingerprint(image_path)
    sub = config.VEC_ROOT / dataset / split / f"wavelet_w={wavelet}_J={J}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{fid}.npz"


def _npz_path_vec(
    dataset: str,
    split: str,
    image_path: str,
    wavelet: str,
    J: int,
    h1_pct: float,
    kind: str = "vec8",
) -> Path:
    """Get path for cached vectorization file."""
    fid = _fingerprint(image_path)
    sub = (
        config.VEC_ROOT
        / dataset
        / split
        / f"{kind}_w={wavelet}_J={J}_h1={int(h1_pct*100)}"
    )
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{fid}.npz"


def _wavelet_details_at_levelJ(img01: np.ndarray, wavelet: str, J: int):
    """Extract wavelet details at level J (internal function, not cached)."""
    coeffs = pywt.wavedec2(img01, wavelet=wavelet, level=J, mode="periodization")
    LLJ = coeffs[0]
    details = coeffs[1:]
    (LHJ, HLJ, HHJ) = details[0]
    return (
        LLJ.astype(np.float32),
        LHJ.astype(np.float32),
        HLJ.astype(np.float32),
        HHJ.astype(np.float32),
    )


def _downsample_for_ph(img01: np.ndarray, max_side: int = 96) -> np.ndarray:
    """Downsample image for persistence homology computation."""
    H, W = img01.shape
    s = max(H, W)
    if s <= max_side:
        return img01
    scale = max_side / float(s)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    im = Image.fromarray((img01 * 255.0).astype(np.uint8))
    im_small = im.resize((new_w, new_h), resample=Image.BILINEAR)
    arr = np.asarray(im_small, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _compute_pd_pairs_from_img(img01_ds: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute persistence diagram pairs from a (downsampled) image.
    
    Args:
        img01_ds: Image array (should be downsampled to ~96px max side)
        
    Returns:
        Dictionary mapping dimension to persistence pairs array
    """
    cc = gudhi.CubicalComplex(
        dimensions=img01_ds.shape,
        top_dimensional_cells=img01_ds.astype(np.float64).ravel(),
    )
    cc.persistence(homology_coeff_field=2, min_persistence=0.0)
    diags = {}
    for d in (0, 1):
        intervals = cc.persistence_intervals_in_dimension(d)
        pts = [(float(b), float(de)) for (b, de) in intervals if np.isfinite(de)]
        diags[d] = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)
    return diags


def _load_pd_pairs_from_cache(dataset: str, split: str, image_path: str) -> Optional[Dict[int, np.ndarray]]:
    """Load persistence diagram pairs from cache.
    
    Args:
        dataset: Dataset name (e.g., "CBIS-DDSM")
        split: Split name (e.g., "train", "val")
        image_path: Path to image file
        
    Returns:
        Dictionary mapping dimension to persistence pairs array, or None if cache miss
    """
    p = _npz_path_pd(dataset, split, image_path)
    
    if not _exists_long(p):
        return None
    
    try:
        # Try LZ4 format first (new format)
        if LZ4_AVAILABLE:
            try:
                with _open_read_long(p, "rb") as f:
                    data = f.read()
                    decompressed = lz4.frame.decompress(data)
                    z = np.load(io.BytesIO(decompressed), allow_pickle=False)
                    diags = {
                        0: z["H0"].astype(np.float32) if "H0" in z else np.zeros((0, 2), np.float32),
                        1: z["H1"].astype(np.float32) if "H1" in z else np.zeros((0, 2), np.float32),
                    }
                    return diags
            except (lz4.frame.LZ4FrameError, ValueError, TypeError):
                # Not LZ4 compressed, try zlib/ZIP format (legacy)
                pass
        
        # Try zlib/ZIP format (legacy format from np.savez_compressed)
        with _open_read_long(p, "rb") as f:
            z = np.load(f, allow_pickle=False)
            diags = {
                0: z["H0"].astype(np.float32) if "H0" in z else np.zeros((0, 2), np.float32),
                1: z["H1"].astype(np.float32) if "H1" in z else np.zeros((0, 2), np.float32),
            }
            return diags
    except (zipfile.BadZipFile, ValueError, KeyError, AssertionError, Exception):
        # Corrupted cache file - delete it
        try:
            p.unlink()
        except Exception:
            pass
        return None


def _save_pd_pairs_to_cache(dataset: str, split: str, image_path: str, diags: Dict[int, np.ndarray]):
    """Save persistence diagram pairs to cache.
    
    Args:
        dataset: Dataset name (e.g., "CBIS-DDSM")
        split: Split name (e.g., "train", "val")
        image_path: Path to image file
        diags: Dictionary mapping dimension to persistence pairs array
    """
    p = _npz_path_pd(dataset, split, image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with _open_write_long(p, "wb") as f:
        if LZ4_AVAILABLE:
            data = {
                "H0": diags.get(0, np.zeros((0, 2), np.float32)),
                "H1": diags.get(1, np.zeros((0, 2), np.float32)),
            }
            # Save uncompressed to bytes, then compress
            buffer = io.BytesIO()
            np.savez(buffer, **data)
            compressed = lz4.frame.compress(buffer.getvalue())
            f.write(compressed)
        else:
            # Fallback to zlib compression
            np.savez_compressed(
                f,
                H0=diags.get(0, np.zeros((0, 2), np.float32)),
                H1=diags.get(1, np.zeros((0, 2), np.float32)),
            )


def _load_or_compute_pd_pairs(
    dataset: str,
    split: str,
    image_path: str,
) -> Dict[int, np.ndarray]:
    """Load or compute persistence diagram pairs with caching.
    
    This function loads the full image, downsamples it, and computes PD pairs.
    
    Args:
        dataset: Dataset name (e.g., "CBIS-DDSM")
        split: Split name (e.g., "train", "val")
        image_path: Path to image file
        
    Returns:
        Dictionary mapping dimension to persistence pairs array
    """
    # Try to load from cache
    diags = _load_pd_pairs_from_cache(dataset, split, image_path)
    if diags is not None:
        return diags
    
    # Cache miss - compute PD pairs
    img01 = _load_image01(image_path)
    img_small = _downsample_for_ph(img01, max_side=96)
    diags = _compute_pd_pairs_from_img(img_small)
    
    # Save to cache
    _save_pd_pairs_to_cache(dataset, split, image_path, diags)
    
    return diags


def _select_pairs(diags: Dict[int, np.ndarray], h1_pct: float):
    """Select persistence pairs for vectorization."""
    H0 = diags.get(0, np.zeros((0, 2), np.float32))
    H1 = diags.get(1, np.zeros((0, 2), np.float32))
    
    if H0.size:
        pers = H0[:, 1] - H0[:, 0]
        keep = np.ones(len(H0), dtype=bool)
        keep[np.argmax(pers)] = False
        H0k = H0[keep]
    else:
        H0k = H0
    
    if H1.size:
        pers1 = H1[:, 1] - H1[:, 0]
        n_keep = max(0, int(math.ceil(len(H1) * h1_pct)))
        idx = np.argsort(pers1)[:n_keep] if n_keep > 0 else []
        H1k = H1[idx] if n_keep > 0 else np.zeros((0, 2), np.float32)
    else:
        H1k = H1
    
    return H0k.astype(np.float32), H1k.astype(np.float32)


def _w_gate(psi: np.ndarray, b: float, d: float, eps: float = 1e-6):
    """Wavelet gate function (single pair version)."""
    m = (psi >= b) & (psi < d)
    out = np.zeros_like(psi, dtype=np.float32)
    if m.any():
        out[m] = ((psi[m] - b) * (d - psi[m])) / ((d - b) + eps)
    return out


def _w_gate_vectorized_impl(psi: np.ndarray, pairs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Vectorized wavelet gate function for multiple pairs (implementation).
    
    Args:
        psi: Input array [H, W]
        pairs: Persistence pairs [N, 2] where each row is (b, d)
        eps: Small epsilon for numerical stability
        
    Returns:
        Accumulated gate function output [H, W]
    """
    if pairs.size == 0 or len(pairs) == 0:
        return np.zeros_like(psi, dtype=np.float32)
    
    # Filter out invalid pairs (d <= b)
    valid_mask = pairs[:, 1] > pairs[:, 0]
    if not valid_mask.any():
        return np.zeros_like(psi, dtype=np.float32)
    
    valid_pairs = pairs[valid_mask]  # [N_valid, 2]
    b_vals = valid_pairs[:, 0]  # [N_valid]
    d_vals = valid_pairs[:, 1]  # [N_valid]
    
    # Broadcast: psi is [H, W], b_vals and d_vals are [N_valid]
    # We want to compute gate for each pair and sum
    psi_expanded = psi[:, :, np.newaxis]  # [H, W, 1]
    b_expanded = b_vals[np.newaxis, np.newaxis, :]  # [1, 1, N_valid]
    d_expanded = d_vals[np.newaxis, np.newaxis, :]  # [1, 1, N_valid]
    
    # Compute mask for each pair: (psi >= b) & (psi < d)
    mask = (psi_expanded >= b_expanded) & (psi_expanded < d_expanded)  # [H, W, N_valid]
    
    # Compute gate function for each pair
    diff = d_expanded - b_expanded  # [1, 1, N_valid]
    numerator = (psi_expanded - b_expanded) * (d_expanded - psi_expanded)  # [H, W, N_valid]
    denominator = diff + eps  # [1, 1, N_valid]
    gates = np.where(mask, numerator / denominator, 0.0)  # [H, W, N_valid]
    
    # Sum over all pairs
    result = gates.sum(axis=2)  # [H, W]
    
    # Normalize by number of pairs
    n_pairs = len(valid_pairs)
    if n_pairs > 0:
        result = result / n_pairs
    
    return result.astype(np.float32)


# Try to JIT compile with Numba if available
if NUMBA_AVAILABLE:
    try:
        _w_gate_vectorized = numba.jit(nopython=False, cache=True)(_w_gate_vectorized_impl)
    except Exception:
        # If JIT fails, use regular function
        _w_gate_vectorized = _w_gate_vectorized_impl
else:
    _w_gate_vectorized = _w_gate_vectorized_impl


def _norm01_impl(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range (implementation)."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


# Try to JIT compile with Numba if available
if NUMBA_AVAILABLE:
    try:
        _norm01 = numba.jit(nopython=False, cache=True)(_norm01_impl)
    except Exception:
        _norm01 = _norm01_impl
else:
    _norm01 = _norm01_impl


def _vec_maps_wavelet6(
    img01: np.ndarray, wavelet: str, J: int, H0: np.ndarray, H1: np.ndarray
):
    """Generate 6-channel wavelet vectorization maps (vectorized version).
    
    Args:
        img01: Image array
        wavelet: Wavelet type
        J: Decomposition level
        H0: H0 persistence pairs
        H1: H1 persistence pairs
    """
    # Compute wavelet decomposition on-the-fly (fast operation)
    _, LH, HL, HH = _wavelet_details_at_levelJ(img01, wavelet, J)
    
    subbands = [_norm01(LH), _norm01(HL), _norm01(HH)]
    out = []
    for pairs in (H0, H1):
        if pairs.size == 0 or len(pairs) == 0:
            out.extend([np.zeros_like(s, dtype=np.float32) for s in subbands])
        else:
            # Vectorized: process all pairs at once for each subband
            accs = [_w_gate_vectorized(s, pairs) for s in subbands]
            out.extend(accs)
    return np.stack(out, axis=0)  # [6,h,w]


def _vec_maps_baseline2(img01: np.ndarray, H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    """Generate 2-channel baseline vectorization maps from original image (vectorized version)."""
    psi = _norm01(img01)
    out = []
    for pairs in (H0, H1):
        if pairs.size == 0 or len(pairs) == 0:
            out.append(np.zeros_like(psi, dtype=np.float32))
        else:
            # Vectorized: process all pairs at once
            acc = _w_gate_vectorized(psi, pairs)
            out.append(acc)
    return np.stack(out, axis=0)  # [2,h,w] => [H0_baseline, H1_baseline]


def _load_vectorization_from_cache(
    dataset: str,
    split: str,
    image_path: str,
    wavelet: str,
    J: int,
    h1_pct: float,
) -> Optional[np.ndarray]:
    """Load vectorization maps from cache.
    
    Vectorizations in cache are normalized to [0, 1] per channel.
    
    Args:
        dataset: Dataset name
        split: Split name
        image_path: Path to image file
        wavelet: Wavelet type
        J: Wavelet decomposition level
        h1_pct: H1 persistence pair keep percentage
        
    Returns:
        Vectorization maps [8, H, W] normalized to [0, 1] per channel, or None if not cached
    """
    p = _npz_path_vec(dataset, split, image_path, wavelet, J, h1_pct, kind="vec8")
    
    if not _exists_long(p):
        return None
    
    try:
        with _open_read_long(p, "rb") as f:
            z = np.load(f, allow_pickle=False)
            if "vec8" in z:
                maps8 = z["vec8"].astype(np.float32)
                # Ensure normalized (legacy cache might not be normalized)
                # Check if already normalized (all values in [0, 1])
                if maps8.min() >= 0 and maps8.max() <= 1:
                    return maps8
                else:
                    # Normalize on load (for legacy cache) - VECTORIZED
                    maps8_norm = np.zeros_like(maps8, dtype=np.float32)
                    mn = maps8.min(axis=(1, 2), keepdims=True)
                    mx = maps8.max(axis=(1, 2), keepdims=True)
                    mask = (mx > mn).squeeze()
                    if mask.any():
                        diff = mx - mn
                        maps8_norm[mask] = (maps8[mask] - mn[mask]) / (diff[mask] + 1e-12)
                    return maps8_norm
            else:
                # Try legacy format
                if "maps8" in z:
                    maps8 = z["maps8"].astype(np.float32)
                    # Normalize legacy cache - VECTORIZED
                    maps8_norm = np.zeros_like(maps8, dtype=np.float32)
                    mn = maps8.min(axis=(1, 2), keepdims=True)
                    mx = maps8.max(axis=(1, 2), keepdims=True)
                    mask = (mx > mn).squeeze()
                    if mask.any():
                        diff = mx - mn
                        maps8_norm[mask] = (maps8[mask] - mn[mask]) / (diff[mask] + 1e-12)
                    return maps8_norm
                return None
    except Exception:
        return None


def build_vec8(
    dataset: str,
    split: str,
    image_path: str,
    wavelet: str = "haar",
    J: int = 1,
    h1_pct: float = 0.25,
    use_cache: bool = True,
) -> np.ndarray:
    """Build 8-channel vectorization maps from cached persistence diagrams.
    
    This function first checks for cached vectorizations. If not found, it computes
    vectorizations on-the-fly using cached PD pairs and optionally caches the result.
    
    **IMPORTANT**: For mathematical consistency, all computations (PD, wavelets, gates)
    are performed on the same downsampled image (96px max side). This ensures that:
    1. PD pairs and wavelets are from the same image representation
    2. Gate functions compare values at the same scale
    3. Memory usage is minimized (2524x reduction)
    
    The output maps are small (e.g., 48×30 for J=1, haar) and will be interpolated
    to full resolution in the dataset loader if needed.
    
    Args:
        dataset: Dataset name (e.g., "CBIS-DDSM")
        split: Split name (e.g., "train", "val")
        image_path: Path to image file
        wavelet: Wavelet type (default: "haar")
        J: Wavelet decomposition level (default: 1)
        h1_pct: H1 persistence pair keep percentage (default: 0.25)
        use_cache: Whether to use cached vectorizations (default: True)
        
    Returns:
        8-channel vectorization maps [8, H, W] where H and W are small (e.g., 48×30)
    """
    # Try to load from cache first
    if use_cache:
        maps8_cached = _load_vectorization_from_cache(
            dataset, split, image_path, wavelet, J, h1_pct
        )
        if maps8_cached is not None:
            return maps8_cached
    
    # Load full image
    img01_full = _load_image01(image_path)
    
    # Downsample FIRST to 96px max side (for mathematical consistency)
    img01_ds = _downsample_for_ph(img01_full, max_side=96)
    
    # Load or compute PD pairs (cached - computed on downsampled image)
    diags = _load_pd_pairs_from_cache(dataset, split, image_path)
    if diags is None:
        # Cache miss - compute from downsampled image
        diags = _compute_pd_pairs_from_img(img01_ds)
        # Save to cache
        _save_pd_pairs_to_cache(dataset, split, image_path, diags)
    
    # Select persistence pairs
    H0, H1 = _select_pairs(diags, h1_pct=h1_pct)
    
    # Compute wavelets on DOWNSAMPLED image (mathematical consistency)
    LLJ, LHJ, HLJ, HHJ = _wavelet_details_at_levelJ(img01_ds, wavelet, J)
    
    # Compute 6-channel wavelet vectorization maps (on downsampled wavelets)
    maps6 = _vec_maps_wavelet6(img01_ds, wavelet, J, H0, H1).astype(np.float32)
    
    # For baseline maps, we need to match the resolution of the wavelet subbands
    # The wavelet subbands are at resolution (h_sub, w_sub) = LLJ.shape
    # So we downsample the image to match this resolution for consistency
    h_sub, w_sub = LLJ.shape
    if (h_sub, w_sub) != img01_ds.shape:
        # Downsample to match wavelet subband resolution
        im_ds = Image.fromarray((img01_ds * 255.0).astype(np.uint8))
        im_sub = im_ds.resize((w_sub, h_sub), resample=Image.BILINEAR)
        img01_sub = np.asarray(im_sub, dtype=np.float32) / 255.0
        img01_sub = np.clip(img01_sub, 0.0, 1.0)
    else:
        img01_sub = img01_ds
    
    # Compute 2-channel baseline vectorization maps (at same resolution as wavelet subbands)
    base2 = _vec_maps_baseline2(img01_sub, H0, H1).astype(np.float32)
    
    # Verify dimensions match
    assert maps6.shape[1:] == base2.shape[1:], (
        f"Dimension mismatch: maps6.shape={maps6.shape}, base2.shape={base2.shape}"
    )
    
    # Concatenate to get 8-channel maps
    maps8 = np.concatenate([maps6, base2], axis=0)  # [8, h_sub, w_sub]
    
    # Normalize each channel to [0, 1] for consistent scaling (VECTORIZED)
    # This is important because:
    # 1. Gate function outputs can have varying ranges depending on persistence pairs
    # 2. Neural networks benefit from consistent input scaling
    # 3. Each channel represents different topological information, so independent normalization is appropriate
    maps8_norm = np.zeros_like(maps8, dtype=np.float32)
    mn = maps8.min(axis=(1, 2), keepdims=True)  # [8, 1, 1]
    mx = maps8.max(axis=(1, 2), keepdims=True)  # [8, 1, 1]
    mask = (mx > mn).squeeze()  # [8] boolean mask
    if mask.any():
        # Only normalize channels where max > min
        diff = mx - mn  # [8, 1, 1]
        maps8_norm[mask] = (maps8[mask] - mn[mask]) / (diff[mask] + 1e-12)
    
    return maps8_norm


# Backward compatibility alias
def build_or_load_vec8(*args, **kwargs):
    """Deprecated: Use build_vec8 instead. This alias is kept for backward compatibility."""
    return build_vec8(*args, **kwargs)

