"""Utility functions for file I/O, hashing, and image loading."""

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None


def _win_long(p: str) -> str:
    """Windows long-path helper."""
    if not os.name == "nt":
        return p
    p = os.path.normpath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\\\")
    return "\\\\?\\" + p


def _sha1(x: str) -> str:
    """Compute SHA1 hash of a string."""
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def _exists_long(path: Path | str) -> bool:
    """Check if path exists (handles Windows long paths)."""
    s = _win_long(str(path))
    return os.path.exists(s)


def _listdir_long(path: Path | str) -> List[str]:
    """List directory contents (handles Windows long paths)."""
    s = _win_long(str(path))
    return os.listdir(s)


def _open_read_long(path: Path | str, mode="rb"):
    """Open file for reading (handles Windows long paths)."""
    return open(_win_long(str(path)), mode)


def _open_write_long(path: Path | str, mode="wb"):
    """Open file for writing (handles Windows long paths)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(_win_long(str(path)), mode)


def _load_image01(path: str) -> np.ndarray:
    """Load image and normalize to [0, 1] range.
    
    Args:
        path: Path to image file (DICOM or standard image format)
        
    Returns:
        Normalized image array in [0, 1] range
    """
    sp = _win_long(path)
    if path.lower().endswith(".dcm"):
        if pydicom is None:
            raise RuntimeError("pydicom not installed but DICOM requested.")
        ds = pydicom.dcmread(sp, force=True)
        arr = ds.pixel_array.astype(np.float32)
    else:
        with _open_read_long(sp, "rb") as f:
            img = Image.open(f).convert("L")
            arr = np.asarray(img, dtype=np.float32)
    
    mn, mx = float(arr.min()), float(arr.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        raise ValueError(f"Bad image dynamic range for {path}: min={mn}, max={mx}")
    
    out = (arr - mn) / (mx - mn)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fingerprint(path: str) -> str:
    """Generate fingerprint for a file path (for caching)."""
    st = os.stat(_win_long(os.path.abspath(path)))
    payload = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    return _sha1(payload)

