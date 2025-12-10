"""Data loading functions for CBIS-DDSM, INbreast, and CMMD datasets."""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Enable tqdm for pandas
tqdm.pandas(desc="progress")

from . import config
from .utils import (
    _exists_long,
    _listdir_long,
    _open_read_long,
    _win_long,
)

try:
    import pydicom
except ImportError:
    pydicom = None


def _norm_side_token(s) -> Optional[str]:
    """Normalize side token."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper()
    return {"L": "LEFT", "LEFT": "LEFT", "R": "RIGHT", "RIGHT": "RIGHT"}.get(t, t)


def _norm_view_token(v) -> Optional[str]:
    """Normalize view token."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).strip().upper()
    if t in {"LM", "ML"}:
        return "MLO"
    if t.startswith("XCC"):
        return "CC"
    if t in {"CC", "MLO", "FB"}:
        return t
    return t


_CBIS_KEEP = [
    "patient_id",
    "left or right breast",
    "image view",
    "pathology",
    "abnormality type",
    "image file path",
    "ROI mask file path",
    "breast density",
    "Breast Density",
    "BREAST DENSITY",
]


def _cbis_manifest_root(cbis_root: Path) -> Path:
    """Find CBIS manifest root directory."""
    cand = sorted(cbis_root.glob("manifest-*"))
    assert len(cand) > 0, f"No manifest-* under {cbis_root}"
    return cand[0]


def _resolve_uncropped_dcm_11_only(base: Path, rel_csv_path: str) -> Optional[str]:
    """Resolve CBIS DICOM image path."""
    rel = str(rel_csv_path).replace("\\", "/").strip()
    folder_rel = rel.rsplit("/", 1)[0] if "/" in rel else rel
    folder = (base / folder_rel).resolve()
    assert _exists_long(folder), f"CBIS image folder missing: {folder}"
    p11 = folder / "1-1.dcm"
    assert _exists_long(p11), f"CBIS image 1-1.dcm missing: {p11}"
    return _win_long(str(p11))


def _resolve_roi_mask_11_only(base: Path, rel_csv_roi_path: str) -> Optional[str]:
    """Resolve CBIS ROI mask path."""
    if pd.isna(rel_csv_roi_path):
        return None
    rel = str(rel_csv_roi_path).replace("\\", "/").strip()
    folder_rel = rel.rsplit("/", 1)[0] if "/" in rel else rel
    folder = (base / folder_rel).resolve()
    assert _exists_long(folder), f"CBIS ROI folder missing: {folder}"
    p11 = folder / "1-1.dcm"
    assert _exists_long(p11), f"CBIS ROI 1-1.dcm missing: {p11}"
    return _win_long(str(p11))


def load_cbis_ddsm(manifest_root: Path) -> pd.DataFrame:
    """Load CBIS-DDSM dataset."""
    print("Loading CBIS-DDSM (with progress)...")
    manifest = manifest_root.resolve()
    base = (manifest / "CBIS-DDSM").resolve()
    assert _exists_long(base), f"Expected CBIS-DDSM base missing: {base}"
    
    csvs = (
        list(manifest.glob("mass_case_description_*_set.csv"))
        + list(manifest.glob("calc_case_description_*_set.csv"))
    )
    assert csvs, f"No CBIS case_description CSVs under {manifest}"
    
    frames = []
    for c in tqdm(csvs, desc="CBIS CSVs"):
        with _open_read_long(str(c), "rb") as f:
            df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        keep = [k for k in _CBIS_KEEP if k in df.columns]
        df = df.loc[:, keep].copy()
        
        for col in [
            "patient_id",
            "left or right breast",
            "image view",
            "pathology",
            "image file path",
            "ROI mask file path",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        df["image_path"] = df["image file path"].progress_apply(
            lambda s: _resolve_uncropped_dcm_11_only(base, s)
        )
        df["roi_mask_path"] = df["ROI mask file path"].progress_apply(
            lambda s: _resolve_roi_mask_11_only(base, s) if isinstance(s, str) and s else None
        )
        df["label"] = df["pathology"].str.upper().str.contains("MALIGNANT").astype(int)
        frames.append(df)
    
    out = pd.concat(frames, ignore_index=True)
    out["side"] = out["left or right breast"].map(
        lambda s: {"L": "LEFT", "R": "RIGHT"}.get(str(s).upper(), str(s).upper())
    )
    out["view"] = out["image view"].map(_norm_view_token)
    out = out.dropna(subset=["image_path"]).reset_index(drop=True)
    assert len(out) > 0, "CBIS table ended up empty after strict resolution."
    
    grp = (
        out.groupby(["patient_id", "side", "view", "image_path"], dropna=False)
        .agg(
            label=("label", "max"),
            roi_mask_path=(
                "ROI mask file path",
                lambda x: [p for p in out.loc[x.index, "roi_mask_path"] if isinstance(p, str)],
            ),
        )
        .reset_index()
    )
    assert len(grp) > 0, "CBIS grouped table is empty."
    return grp


def _inbreast_norm_file_id(x) -> str:
    """Normalize INbreast file ID."""
    s = re.sub(r"\.0+$", "", str(x).strip())
    return re.sub(r"[^\d]", "", s)


def _inbreast_parse(path: str):
    """Parse INbreast filename to extract patient ID, side, and view."""
    stem = re.sub(r"\.dcm$", "", os.path.basename(path), flags=re.I)
    m = re.search(r"_MG_([LR])_([A-Za-z]+)", stem, flags=re.I)
    side = view = None
    if m:
        side = {"L": "LEFT", "R": "RIGHT"}[m.group(1).upper()]
        vraw = m.group(2).upper()
        view = {"ML": "MLO", "LM": "MLO"}.get(vraw, vraw)
    pid = None
    toks = stem.split("_")
    if len(toks) >= 2:
        pid = toks[1]
    return pid, side, view


def load_inbreast(raw: Path) -> pd.DataFrame:
    """Load INbreast dataset."""
    print("Loading INbreast (with progress)...")
    inbreast_root = raw / "INbreast"
    if (inbreast_root / "INbreast").is_dir():
        inbreast_root = inbreast_root / "INbreast"
    xls = inbreast_root / "INbreast.xls"
    imgs = inbreast_root / "ALL-IMGS"
    assert _exists_long(xls), f"Missing INbreast.xls (checked: {xls})"
    assert _exists_long(imgs), f"Missing INbreast/ALL-IMGS (checked: {imgs})"
    
    with _open_read_long(xls, "rb") as f:
        sheet = pd.read_excel(f, engine="xlrd")
    sheet.columns = sheet.columns.str.strip()
    sheet["file_base"] = sheet["File Name"].map(_inbreast_norm_file_id)
    
    rows = []
    all_files = [fn for fn in _listdir_long(imgs) if fn.lower().endswith(".dcm")]
    
    for _, r in tqdm(list(sheet.iterrows()), total=len(sheet), desc="INbreast rows"):
        base_id = r["file_base"]
        matches = sorted(
            [
                str((imgs / fn).resolve())
                for fn in all_files
                if re.match(rf"^{re.escape(base_id)}_.*\.dcm$", fn, flags=re.I)
            ]
        )
        if not matches:
            continue
        p = _win_long(matches[0])
        pid, side, view = _inbreast_parse(p)
        birads = str(r.get("Bi-Rads", "")).strip()
        label = 1 if re.match(r"^\s*(4|5|6)", birads) else 0
        rows.append(
            dict(
                dataset="INbreast",
                patient_id=str(pid),
                image_path=p,
                label=label,
                side=side,
                view=view,
            )
        )
    
    df = pd.DataFrame(rows)
    assert len(df) > 0, "INbreast table is empty."
    return df


_SNOMED_TO_VIEW = {"399162004": "CC", "399368009": "MLO"}


def _norm_view_from_meaning(s) -> Optional[str]:
    """Normalize view from DICOM meaning."""
    if not s:
        return None
    t = str(s).strip().lower()
    if "cranio" in t and "caudal" in t:
        return "CC"
    if "medio" in t and "oblique" in t:
        return "MLO"
    return None


def _extract_side_view_from_dicom(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract side and view from DICOM file."""
    side = view = None
    sp = _win_long(path)
    if pydicom is not None:
        ds = pydicom.dcmread(sp, stop_before_pixels=True, force=True)
        side = _norm_side_token(
            getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))
        )
        vseq = getattr(ds, "ViewCodeSequence", None)
        if vseq:
            for it in vseq:
                view = _SNOMED_TO_VIEW.get(
                    str(getattr(it, "CodeValue", "")).strip().upper(),
                    _norm_view_from_meaning(getattr(it, "CodeMeaning", "")),
                )
                if view:
                    break
        if view is None:
            vp = str(getattr(ds, "ViewPosition", "")).strip().upper()
            if vp in {"CC", "MLO"}:
                view = vp
            elif vp in {"LM", "ML"}:
                view = "MLO"
    
    if (side is None) or (view is None):
        parts = [Path(path).name] + [par.name for par in list(Path(path).parents)[:3]]
        hint = re.sub(r"[\-./\\]+", " ", " ".join(parts).upper())
        m = re.search(r"([LR])\s*(CC|MLO)\b", hint) or re.search(r"(CC|MLO)\s*([LR])\b", hint)
        if m:
            a, b = m.groups()
            if a in {"L", "R"}:
                side = side or _norm_side_token(a)
                view = view or b.upper()
            else:
                view = view or a.upper()
                side = side or _norm_side_token(b)
        if view in {"LM", "ML"}:
            view = "MLO"
    
    return side, view


def load_cmmd_for_test(raw_dir: Path) -> pd.DataFrame:
    """Load CMMD dataset for testing."""
    print("Loading CMMD (with progress)...")
    cmmd_root = raw_dir / "TheChineseMammographyDatabase"
    if (cmmd_root / "TheChineseMammographyDatabase").is_dir():
        cmmd_root = cmmd_root / "TheChineseMammographyDatabase"
    xlsx = cmmd_root / "CMMD_clinicaldata_revision.xlsx"
    root = cmmd_root / "CMMD"
    assert _exists_long(xlsx), f"Missing CMMD_clinicaldata_revision.xlsx (checked: {xlsx})"
    assert _exists_long(root), f"Missing CMMD directory (checked: {root})"
    
    with _open_read_long(xlsx, "rb") as f:
        df = pd.read_excel(f, engine="openpyxl")
    
    req = {"ID1", "abnormality", "classification"}
    assert req.issubset(df.columns), f"CMMD sheet missing {req - set(df.columns)}"
    df = df[df["abnormality"].astype(str).str.lower().isin({"mass", "both"})]
    
    rows = []
    for id1, cls in tqdm(
        list(zip(df["ID1"].astype(str), df["classification"].astype(str))),
        total=len(df),
        desc="CMMD subjects",
    ):
        subj = (root / id1).resolve()
        if not _exists_long(subj):
            continue
        for dirpath, _, filenames in os.walk(_win_long(str(subj))):
            for fn in filenames:
                if fn.lower().endswith(".dcm"):
                    h = os.path.join(dirpath, fn)
                    s, v = _extract_side_view_from_dicom(h)
                    rows.append(
                        {
                            "dataset": "CMMD",
                            "patient_id": id1,
                            "image_path": _win_long(h),
                            "label": 1 if str(cls).strip().lower() == "malignant" else 0,
                            "side": s,
                            "view": v,
                        }
                    )
    
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["side", "view"]).reset_index(drop=True)
    assert len(out) > 0, "CMMD table is empty after parsing."
    return out


def patient_split(
    df: pd.DataFrame,
    val_frac=0.2,
    threshold_frac=0.0,
    seed=None,
):
    """Split dataset by patient ID into train/val/threshold sets.
    
    Args:
        df: DataFrame with patient_id and label columns
        val_frac: Fraction of patients for validation set
        threshold_frac: Fraction of patients for threshold selection set
        seed: Random seed (defaults to config.SEED)
        
    Returns:
        If threshold_frac > 0: (train_df, val_df, threshold_df)
        Otherwise: (train_df, val_df)
    """
    if seed is None:
        seed = config.SEED
    y_pat = df.groupby("patient_id")["label"].max().astype(int)
    ids_pos = sorted(y_pat.index[y_pat == 1])
    ids_neg = sorted(y_pat.index[y_pat == 0])
    rng = np.random.default_rng(seed)
    
    def _pick(ids, frac):
        """Pick fraction of patients from list."""
        n = len(ids)
        k = max(1, int(round(frac * n)))
        assert n > 0, "No patients found for one of the classes in CBIS split."
        return set(rng.choice(ids, size=k, replace=False).tolist())
    
    # First pick validation set
    val_pat = _pick(ids_pos, val_frac) | _pick(ids_neg, val_frac)
    
    # If threshold set needed, pick from remaining patients
    if threshold_frac > 0:
        remaining_pos = [pid for pid in ids_pos if pid not in val_pat]
        remaining_neg = [pid for pid in ids_neg if pid not in val_pat]
        
        # Calculate threshold set size from original set, not remaining
        # This ensures we have enough patients left for training
        n_pos_total = len(ids_pos)
        n_neg_total = len(ids_neg)
        n_thresh_pos = max(1, int(round(threshold_frac * n_pos_total)))
        n_thresh_neg = max(1, int(round(threshold_frac * n_neg_total)))
        
        # Pick from remaining patients, but limit to available
        n_thresh_pos = min(n_thresh_pos, len(remaining_pos))
        n_thresh_neg = min(n_thresh_neg, len(remaining_neg))
        
        if n_thresh_pos > 0:
            threshold_pos = set(
                rng.choice(remaining_pos, size=n_thresh_pos, replace=False).tolist()
            )
        else:
            threshold_pos = set()
        
        if n_thresh_neg > 0:
            threshold_neg = set(
                rng.choice(remaining_neg, size=n_thresh_neg, replace=False).tolist()
            )
        else:
            threshold_neg = set()
        
        threshold_pat = threshold_pos | threshold_neg
        
        # Remaining patients go to training
        train_pat = set(df["patient_id"].unique()) - val_pat - threshold_pat
        
        tr = df[df["patient_id"].isin(train_pat)].reset_index(drop=True)
        va = df[df["patient_id"].isin(val_pat)].reset_index(drop=True)
        th = df[df["patient_id"].isin(threshold_pat)].reset_index(drop=True)
        
        assert len(tr) > 0, "Empty train set after split."
        assert len(va) > 0, "Empty validation set after split."
        assert len(th) > 0, "Empty threshold set after split."
        
        return tr, va, th
    else:
        # Standard 2-way split
        tr = df[~df["patient_id"].isin(val_pat)].reset_index(drop=True)
        va = df[df["patient_id"].isin(val_pat)].reset_index(drop=True)
        assert len(tr) > 0 and len(va) > 0, "Empty train/val split."
        return tr, va

