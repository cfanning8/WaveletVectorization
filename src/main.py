"""Main entry point for mammography pipeline ablation study."""

import gc
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import config
from .checkpointing import save_checkpoint
from .config_experiments import ExperimentConfig
from .data_loading import (
    _cbis_manifest_root,
    load_cbis_ddsm,
    load_cmmd_for_test,
    load_inbreast,
    patient_split,
)
from .evaluation import (
    aggregate_patient_scores,
    infer_stage2_on_df,
    metrics_with_ci,
    pick_thresholds,
    pick_thresholds_on_val,
)
from .models import FasterRCNNWithAttnMIL
from .training import TrainCfg, train_stage1_mil, train_stage2_det
from .utils import _open_write_long


def _one_patient(df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
    """Select first n patients from dataframe.
    
    Args:
        df: DataFrame with patient_id column
        n: Number of patients to select (None = all patients)
        
    Returns:
        Subsampled DataFrame
    """
    pats = sorted(df["patient_id"].astype(str).unique())
    assert len(pats) > 0, "Empty test set."
    if n is None:
        return df
    pick = pats[: min(n, len(pats))]
    return df[df["patient_id"].isin(pick)].reset_index(drop=True)


def _eval_on_tests(
    model_s2: FasterRCNNWithAttnMIL,
    thresholds: Dict[str, float],
    *,
    inb: pd.DataFrame,
    cmmd: pd.DataFrame,
    wv="haar",
    J=1,
    h1=0.25,
    use_topo: bool = True,
    bootstrap_samples: int = 10,
):
    """Evaluate model on test sets."""
    import gc
    model_s2 = model_s2.to(config.DEVICE)
    
    # Evaluate on INbreast first
    df_inb = infer_stage2_on_df(
        model_s2, inb, wavelet=wv, J=J, h1_pct=h1, use_topo=use_topo
    )
    yI, pI, _ = aggregate_patient_scores(df_inb)
    # Aggressive cleanup after INbreast
    del df_inb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate on CMMD
    df_cm = infer_stage2_on_df(
        model_s2, cmmd, wavelet=wv, J=J, h1_pct=h1, use_topo=use_topo
    )
    yC, pC, _ = aggregate_patient_scores(df_cm)
    # Cleanup after CMMD
    del df_cm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    res = {}
    for name, (y, p) in {"INbreast": (yI, pI), "CMMD": (yC, pC)}.items():
        stats = metrics_with_ci(
            y,
            p,
            thresholds["thr_J"],
            thresholds["thr_spec95"],
            thresholds["thr_sens95"],
            B=bootstrap_samples,
            seed=config.SEED,
        )
        res[name] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in stats.items()
        }
    return res


def _free(*objs):
    """Free memory by moving objects to CPU and deleting them.
    
    This is used to free GPU memory after each ablation experiment.
    """
    for o in objs:
        try:
            if hasattr(o, "to"):
                o.to("cpu")
            if hasattr(o, "cpu"):
                o.cpu()
        except Exception:
            pass
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_table(rows, title):
    """Print evaluation results table."""
    print(f"\n=== {title} ===")
    for key, res in rows:
        I = res["INbreast"]
        C = res["CMMD"]
        line = (
            f"{str(key):>18s} | "
            f"INbreast: P={I['Precision']:.3f} [{I['Precision_CI'][0]:.3f},{I['Precision_CI'][1]:.3f}] "
            f"Se={I['Sensitivity']:.3f} [{I['Sensitivity_CI'][0]:.3f},{I['Sensitivity_CI'][1]:.3f}] "
            f"Sp={I['Specificity']:.3f} [{I['Specificity_CI'][0]:.3f},{I['Specificity_CI'][1]:.3f}] "
            f"Acc={I['Accuracy']:.3f} [{I['Accuracy_CI'][0]:.3f},{I['Accuracy_CI'][1]:.3f}] "
            f"AUC={I['AUC']:.3f} [{I['AUC_CI'][0]:.3f},{I['AUC_CI'][1]:.3f}]  "
            f"||  CMMD: P={C['Precision']:.3f} [{C['Precision_CI'][0]:.3f},{C['Precision_CI'][1]:.3f}] "
            f"Se={C['Sensitivity']:.3f} [{C['Sensitivity_CI'][0]:.3f},{C['Sensitivity_CI'][1]:.3f}] "
            f"Sp={C['Specificity']:.3f} [{C['Specificity_CI'][0]:.3f},{C['Specificity_CI'][1]:.3f}] "
            f"Acc={C['Accuracy']:.3f} [{C['Accuracy_CI'][0]:.3f},{C['Accuracy_CI'][1]:.3f}] "
            f"AUC={C['AUC']:.3f} [{C['AUC_CI'][0]:.3f},{C['AUC_CI'][1]:.3f}]"
        )
        print(line)


def run_all(exp_config: ExperimentConfig):
    """Run ablation study with configuration.
    
    Args:
        exp_config: Experiment configuration
    """
    print("Loading datasets...")
    cbis = load_cbis_ddsm(_cbis_manifest_root(config.CBIS_ROOT))
    assert len(cbis) > 0, "CBIS loaded empty."
    
    # 3-way split if threshold set needed, otherwise 2-way
    if exp_config.use_separate_threshold_set:
        cbis_tr, cbis_va, cbis_th = patient_split(
            cbis,
            val_frac=0.15,
            threshold_frac=exp_config.threshold_set_fraction,
            seed=config.SEED,
        )
        print(f"Split: train={len(cbis_tr)}, val={len(cbis_va)}, threshold={len(cbis_th)}")
    else:
        cbis_tr, cbis_va = patient_split(cbis, val_frac=0.2, seed=config.SEED)
        cbis_th = None
        print(f"Split: train={len(cbis_tr)}, val={len(cbis_va)}")

    # Subsample if configured
    if exp_config.train_samples is not None:
        cbis_tr = cbis_tr.sample(
            n=min(exp_config.train_samples, len(cbis_tr)), random_state=config.SEED
        ).reset_index(drop=True)
    if exp_config.val_samples is not None:
        cbis_va = cbis_va.sample(
            n=min(exp_config.val_samples, len(cbis_va)), random_state=config.SEED
        ).reset_index(drop=True)
    if cbis_th is not None and exp_config.threshold_samples is not None:
        cbis_th = cbis_th.sample(
            n=min(exp_config.threshold_samples, len(cbis_th)), random_state=config.SEED
        ).reset_index(drop=True)
    
    print(f"Using {len(cbis_tr)} train and {len(cbis_va)} val samples.")

    print("Stage-1 training...")
    cfg = TrainCfg(
        backbone=exp_config.s1_backbone,
        steps=exp_config.s1_steps,
        batch_size=exp_config.s1_batch_size,
    )
    s1 = train_stage1_mil(
        cbis_tr,
        cbis_va,
        cfg,
        val_limit=exp_config.val_limit,
        early_stopping=exp_config.early_stopping,
        early_stopping_patience=exp_config.early_stopping_patience,
        early_stopping_min_delta=exp_config.early_stopping_min_delta,
        early_stopping_metric=exp_config.early_stopping_metric,
    )
    
    # Checkpoint Stage-1 if configured
    if exp_config.save_checkpoints:
        checkpoint_dir = (
            Path(exp_config.checkpoint_dir)
            if exp_config.checkpoint_dir
            else config.RESULTS_ROOT / "checkpoints"
        )
        save_checkpoint(
            s1,
            epoch=None,
            step=exp_config.s1_steps,
            config_dict={"backbone": exp_config.s1_backbone, "stage": "stage1"},
            checkpoint_dir=checkpoint_dir,
            name=f"stage1_{exp_config.s1_backbone}",
        )

    print("\n[Evaluation] Loading test sets...")
    inb = _one_patient(load_inbreast(config.RAW_DIR), n=exp_config.test_patients)
    cmmd = _one_patient(load_cmmd_for_test(config.RAW_DIR), n=exp_config.test_patients)
    print(
        f"[SMOKE] Test subsets -> INbreast patients={inb['patient_id'].nunique()}  "
        f"CMMD patients={cmmd['patient_id'].nunique()}"
    )

    tables = {"wavelet_type": [], "depth_J": [], "h1_pct": [], "arch": []}

    print("\n[Ablation] Wavelet type (with topo x8 → input 9ch)")
    for wv in tqdm(exp_config.ablate_wavelet_types, desc="Wavelet types", leave=False):
        s2_wv = train_stage2_det(
            cbis_tr,
            cbis_va,
            s1,
            backbone_name=exp_config.s1_backbone,
            wavelet=wv,
            J=1,
            h1_pct=0.25,
            steps=exp_config.s2_steps,
            lr=cfg.lr,
            betas=cfg.betas,
            use_topo=True,
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        
        # Pick thresholds
        if exp_config.use_separate_threshold_set and cbis_th is not None:
            thr = pick_thresholds(
                s2_wv, cbis_th, wavelet=wv, J=1, h1=0.25, use_topo=True
            )
        else:
            thr = pick_thresholds_on_val(
                s2_wv, cbis_va, wavelet=wv, J=1, h1=0.25, use_topo=True
            )
        
        res = _eval_on_tests(
            s2_wv,
            thr,
            inb=inb,
            cmmd=cmmd,
            wv=wv,
            J=1,
            h1=0.25,
            use_topo=True,
            bootstrap_samples=exp_config.bootstrap_samples,
        )
        tables["wavelet_type"].append((wv, res))
        
        # Checkpoint if configured
        if exp_config.save_checkpoints:
            checkpoint_dir = (
                Path(exp_config.checkpoint_dir)
                if exp_config.checkpoint_dir
                else config.RESULTS_ROOT / "checkpoints"
            )
            save_checkpoint(
                s2_wv,
                step=exp_config.s2_steps,
                metrics=res,
                config_dict={
                    "wavelet": wv,
                    "J": 1,
                    "h1_pct": 0.25,
                    "use_topo": True,
                    "ablation": "wavelet_type",
                },
                checkpoint_dir=checkpoint_dir,
                name=f"stage2_wavelet_{wv}",
            )
        
        _free(s2_wv)
    print("\n[Ablation] Wavelet depth J")
    wavelet_for_j_ablation = "haar"
    for J in tqdm(exp_config.ablate_J, desc="Wavelet depth J", leave=False):
        s2_J = train_stage2_det(
            cbis_tr,
            cbis_va,
            s1,
            backbone_name=exp_config.s1_backbone,
            wavelet=wavelet_for_j_ablation,
            J=J,
            h1_pct=0.25,
            steps=exp_config.s2_steps,
            lr=cfg.lr,
            betas=cfg.betas,
            use_topo=True,
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        
        # Pick thresholds
        if exp_config.use_separate_threshold_set and cbis_th is not None:
            thr = pick_thresholds(
                s2_J, cbis_th, wavelet=wavelet_for_j_ablation, J=J, h1=0.25, use_topo=True
            )
        else:
            thr = pick_thresholds_on_val(
                s2_J, cbis_va, wavelet=wavelet_for_j_ablation, J=J, h1=0.25, use_topo=True
            )
        
        res = _eval_on_tests(
            s2_J,
            thr,
            inb=inb,
            cmmd=cmmd,
            wv=wavelet_for_j_ablation,
            J=J,
            h1=0.25,
            use_topo=True,
            bootstrap_samples=exp_config.bootstrap_samples,
        )
        tables["depth_J"].append((J, res))
        
        # Checkpoint if configured
        if exp_config.save_checkpoints:
            checkpoint_dir = (
                Path(exp_config.checkpoint_dir)
                if exp_config.checkpoint_dir
                else config.RESULTS_ROOT / "checkpoints"
            )
            save_checkpoint(
                s2_J,
                step=exp_config.s2_steps,
                metrics=res,
                config_dict={
                    "wavelet": wavelet_for_j_ablation,
                    "J": J,
                    "h1_pct": 0.25,
                    "use_topo": True,
                    "ablation": "depth_J",
                },
                checkpoint_dir=checkpoint_dir,
                name=f"stage2_J_{J}",
            )
        
        _free(s2_J)

    print("\n[Ablation] H1 keep-percent")
    wavelet_for_h1_ablation = "haar"
    for h1 in tqdm(exp_config.ablate_h1, desc="H1 keep-percent", leave=False):
        s2_h = train_stage2_det(
            cbis_tr,
            cbis_va,
            s1,
            backbone_name=exp_config.s1_backbone,
            wavelet=wavelet_for_h1_ablation,
            J=1,
            h1_pct=h1,
            steps=exp_config.s2_steps,
            lr=cfg.lr,
            betas=cfg.betas,
            use_topo=True,
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        
        # Pick thresholds
        if exp_config.use_separate_threshold_set and cbis_th is not None:
            thr = pick_thresholds(
                s2_h, cbis_th, wavelet=wavelet_for_h1_ablation, J=1, h1=h1, use_topo=True
            )
        else:
            thr = pick_thresholds_on_val(
                s2_h, cbis_va, wavelet=wavelet_for_h1_ablation, J=1, h1=h1, use_topo=True
            )
        
        res = _eval_on_tests(
            s2_h,
            thr,
            inb=inb,
            cmmd=cmmd,
            wv=wavelet_for_h1_ablation,
            J=1,
            h1=h1,
            use_topo=True,
            bootstrap_samples=exp_config.bootstrap_samples,
        )
        tables["h1_pct"].append((h1, res))
        
        # Checkpoint if configured
        if exp_config.save_checkpoints:
            checkpoint_dir = (
                Path(exp_config.checkpoint_dir)
                if exp_config.checkpoint_dir
                else config.RESULTS_ROOT / "checkpoints"
            )
            save_checkpoint(
                s2_h,
                step=exp_config.s2_steps,
                metrics=res,
                config_dict={
                    "wavelet": wavelet_for_h1_ablation,
                    "J": 1,
                    "h1_pct": h1,
                    "use_topo": True,
                    "ablation": "h1_pct",
                },
                checkpoint_dir=checkpoint_dir,
                name=f"stage2_h1_{h1}",
            )
        
        _free(s2_h)

    print("\n[Ablation] Architectures (± topo, now topo=9ch input when enabled)")
    for arch in tqdm(exp_config.arch_list, desc="Architectures", leave=False):
        print(f" - {arch} (no topo)")
        s1_arch = train_stage1_mil(
            cbis_tr,
            cbis_va,
            TrainCfg(
                backbone=arch,
                steps=exp_config.s1_steps,
                batch_size=exp_config.s1_batch_size,
            ),
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        s2_arch_notopo = train_stage2_det(
            cbis_tr,
            cbis_va,
            s1_arch,
            backbone_name=arch,
            wavelet="haar",
            J=1,
            h1_pct=0.25,
            steps=exp_config.s2_steps,
            lr=cfg.lr,
            betas=cfg.betas,
            use_topo=False,
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        
        # Pick thresholds
        if exp_config.use_separate_threshold_set and cbis_th is not None:
            thr0 = pick_thresholds(
                s2_arch_notopo, cbis_th, wavelet="haar", J=1, h1=0.25, use_topo=False
            )
        else:
            thr0 = pick_thresholds_on_val(
                s2_arch_notopo, cbis_va, wavelet="haar", J=1, h1=0.25, use_topo=False
            )
        
        res0 = _eval_on_tests(
            s2_arch_notopo,
            thr0,
            inb=inb,
            cmmd=cmmd,
            wv="haar",
            J=1,
            h1=0.25,
            use_topo=False,
            bootstrap_samples=exp_config.bootstrap_samples,
        )
        tables["arch"].append((f"{arch}-noTopo", res0))
        
        # Checkpoint if configured
        if exp_config.save_checkpoints:
            checkpoint_dir = (
                Path(exp_config.checkpoint_dir)
                if exp_config.checkpoint_dir
                else config.RESULTS_ROOT / "checkpoints"
            )
            save_checkpoint(
                s2_arch_notopo,
                step=exp_config.s2_steps,
                metrics=res0,
                config_dict={
                    "backbone": arch,
                    "wavelet": "haar",
                    "J": 1,
                    "h1_pct": 0.25,
                    "use_topo": False,
                    "ablation": "arch",
                },
                checkpoint_dir=checkpoint_dir,
                name=f"stage2_{arch}_noTopo",
            )
        
        _free(s2_arch_notopo)

        print(f" - {arch} (+ topo: FiLM+RoI)")
        s2_arch_topo = train_stage2_det(
            cbis_tr,
            cbis_va,
            s1_arch,
            backbone_name=arch,
            wavelet="haar",
            J=1,
            h1_pct=0.25,
            steps=exp_config.s2_steps,
            lr=cfg.lr,
            betas=cfg.betas,
            use_topo=True,
            val_limit=exp_config.val_limit,
            early_stopping=exp_config.early_stopping,
            early_stopping_patience=exp_config.early_stopping_patience,
            early_stopping_min_delta=exp_config.early_stopping_min_delta,
            early_stopping_metric=exp_config.early_stopping_metric,
        )
        
        # Pick thresholds
        if exp_config.use_separate_threshold_set and cbis_th is not None:
            thr1 = pick_thresholds(
                s2_arch_topo, cbis_th, wavelet="haar", J=1, h1=0.25, use_topo=True
            )
        else:
            thr1 = pick_thresholds_on_val(
                s2_arch_topo, cbis_va, wavelet="haar", J=1, h1=0.25, use_topo=True
            )
        
        res1 = _eval_on_tests(
            s2_arch_topo,
            thr1,
            inb=inb,
            cmmd=cmmd,
            wv="haar",
            J=1,
            h1=0.25,
            use_topo=True,
            bootstrap_samples=exp_config.bootstrap_samples,
        )
        tables["arch"].append((f"{arch}+Topo(8)+FiLM+RoI", res1))
        
        # Checkpoint if configured
        if exp_config.save_checkpoints:
            checkpoint_dir = (
                Path(exp_config.checkpoint_dir)
                if exp_config.checkpoint_dir
                else config.RESULTS_ROOT / "checkpoints"
            )
            save_checkpoint(
                s2_arch_topo,
                step=exp_config.s2_steps,
                metrics=res1,
                config_dict={
                    "backbone": arch,
                    "wavelet": "haar",
                    "J": 1,
                    "h1_pct": 0.25,
                    "use_topo": True,
                    "ablation": "arch",
                },
                checkpoint_dir=checkpoint_dir,
                name=f"stage2_{arch}_Topo",
            )
        
        _free(s2_arch_topo, s1_arch)
        # Clear cache only at end of architecture ablation loop
        if arch == exp_config.arch_list[-1]:  # Last architecture
            torch.cuda.empty_cache()

    print_table(tables["wavelet_type"], "Wavelet Type (J=1, H1=25%)")
    print_table(tables["depth_J"], "Wavelet Depth J (haar, H1=25%)")
    print_table(tables["h1_pct"], "H1 Keep-percent (haar, J=1)")
    print_table(tables["arch"], "Architecture Ablation (± Topo, FiLM+RoI)")

    out_json = config.RESULTS_ROOT / "ablations_summary.json"
    with _open_write_long(out_json, "wb") as f:
        f.write(json.dumps(tables, indent=2).encode("utf-8"))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    from .config_experiments import PRODUCTION

    # Use PRODUCTION config
    exp_config = PRODUCTION
    
    # Enable separate threshold set for better threshold selection
    exp_config.use_separate_threshold_set = True
    exp_config.threshold_set_fraction = 0.15
    
    run_all(exp_config)
