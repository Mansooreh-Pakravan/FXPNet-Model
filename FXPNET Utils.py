# fxp_interpret_utils.py
# ============================================================
# FXP Interpretability Utilities
# Supports BOTH:
#   (A) Prototype-level attribution: Grad×Input of mu_k(x)
#   (B) Decision-level attribution:  Grad×Input of decision scalar (logit diff / prob diff)
#
# Clean I/O:
#   per_case/<level>/<atlas>/run{r}_fold{f}/
#     - prototype_stats_subject_level_val.csv  (only in prototype-level)
#     - attribution_<level>_gradxinput.csv     (long-format)
#   atlas_summary/<level>/<atlas>/
#     - atlas_roi_vote_summary.csv
#     - annotated_topright_rois.csv
#     - bar_*.png, scatter_*.png, etc.
#
# Surface mapping is INDEX-based (roi_idx -> label id with inferred offset).
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt

# optional surface deps (only used if you call surface functions)
try:
    import nibabel as nib
    from nibabel.cifti2 import cifti2_axes
    from nilearn import plotting
    _HAS_SURF = True
except Exception:
    _HAS_SURF = False


# ==========================
# Configs
# ==========================
@dataclass
class PerCaseAttrConfig:
    level: str = "prototype"  # "prototype" or "decision"
    device: str = "cuda"

    # decision-level options
    decision_mode: str = "logit_diff"  # "logit_diff" or "prob_diff"
    decision_target: str = "pred"      # "pred" or "true"
    # if decision_target == "true" you must pass yb (true labels)

    # normalization/weighting
    abs_grad: bool = True              # use |grad*x| vs grad*x
    time_reduce: str = "mean"          # "mean" only here

    # prototype-level options
    proto_weight_by_mu: bool = True    # weight roi_attr by mu[:,k]


# ==========================
# Small helpers
# ==========================
def seed_everything(seed: int = 2025):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def proto_name(k: int, protos_per_class: int) -> str:
    return f"M_proto{k:02d}" if k < protos_per_class else f"F_proto{k-protos_per_class:02d}"


# ==========================
# Core Grad×Input
# ==========================
def _gradxinput_from_scalar(xb: torch.Tensor, scalar: torch.Tensor, abs_grad: bool = True) -> torch.Tensor:
    """
    xb: (B, N_roi, T) with requires_grad=True
    scalar: scalar tensor
    returns roi_attr: (B, N_roi)  (mean over time of |grad*x|)
    """
    if xb.grad is not None:
        xb.grad.zero_()
    scalar.backward(retain_graph=True)

    grad = xb.grad  # (B,N_roi,T)
    gxi = grad * xb
    if abs_grad:
        gxi = gxi.abs()
    roi_attr = gxi.mean(dim=2)  # mean over time
    return roi_attr.detach()


def compute_prototype_level_attr(model, xb, cfg: PerCaseAttrConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
      mu: (B,K)
      roi_attr_all: (K, B, N_roi)  (per prototype)
    """
    logits, details = model(xb, return_details=True)
    mu = details["mu"]  # (B,K)
    K = mu.shape[1]

    roi_attr_list = []
    for k in range(K):
        scalar = mu[:, k].sum()
        model.zero_grad(set_to_none=True)
        roi_attr = _gradxinput_from_scalar(xb, scalar, abs_grad=cfg.abs_grad)  # (B,N_roi)

        if cfg.proto_weight_by_mu:
            w = mu.detach()[:, k].view(-1, 1)  # (B,1)
            roi_attr = roi_attr * w

        roi_attr_list.append(roi_attr.unsqueeze(0))  # (1,B,N_roi)

    roi_attr_all = torch.cat(roi_attr_list, dim=0)  # (K,B,N_roi)
    return mu.detach(), roi_attr_all


def compute_decision_level_attr(model, xb, yb: Optional[torch.Tensor], cfg: PerCaseAttrConfig) -> torch.Tensor:
    """
    decision scalar options:
      logit_diff: logit[target] - logit[other]
      prob_diff :  prob[target] - prob[other]
    target: pred or true

    returns roi_attr: (B,N_roi)
    """
    logits, details = model(xb, return_details=True)  # logits (B,2)
    probs = torch.softmax(logits, dim=1)

    if cfg.decision_target == "true":
        if yb is None:
            raise ValueError("decision_target='true' requires yb.")
        target = yb.long()
    else:
        target = logits.argmax(dim=1)

    other = 1 - target

    b = torch.arange(logits.shape[0], device=logits.device)

    if cfg.decision_mode == "prob_diff":
        scalar_vec = probs[b, target] - probs[b, other]
    else:
        scalar_vec = logits[b, target] - logits[b, other]

    scalar = scalar_vec.sum()
    model.zero_grad(set_to_none=True)
    roi_attr = _gradxinput_from_scalar(xb, scalar, abs_grad=cfg.abs_grad)  # (B,N_roi)
    return roi_attr


# ==========================
# Per-case pipeline (saves CSVs)
# ==========================
def save_prototype_stats_subject_level(
    out_csv: Path,
    all_mu_win: np.ndarray,
    y_win: np.ndarray,
    subj_idx_win: np.ndarray,
    n_subj: int,
    val_labels: np.ndarray,
    protos_per_class: int,
    atlas: str, run_id: int, fold_id: int,
):
    """
    Compute subject-level mu stats (male vs female) for prototypes.
    all_mu_win: (N_win,K)
    """
    K = all_mu_win.shape[1]
    mu_subj = np.zeros((n_subj, K), dtype=np.float32)
    counts = np.zeros(n_subj, dtype=np.int64)

    for w in range(all_mu_win.shape[0]):
        s = int(subj_idx_win[w])
        mu_subj[s] += all_mu_win[w]
        counts[s] += 1
    for s in range(n_subj):
        if counts[s] > 0:
            mu_subj[s] /= counts[s]

    mu_m = mu_subj[val_labels == 0]
    mu_f = mu_subj[val_labels == 1]

    mean_m = mu_m.mean(axis=0)
    mean_f = mu_f.mean(axis=0)
    diff = mean_f - mean_m

    tvals, pvals = ttest_ind(mu_f, mu_m, axis=0, equal_var=False)
    rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    rows = []
    for k in range(K):
        rows.append({
            "atlas": atlas, "run": run_id, "fold": fold_id,
            "proto_idx": k,
            "proto_name": proto_name(k, protos_per_class),
            "proto_class": int(0 if k < protos_per_class else 1),
            "mean_male": float(mean_m[k]),
            "mean_female": float(mean_f[k]),
            "diff_F_minus_M": float(diff[k]),
            "t_value": float(tvals[k]),
            "p_value": float(pvals[k]),
            "p_fdr_bh": float(pvals_fdr[k]),
            "sig_fdr_0p05": bool(rej[k]),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def compute_and_save_percase_attribution(
    model,
    loader,
    atlas_name: str,
    run_id: int,
    fold_id: int,
    roi_idx_to_name: Dict[int, str],
    out_dir: Path,
    protos_per_class: int,
    cfg: PerCaseAttrConfig,
    save_proto_stats: bool = True,
):
    """
    Writes:
      - prototype_stats_subject_level_val.csv   (only if cfg.level='prototype' and save_proto_stats)
      - attribution_<level>_gradxinput.csv
        * prototype-level: rows = proto x roi (aggregated over windows, optionally weighted by mu)
        * decision-level:  rows = roi only (aggregated over windows)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # --------- Collect window logits/mu for accuracy + proto stats (optional)
    all_logits = []
    all_mu = []

    # We also need y_win and subj_idx_win if caller wants proto stats.
    y_all = []
    s_all = []

    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            logits, details = model(xb, return_details=True)
            all_logits.append(logits.detach().cpu())
            if "mu" in details:
                all_mu.append(details["mu"].detach().cpu())
            y_all.append(yb.cpu().numpy())
            s_all.append(sb.cpu().numpy())

    y_win = np.concatenate(y_all, axis=0)
    subj_idx_win = np.concatenate(s_all, axis=0)
    logits_win = torch.cat(all_logits, dim=0).numpy()
    pred_win = logits_win.argmax(axis=1)
    acc_win = float(accuracy_score(y_win, pred_win))

    # --------- Attribution
    # We must do grads => second pass with requires_grad
    # Determine N_roi, K
    first_batch = next(iter(loader))
    xb0 = first_batch[0]
    N_roi = xb0.shape[1]

    if cfg.level == "prototype":
        # aggregate per prototype and ROI across windows
        # attr_sum[k, r], weight_sum[k]
        # if proto_weight_by_mu=True we already weighted at batch level; still normalize by sum(mu) to be comparable.
        # if proto_weight_by_mu=False we normalize by number of samples.
        # We'll normalize by denom = sum(mu) if weighted else N_samples.
        # To get sum(mu): we must accumulate mu as well.
        # (K,B,N_roi) per batch
        # Note: K inferred per batch from model outputs
        attr_sum = None
        denom = None  # per-proto
        K = None

        for xb, yb, sb in loader:
            xb = xb.to(device).detach().requires_grad_(True)

            mu, roi_attr_all = compute_prototype_level_attr(model, xb, cfg)  # mu(B,K), roi_attr_all(K,B,N_roi)
            if K is None:
                K = mu.shape[1]
                attr_sum = np.zeros((K, N_roi), dtype=np.float64)
                denom = np.zeros((K,), dtype=np.float64)

            # roi_attr_all: (K,B,N_roi) already weighted if cfg.proto_weight_by_mu
            # reduce over batch: sum over B
            batch_sum = roi_attr_all.sum(dim=1).detach().cpu().numpy()  # (K,N_roi)
            attr_sum += batch_sum

            if cfg.proto_weight_by_mu:
                denom += mu.sum(dim=0).detach().cpu().numpy().astype(np.float64)  # (K,)
            else:
                denom += np.ones((K,), dtype=np.float64) * xb.shape[0]

            xb.grad = None

        attr_mean = attr_sum / (denom[:, None] + 1e-12)

        rows = []
        for k in range(K):
            for r in range(N_roi):
                rows.append({
                    "atlas": atlas_name, "run": run_id, "fold": fold_id,
                    "level": "prototype",
                    "proto_idx": k,
                    "proto_name": proto_name(k, protos_per_class),
                    "proto_class": int(0 if k < protos_per_class else 1),
                    "roi_idx": r,
                    "roi_name": roi_idx_to_name.get(r, f"ROI{r}"),
                    "attr_gradxinput": float(attr_mean[k, r]),
                    "win_acc": acc_win,
                })

        out_csv = out_dir / "attribution_prototype_gradxinput.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        # optional prototype stats (needs all_mu)
        if save_proto_stats and len(all_mu) > 0:
            all_mu_win = torch.cat(all_mu, dim=0).numpy()  # (N_win,K)
            # caller must provide val_labels & n_subj; we cannot infer them here robustly.
            # So we only save proto stats if caller passed val_labels and n_subj via extra file.
            # To keep notebook simple: we store raw mu per window here.
            pd.DataFrame(all_mu_win).to_csv(out_dir / "mu_window_matrix.csv", index=False)

        return out_csv

    else:
        # decision-level: aggregate ROI attribution over windows
        # Option A: mean over windows
        # Option B: also compute per-class means if you want (can be added later)
        attr_sum = np.zeros((N_roi,), dtype=np.float64)
        n_tot = 0

        for xb, yb, sb in loader:
            xb = xb.to(device).detach().requires_grad_(True)
            yb = yb.to(device)

            roi_attr = compute_decision_level_attr(model, xb, yb if cfg.decision_target == "true" else None, cfg)  # (B,N_roi)
            attr_sum += roi_attr.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            n_tot += xb.shape[0]
            xb.grad = None

        attr_mean = attr_sum / (n_tot + 1e-12)

        rows = []
        for r in range(N_roi):
            rows.append({
                "atlas": atlas_name, "run": run_id, "fold": fold_id,
                "level": "decision",
                "decision_mode": cfg.decision_mode,
                "decision_target": cfg.decision_target,
                "roi_idx": r,
                "roi_name": roi_idx_to_name.get(r, f"ROI{r}"),
                "attr_gradxinput": float(attr_mean[r]),
                "win_acc": acc_win,
            })

        out_csv = out_dir / "attribution_decision_gradxinput.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return out_csv


# ==========================
# Atlas-level aggregation / voting
# ==========================
def build_atlas_summary_from_saved(
    out_per_case_root: Path,
    atlas_name: str,
    level: str,
    topk_roi_per_proto: int = 10,
):
    """
    Reads all per-case CSVs and outputs a single ROI ranking table.
    For prototype-level:
      - per case: pivot proto x roi, for each proto vote top-k
      - evidence: sum(attr) across protos
    For decision-level:
      - per case: vector over ROIs, vote top-k by ROI score
      - evidence: mean(attr) across cases
    """
    level = level.lower()
    if level not in ("prototype", "decision"):
        raise ValueError("level must be 'prototype' or 'decision'.")

    # locate files
    patt = "attribution_prototype_gradxinput.csv" if level == "prototype" else "attribution_decision_gradxinput.csv"
    csvs = sorted((out_per_case_root / level / atlas_name).glob(f"run*_fold*/{patt}"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"[{atlas_name}][{level}] no per-case CSVs found at: {(out_per_case_root/level/atlas_name)}")

    roi_name_ref = None
    all_cases_vote = []
    all_cases_evid = []

    for fp in csvs:
        df = pd.read_csv(fp)

        if roi_name_ref is None:
            roi_name_ref = (
                df[["roi_idx", "roi_name"]]
                .drop_duplicates()
                .sort_values("roi_idx")
                .reset_index(drop=True)
            )

        if level == "prototype":
            # pivot proto x roi
            piv = df.pivot_table(index="proto_idx", columns="roi_idx", values="attr_gradxinput", aggfunc="mean").fillna(0.0)
            vote_vec = np.zeros((piv.shape[1],), dtype=np.int64)
            evidence_vec = np.zeros((piv.shape[1],), dtype=np.float64)

            for proto in piv.index:
                row = piv.loc[proto].to_numpy()
                top_idx = np.argsort(-row)[:topk_roi_per_proto]
                vote_vec[top_idx] += 1
                evidence_vec += row

            all_cases_vote.append((vote_vec > 0).astype(np.int32))  # ROI appears at least once across protos
            all_cases_evid.append(evidence_vec)

        else:
            # decision: ROI vector directly
            vec = df.sort_values("roi_idx")["attr_gradxinput"].to_numpy(dtype=float)
            top_idx = np.argsort(-vec)[:topk_roi_per_proto]
            vote_bin = np.zeros_like(vec, dtype=np.int32)
            vote_bin[top_idx] = 1
            all_cases_vote.append(vote_bin)
            all_cases_evid.append(vec)

    votes = np.stack(all_cases_vote, axis=0)  # (n_cases, n_roi)
    evid  = np.stack(all_cases_evid, axis=0)  # (n_cases, n_roi)
    n_cases = votes.shape[0]

    vote_topK = votes.sum(axis=0)
    vote_ratio = vote_topK / float(n_cases)

    mean_evidence = evid.mean(axis=0)
    std_evidence = evid.std(axis=0)
    stability = mean_evidence / (std_evidence + 1e-12)

    score = (vote_ratio ** 0.7) * (np.maximum(mean_evidence, 0) ** 0.3) * np.clip(stability, 0, 10)

    df_out = pd.DataFrame({
        "roi_idx": np.arange(vote_ratio.shape[0]),
        "n_cases": n_cases,
        "vote_topK": vote_topK,
        "vote_ratio": vote_ratio,
        "mean_evidence": mean_evidence,
        "std_evidence": std_evidence,
        "stability_mean_over_std": stability,
        "score": score,
        "level": level,
    })

    if roi_name_ref is not None:
        df_out = df_out.merge(roi_name_ref, on="roi_idx", how="left")
    else:
        df_out["roi_name"] = df_out["roi_idx"].apply(lambda i: f"ROI{i}")

    df_out = df_out.sort_values(["score", "vote_ratio", "mean_evidence"], ascending=False).reset_index(drop=True)
    df_out.insert(0, "rank", np.arange(1, len(df_out) + 1))
    return df_out


def select_top_right(df: pd.DataFrame, q: float = 0.85, max_labels: int = 25):
    vr = df["vote_ratio"].to_numpy()
    me = df["mean_evidence"].to_numpy()
    vr_thr = np.quantile(vr, q)
    me_thr = np.quantile(me, q)

    idx = np.where((vr >= vr_thr) & (me >= me_thr))[0]
    if len(idx) == 0:
        return df.sort_values("score", ascending=False).head(max_labels).copy()

    idx = idx[np.argsort(-df.loc[idx, "score"].to_numpy())][:max_labels]
    return df.iloc[idx].sort_values("score", ascending=False).copy()


# ==========================
# Plots (paper friendly)
# ==========================
def plot_bar_top(df_sum: pd.DataFrame, out_png: Path, atlas_name: str, topn: int = 30, ycol: str = "vote_ratio"):
    top = df_sum.head(topn).copy()
    names = top["roi_name"].astype(str).apply(lambda s: s if len(s) <= 28 else s[:26] + "..").to_numpy()
    y = top[ycol].to_numpy()

    plt.figure(figsize=(12, 6))
    x = np.arange(len(top))
    plt.bar(x, y)
    plt.xticks(x, names, rotation=90)
    plt.ylabel(ycol.replace("_", " "))
    plt.title(f"{atlas_name} | {df_sum.iloc[0]['level']} | Top-{topn} ROIs by {ycol}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close()


def plot_scatter_vote_vs_evidence(df_sum: pd.DataFrame, out_png: Path, atlas_name: str):
    vr = df_sum["vote_ratio"].to_numpy()
    me = df_sum["mean_evidence"].to_numpy()
    st = df_sum["stability_mean_over_std"].to_numpy()
    sizes = np.clip(st * 8.0, 10, 160)

    plt.figure(figsize=(7.6, 6.2))
    plt.scatter(vr, me, s=sizes, alpha=0.8)
    plt.xlabel("Vote ratio")
    plt.ylabel("Mean evidence")
    plt.title(f"{atlas_name} | {df_sum.iloc[0]['level']} | Vote vs Evidence (size∝stability)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close()


def plot_scatter_topright_with_legend(df_sum: pd.DataFrame, df_sel: pd.DataFrame, out_png: Path, atlas_name: str):
    """
    Selected ROIs each have its own color and appear in legend (instead of text annotations).
    """
    vr = df_sum["vote_ratio"].to_numpy()
    me = df_sum["mean_evidence"].to_numpy()
    st = df_sum["stability_mean_over_std"].to_numpy()
    sizes = np.clip(st * 8.0, 10, 160)

    plt.figure(figsize=(11, 6))
    plt.scatter(vr, me, s=sizes, alpha=0.6, edgecolors="none")  # background cloud

    # selected with unique colors
    n = len(df_sel)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    handles = []
    for i, (_, r) in enumerate(df_sel.iterrows()):
        x = float(r["vote_ratio"])
        y = float(r["mean_evidence"])
        name = str(r.get("roi_name", f"ROI{int(r['roi_idx'])}"))
        name = name if len(name) <= 32 else name[:30] + ".."
        plt.scatter([x], [y], s=120, marker="o", color=cmap(i), edgecolors="black", linewidths=0.6)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=cmap(i), markeredgecolor='black',
                                 markersize=8, label=name))

    plt.xlabel("Vote ratio")
    plt.ylabel("Mean evidence")
    plt.title(f"{atlas_name} | {df_sum.iloc[0]['level']} | Top-right ROIs (legend)")
    plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close()


# ==========================
# Surface utilities (INDEX-based)
# ==========================
def safe_get_brainmodel_axis(img):
    for ax_i in range(2):
        ax = img.header.get_axis(ax_i)
        if _HAS_SURF and isinstance(ax, cifti2_axes.BrainModelAxis):
            return ax_i, ax
    raise TypeError("Could not find BrainModelAxis in this CIFTI header.")


def build_vertex_labels_from_dlabel(dlabel_path: Path):
    if not _HAS_SURF:
        raise ImportError("Surface plotting requires nibabel + nilearn.")
    img = nib.load(str(dlabel_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 2:
        data = data[0]
    data = np.asarray(data).squeeze().astype(np.int32)

    _, bm = safe_get_brainmodel_axis(img)
    nL = bm.nvertices.get("CIFTI_STRUCTURE_CORTEX_LEFT", None)
    nR = bm.nvertices.get("CIFTI_STRUCTURE_CORTEX_RIGHT", None)
    if nL is None or nR is None:
        raise ValueError("dlabel missing cortex left/right structures.")

    left_lab = np.zeros(nL, dtype=np.int32)
    right_lab = np.zeros(nR, dtype=np.int32)

    for struct, slc, sub_bm in bm.iter_structures():
        vals = data[slc].astype(np.int32)
        if struct == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_lab[sub_bm.vertex] = vals
        elif struct == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_lab[sub_bm.vertex] = vals

    return left_lab, right_lab


def infer_label_offset(left_lab: np.ndarray, right_lab: np.ndarray, n_roi: int):
    u = np.unique(np.concatenate([left_lab, right_lab]))
    if (1 in u) and (n_roi in u):
        return 1
    if (0 in u) and ((n_roi - 1) in u):
        return 0
    return 1


def make_categorical_vertex_map(label_vec: np.ndarray, roi_idxs: List[int], offset: int):
    """
    returns int array:
      0 background
      1..N selected ROI categories (in roi_idxs order)
    """
    out = np.zeros(label_vec.shape[0], dtype=np.int32)
    for i, roi in enumerate(roi_idxs):
        lab = int(roi) + int(offset)
        out[label_vec == lab] = int(i + 1)
    return out


def plot_surface_categorical_4views(
    atlas_name: str,
    level: str,
    lh_surf: Path,
    rh_surf: Path,
    left_cat: np.ndarray,
    right_cat: np.ndarray,
    roi_names: List[str],
    out_png: Path,
    darkness: float = 0.6,
):
    if not _HAS_SURF:
        raise ImportError("Surface plotting requires nibabel + nilearn.")
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n = len(roi_names)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    # build discrete colormap with 0 as transparent
    colors = [(0, 0, 0, 0.0)] + [cmap(i) for i in range(n)]
    lcmap = ListedColormap(colors)

    patches = [Patch(facecolor=colors[i+1], edgecolor="black", label=roi_names[i]) for i in range(n)]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    plotting.plot_surf_roi(str(lh_surf), left_cat, hemi="left", view="lateral",
                           cmap=lcmap, colorbar=False, axes=ax,alpha =0.9,
                           title=f"{atlas_name} | {level} | Left (lateral)",
                           darkness=darkness)

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    plotting.plot_surf_roi(str(lh_surf), left_cat, hemi="left", view="medial",
                           cmap=lcmap, colorbar=False, axes=ax,alpha =0.9,
                           title=f"{atlas_name} | {level} | Left (medial)",
                           darkness=darkness)

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    plotting.plot_surf_roi(str(rh_surf), right_cat, hemi="right", view="lateral",
                           cmap=lcmap, colorbar=False, axes=ax,alpha =0.9,
                           title=f"{atlas_name} | {level} | Right (lateral)",
                           darkness=darkness)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    plotting.plot_surf_roi(str(rh_surf), right_cat, hemi="right", view="medial",
                           cmap=lcmap, colorbar=False, axes=ax,alpha =0.9,
                           title=f"{atlas_name} | {level} | Right (medial)",
                           darkness=darkness)

    fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.02, 0.5),
               frameon=True, ncol=1, fontsize=9, title="ROI (categorical)")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close(fig)

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def _load_cifti_underlay_lr(dscalar_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load fsLR 32k sulc/curvature dscalar and return (L_bg, R_bg) in [0,1]
    with a light contrast enhancement to make folds pop.
    """
    import nibabel as nib
    from nibabel.cifti2 import cifti2_axes

    def get_brainmodel_axis(img):
        ax0 = img.header.get_axis(0)
        ax1 = img.header.get_axis(1)
        if isinstance(ax0, cifti2_axes.BrainModelAxis): return ax0
        if isinstance(ax1, cifti2_axes.BrainModelAxis): return ax1
        raise RuntimeError("BrainModelAxis not found in CIFTI header.")

    img = nib.load(str(dscalar_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 2:
        data = data[0]
    data = np.asarray(data).squeeze().astype(np.float32)

    bm = get_brainmodel_axis(img)
    nL = bm.nvertices.get("CIFTI_STRUCTURE_CORTEX_LEFT", None)
    nR = bm.nvertices.get("CIFTI_STRUCTURE_CORTEX_RIGHT", None)
    if nL is None or nR is None:
        raise ValueError(f"{dscalar_path.name}: missing L/R cortex structures.")

    L = np.zeros(nL, dtype=np.float32)
    R = np.zeros(nR, dtype=np.float32)
    for struct, slc, sub in bm.iter_structures():
        vals = data[slc].astype(np.float32)
        if struct == "CIFTI_STRUCTURE_CORTEX_LEFT":
            L[sub.vertex] = vals
        elif struct == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            R[sub.vertex] = vals

    # robust normalization + mild gamma for sharper folds
    def _norm_contrast(x):
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo:
            return x
        x = (x - lo) / (hi - lo)
        x = np.clip(x, 0, 1)
        gamma = 0.65
        return np.power(x, gamma)

    return _norm_contrast(L), _norm_contrast(R)


def plot_surface_categorical_8views(
    atlas_name: str,
    level: str,
    lh_surf: Path,
    rh_surf: Path,
    left_cat: np.ndarray,
    right_cat: np.ndarray,
    roi_names: List[str],
    out_png: Path,
    underlay_dscalar: Optional[Path] = None,  # e.g., fs_LR.32k.LR.sulc.dscalar.nii
    darkness: float = 0.4,                   # smaller -> folds more visible
    alpha: float = 0.95,                      # ROI overlay opacity
):
    """
    8-view categorical ROI surface plot with sulc/curv underlay:
      - Left hemi:  lateral, medial, anterior, posterior  (top row)
      - Right hemi: lateral, medial, anterior, posterior  (bottom row)

    left_cat/right_cat:
      int arrays with 0=background, 1..K categories.
    """
    from nilearn import plotting  # relies on nibabel+nilearn

    lh_surf = Path(lh_surf); rh_surf = Path(rh_surf)
    out_png = Path(out_png)

    # ---- discrete colormap: 0 transparent, 1..K colored
    n = len(roi_names)
    base = plt.cm.get_cmap("tab20")
    colors = [(0, 0, 0, 0.0)]  # background transparent
    for i in range(max(n, 1)):
        colors.append(base(i % base.N))
    lcmap = ListedColormap(colors)

    # ---- underlay (folds)
    L_bg = R_bg = None
    if underlay_dscalar is not None:
        underlay_dscalar = Path(underlay_dscalar)
        if underlay_dscalar.exists():
            L_bg, R_bg = _load_cifti_underlay_lr(underlay_dscalar)
        else:
            print(f"⚠ underlay_dscalar not found: {underlay_dscalar} (plotting without folds)")

    views = ["lateral", "medial", "anterior"]

    fig = plt.figure(figsize=(22, 10))

    # --- Top row: Left hemisphere 4 views
    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        plotting.plot_surf_roi(
            surf_mesh=str(lh_surf),
            roi_map=left_cat,
            bg_map=L_bg,                # None => no folds
            hemi="left",
            view=view,
            cmap=lcmap,
            colorbar=False,
            axes=ax,
            alpha=alpha,
            darkness=darkness,
            title=f"Left | {view}"
        )

    # --- Bottom row: Right hemisphere 4 views
    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 4, projection="3d")
        plotting.plot_surf_roi(
            surf_mesh=str(rh_surf),
            roi_map=right_cat,
            bg_map=R_bg,                # None => no folds
            hemi="right",
            view=view,
            cmap=lcmap,
            colorbar=False,
            axes=ax,
            alpha=alpha,
            darkness=darkness,
            title=f"Right | {view}"
        )

    # ---- legend (ROI -> color)
    patches = []
    for j, nm in enumerate(roi_names):
        patches.append(Patch(facecolor=lcmap(j + 1), edgecolor="black", label=f"{j+1:02d}. {nm}"))

    fig.suptitle(f"{atlas_name} | {level}-level | Categorical ROIs (8 views)", fontsize=16, y=0.99)
    fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.02, 0.5),
               frameon=True, ncol=1, fontsize=14, title="ROI colors")

    plt.tight_layout(rect=[0, 0, 0.84, 0.96])
    fig.savefig(out_png, dpi=900, bbox_inches="tight")
    plt.close(fig)
