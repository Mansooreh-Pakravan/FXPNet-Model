    # utilityFunctions_21.py
import numpy as np

def reshapeData(data):
    """
    ورودی: data با شکل (N_samples, T, C)
           (در اینجا C=21)
    خروجی: data_reshape با شکل (N_samples, C, T)
           مطابق انتظار Conv1d
    """
    n_samples, n_ts, n_channels = data.shape
    data_reshape = np.empty((n_samples, n_channels, n_ts), dtype=data.dtype)
    for i in range(n_samples):
        data_reshape[i] = data[i].T
    return data_reshape

import numpy as np

def prepare_data_sliding_window(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 256,
    step: int = 128,
    return_subj_index: bool = False,
):
    """
    data: (N_subj, T, N_roi)
    labels: (N_subj,)
    خروجی:
      X_win: (N_win, window_size, N_roi)
      y_win: (N_win,)
      subj_idx_win: (N_win,)  (اختیاری) اندیس سوژه‌ی هر پنجره (0..N_subj-1)

    نکته: این نسخه start را از 0 شروع می‌کند و تا T-window_size (شامل) جلو می‌رود.
    """
    N_subj, T, N_roi = data.shape

    X_list = []
    y_list = []
    s_list = []

    for s in range(N_subj):
        ts = data[s]  # (T, N_roi)
        y  = labels[s]

        # تعداد پنجره‌ها: start = 0..T-window_size (inclusive) با گام step
        for start in range(0, T - window_size + 1, step):
            X_list.append(ts[start:start + window_size, :])
            y_list.append(y)
            if return_subj_index:
                s_list.append(s)

    if len(X_list) == 0:
        # هیچ پنجره‌ای ساخته نشد
        X_win = np.zeros((0, window_size, N_roi), dtype=np.float32)
        y_win = np.zeros((0,), dtype=np.int64)
        if return_subj_index:
            subj_idx_win = np.zeros((0,), dtype=np.int64)
            return X_win, y_win, subj_idx_win
        return X_win, y_win

    X_win = np.stack(X_list, axis=0).astype(np.float32)
    y_win = np.array(y_list, dtype=np.int64)

    if return_subj_index:
        subj_idx_win = np.array(s_list, dtype=np.int64)
        return X_win, y_win, subj_idx_win

    return X_win, y_win

from pathlib import Path

def load_region_names_full(txt_path: Path):
    """
    فایل‌هایی مثل:
      medial area 8 L
      medial area 8 R
      ...
    یعنی هر خط = یک ROI name.
    """
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("//") or s.startswith("%"):
                continue
            names.append(s)
    return names


def build_roi_idx_to_name_from_fullnames(
    atlas_name: str,
    n_roi: int,
    names_dir: Path
):
    atlas_to_file = {
        "Brainnetome": "BNA246_region_names_full.txt",
        "Glasser": "GlasserFreesurfer_region_names_full.txt",
        "Gordon": "Gordon333_Tian_Subcortex_S1_3T_region_names_full.txt",
    }
    if atlas_name not in atlas_to_file:
        raise ValueError(f"atlas_name={atlas_name} not supported. Choose from {list(atlas_to_file.keys())}")

    txt_path = names_dir / atlas_to_file[atlas_name]
    if not txt_path.exists():
        raise FileNotFoundError(f"Region-names file not found: {txt_path}")

    names = load_region_names_full(txt_path)

    # حالت‌های رایج
    if len(names) == n_roi:
        roi_idx_to_name = {i: names[i] for i in range(n_roi)}
        meta = {"file": str(txt_path), "mode": "direct", "n_names": len(names)}
        return roi_idx_to_name, meta

    # اگر یک خط اضافی مثل Unknown داشته باشد
    if len(names) == n_roi + 1:
        first = names[0].lower()
        if "unknown" in first or "background" in first:
            names = names[1:]
            roi_idx_to_name = {i: names[i] for i in range(n_roi)}
            meta = {"file": str(txt_path), "mode": "drop-first-unknown", "n_names": len(names)+1}
            return roi_idx_to_name, meta

    raise ValueError(
        f"ROI names count mismatch for atlas={atlas_name}: n_roi={n_roi}, file_lines={len(names)}. "
        f"Check whether your ROI ordering matches the file: {txt_path}"
    )


def shorten_roi_name(name: str, max_len: int = 22):
    """
    برای قابل‌خواندن شدن برچسب‌های نمودار.
    مثال: 'area 4 (head and face region) L' → 'area 4 (head..) L'
    """
    s = name
    # کمی کوتاه‌سازی پرانتزهای طولانی
    if "(" in s and ")" in s and len(s) > max_len:
        s = s.replace("region", "reg.")
        s = s.replace(" and ", " & ")
    if len(s) > max_len:
        s = s[:max_len-2] + ".."
    return s

import torch
import torch.nn.functional as F

def roi_gate_regularizer(
    roi_w: torch.Tensor,
    l1_alpha: float = 1e-3,
    ent_beta: float = 1e-3,
):
    """
    roi_w: (B, C) خروجی sparsemax یا softmax (ترجیحاً sparsemax)
    - L1: برای sparse شدن (کم شدن تعداد ROIهای فعال)
    - Entropy: برای تیز شدن توزیع (ضد یکنواختی)
    """
    # L1
    l1 = roi_w.abs().mean()

    # Entropy
    p = roi_w / (roi_w.sum(dim=1, keepdim=True) + 1e-12)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=1).mean()

    return l1_alpha * l1 + ent_beta * ent


def schedule_reg(epoch: int, warmup_epochs: int = 5):
    """
    تا چند epoch اول فقط accuracy (CE) را یاد بگیر، بعد reg را روشن کن.
    """
    if epoch <= warmup_epochs:
        return 0.0, 0.0
    return 1e-3, 1e-3   # (l1_alpha, ent_beta) ← می‌توانی تیون کنی

def proto_losses(details, y, w_clust=1e-2, w_sep=1e-2, w_div=1e-3, w_sparse=1e-3):
    """
    details: خروجی return_details=True از مدل
    y: (B,) 0/1
    loss terms:
      - clustering: نزدیک شدن نمونه به پروتوی کلاس خودش
      - separation: دور شدن از پروتوی کلاس مخالف
      - diversity: دور شدن پروتوها از هم (تا تکراری نشوند)
      - sparsity: فعال شدن تعداد کم پروتو برای هر نمونه (توضیح کوتاه‌تر)
    """
    z = details["z"]                      # (B,d)
    mu = details["mu"]                    # (B,K)
    proto_class = details["proto_class"]  # (K,)
    P = details["prototypes"]             # (K,d)

    B, K = mu.shape

    # dist2 برای lossها (اگر proto_extra دارید می‌توانید dist2 را مستقیم بردارید)
    z2 = (z ** 2).sum(dim=1, keepdim=True)
    p2 = (P ** 2).sum(dim=1).view(1, -1)
    dist2 = z2 + p2 - 2 * (z @ P.t())     # (B,K)

    # masks
    y = y.view(-1, 1)                     # (B,1)
    mask_same = (proto_class.view(1, -1) == y).float()          # (B,K)
    mask_diff = 1.0 - mask_same

    # ---- 1) clustering: minimize min dist to same-class prototypes
    large = 1e9
    dist_same = dist2 + (1.0 - mask_same) * large
    min_same = dist_same.min(dim=1).values
    L_clust = min_same.mean()

    # ---- 2) separation: maximize min dist to opposite-class (=> minimize negative)
    dist_diff = dist2 + (1.0 - mask_diff) * large
    min_diff = dist_diff.min(dim=1).values
    # encourage min_diff to be large => penalize negative margin
    # ساده: L_sep = exp(-min_diff)  (کوچک‌تر اگر دور باشد)
    L_sep = torch.exp(-min_diff).mean()

    # ---- 3) diversity among prototypes (repulsion)
    # encourage prototypes to be apart: penalize exp(-||pi-pj||^2)
    # (K small => ok)
    Pn = F.normalize(P, dim=1)
    sim = Pn @ Pn.t()                      # cosine sim
    # remove diagonal
    sim = sim - torch.eye(K, device=sim.device) * sim
    L_div = (sim ** 2).mean()              # کوچک‌تر => پروتوها متفاوت‌تر

    # ---- 4) sparsity on mu: encourage few active prototypes
    # entropy-like: sum mu, or L1 on mu normalized
    mu_norm = mu / (mu.sum(dim=1, keepdim=True) + 1e-8)
    L_sparse = (mu_norm * mu_norm).sum(dim=1).mean()  # بزرگ‌تر => تیزتر (کم‌پروتو فعال)

    loss = w_clust * L_clust + w_sep * L_sep + w_div * L_div + w_sparse * (-L_sparse)
    # توجه: اگر می‌خواهید mu تیز شود، منفی L_sparse را کمینه می‌کنیم (یعنی L_sparse را بیشینه کنیم)
    return loss, {
        "L_clust": float(L_clust.detach().cpu()),
        "L_sep": float(L_sep.detach().cpu()),
        "L_div": float(L_div.detach().cpu()),
        "L_sparse": float(L_sparse.detach().cpu()),
    }
