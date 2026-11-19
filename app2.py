import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import label, find_objects, binary_dilation

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest-Loss Detection — Error Analysis",
    page_icon="🌳",
    layout="wide",
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
TILES_DIR = os.path.join(DATA_DIR, "tiles")
RGB_DIR   = os.path.join(DATA_DIR, "rgb")

# On Hugging Face Spaces the 2.8 GB test set isn't bundled with the app;
# fetch it once from the public dataset on first boot.
if not os.path.exists(os.path.join(DATA_DIR, "models.json")):
    from huggingface_hub import snapshot_download
    with st.spinner("First launch: downloading the test set (~2.8 GB). This takes a few minutes…"):
        snapshot_download(
            repo_id="mqraitem/Deforest-Analysis",
            repo_type="dataset",
            allow_patterns=["data/*"],
            local_dir=os.path.dirname(__file__),
        )

CLASS_NAMES  = ["No Loss", "Forest Loss"]
CLASS_COLORS = [
    (128, 128, 128),   # No Loss — gray
    (220,  20,  60),   # Forest Loss — crimson
]
MODEL_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
PHASE_SUFFIXES = ["before", "during_1", "during_2", "after"]
PHASE_LABELS   = ["Before", "During (1)", "During (2)", "After"]
MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Component-size analysis (README/tasks.md spec)
SIZE_BINS   = [10, 25, 100, 500, 1_000_000]
BIN_LABELS  = ["10–24", "25–99", "100–499", "500+"]
MIN_CC_SIZE = 10   # GT already has sub-10px components removed
CONF_LEVELS = [2, 3, 4]
REGION_PAD  = 16

# Tile loss-density stratification (report slide 9)
DENSITY_BINS   = [0, 0.01, 0.05, 0.10, 0.20, 1.0001]
DENSITY_LABELS = ["0–1%", "1–5%", "5–10%", "10–20%", "20–100%"]

# Alert-confidence stratification across all models (report slide 11)
ALL_CONF_LEVELS = [0, 2, 3, 4]
CONF_DISPLAY    = {0: "1", 2: "2", 3: "3", 4: "4"}


# ── Model / tile metadata ─────────────────────────────────────────────────────

@st.cache_data
def get_model_info():
    with open(os.path.join(DATA_DIR, "models.json")) as f:
        info = json.load(f)
    return info


@st.cache_data
def get_tiles_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "test_tiles.csv"))
    return df.set_index("tile_id", drop=False).sort_index()


def model_short(key: str) -> str:
    return get_model_info()[key]["display_name"]


def model_keys():
    return sorted(get_model_info().keys())


# ── Tile loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_tile(tile_id: str):
    path = os.path.join(TILES_DIR, f"{tile_id}.npz")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@st.cache_data
def load_rgb_frame(tile_id: str, phase_idx: int):
    suffix = PHASE_SUFFIXES[phase_idx]
    path = os.path.join(RGB_DIR, f"{tile_id}_input_{phase_idx + 1}_{suffix}.png")
    return np.array(Image.open(path))


def seed_position(tile: dict, model_key: str, seed: int = 0) -> int:
    seeds = tile[f"seeds__{model_key}"]
    matches = np.flatnonzero(seeds == seed)
    return int(matches[0]) if len(matches) else 0


def get_pred(tile: dict, model_key: str, seed: int = 0) -> np.ndarray:
    idx = seed_position(tile, model_key, seed)
    return tile[f"pred__{model_key}"][idx].astype(bool)


def get_all_seed_preds(tile: dict, model_key: str):
    """Returns list of (seed_id, pred_bool_array)."""
    seeds = tile[f"seeds__{model_key}"]
    preds = tile[f"pred__{model_key}"]
    return [(int(seeds[i]), preds[i].astype(bool)) for i in range(preds.shape[0])]


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy":  (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) else 0.0,
        "dice":      2 * tp / denom_dice if denom_dice else 0.0,
        "iou":       tp / denom_iou if denom_iou else 0.0,
    }


# ── Aggregate metrics (seed-averaged) ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_model_tile_metrics(model_key: str) -> pd.DataFrame:
    """Per-tile metrics for one model, averaged across that model's seeds."""
    rows = []
    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            preds = data[f"pred__{model_key}"]
            seeds = data[f"seeds__{model_key}"]
            per_seed = [binary_metrics(preds[i].astype(bool), gt) for i in range(preds.shape[0])]

        row = {"tile": tile_id, "n_seeds": len(seeds),
               "gt_loss_fraction": float(gt.mean())}
        for key in ("accuracy", "precision", "recall", "dice", "iou"):
            row[key] = float(np.mean([m[key] for m in per_seed]))
        row["pred_loss_fraction"] = float(np.mean(
            [(preds[i] > 0).mean() for i in range(preds.shape[0])]
        ))
        row["error_rate"] = 1.0 - row["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("tile", drop=False)


@st.cache_data(show_spinner=False)
def compute_confusion_binary(model_key: str) -> np.ndarray:
    """2x2 pixel-pooled confusion matrix, averaged across seeds."""
    seed_cms = None
    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            preds = data[f"pred__{model_key}"]
        if seed_cms is None:
            seed_cms = [np.zeros((2, 2), dtype=np.float64) for _ in range(preds.shape[0])]
        for i in range(preds.shape[0]):
            p = preds[i].astype(bool)
            seed_cms[i][1, 1] += int((p & gt).sum())
            seed_cms[i][1, 0] += int((~p & gt).sum())
            seed_cms[i][0, 1] += int((p & ~gt).sum())
            seed_cms[i][0, 0] += int((~p & ~gt).sum())
    return np.mean(seed_cms, axis=0)


@st.cache_data(show_spinner=False)
def compute_model_micro_metrics(model_key: str) -> pd.DataFrame:
    """Per-seed, whole-dataset-pooled metrics — the same method the project write-up uses
    ('aggregate over the entire dataset', page 4), reported per seed so the UI can show
    mean ± std across seeds like every 'Across Seeds' chart in the write-up. One row per seed."""
    seeds = get_model_info()[model_key]["seeds"]
    sums = [np.zeros(4) for _ in seeds]  # tp, fp, fn, tn

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            preds = data[f"pred__{model_key}"]
        for i in range(preds.shape[0]):
            p = preds[i].astype(bool)
            sums[i][0] += int((p & gt).sum())
            sums[i][1] += int((p & ~gt).sum())
            sums[i][2] += int((~p & gt).sum())
            sums[i][3] += int((~p & ~gt).sum())

    rows = []
    for seed, (tp, fp, fn, tn) in zip(seeds, sums):
        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn
        total = tp + fp + fn + tn
        rows.append({
            "seed": seed,
            "accuracy":       (tp + tn) / total if total else 0.0,
            "precision":      tp / (tp + fp) if (tp + fp) else 0.0,
            "recall":         tp / (tp + fn) if (tp + fn) else 0.0,
            "dice":           2 * tp / denom_dice if denom_dice else 0.0,
            "iou":            tp / denom_iou if denom_iou else 0.0,
            "pred_pos_share": (tp + fp) / total if total else 0.0,
            "true_pos_share": (tp + fn) / total if total else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_grouped_confusion(model_key: str, group_kind: str, low_thresh: float, high_thresh: float):
    """Pixel-pooled confusion split by cloud fraction or alert-confidence groups."""
    group_names = ["Overall", "Low", "Medium", "High"]
    n_seeds = len(get_model_info()[model_key]["seeds"])
    seed_cms = {g: [np.zeros((2, 2)) for _ in range(n_seeds)] for g in group_names}

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            preds = data[f"pred__{model_key}"]
            if group_kind == "cloud":
                summary = data["clarity"].astype(np.float32).mean(axis=0)
            else:
                summary = data["alert_conf"].astype(np.float32)

        masks = {
            "Overall": np.ones_like(gt, dtype=bool),
            "Low":     summary < low_thresh,
            "Medium":  (summary >= low_thresh) & (summary < high_thresh),
            "High":    summary >= high_thresh,
        }
        for i in range(preds.shape[0]):
            p = preds[i].astype(bool)
            for g, m in masks.items():
                seed_cms[g][i][1, 1] += int((p & gt & m).sum())
                seed_cms[g][i][1, 0] += int((~p & gt & m).sum())
                seed_cms[g][i][0, 1] += int((p & ~gt & m).sum())
                seed_cms[g][i][0, 0] += int((~p & ~gt & m).sum())

    return {g: np.mean(seed_cms[g], axis=0) for g in group_names}


# ── Component helpers ─────────────────────────────────────────────────────────

def connected_components(mask: np.ndarray):
    labeled, n = label(mask, structure=np.ones((3, 3)))
    return labeled, n, find_objects(labeled)


@st.cache_data(show_spinner=False)
def compute_region_df() -> pd.DataFrame:
    """GT connected components x models: best-overlap matched predicted component metrics."""
    models = model_keys()
    rows = []

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            alert = data["alert_conf"]
            preds = {m: data[f"pred__{m}"] for m in models}
            seeds = {m: data[f"seeds__{m}"] for m in models}

        gt_labeled, n_gt, gt_slices = connected_components(gt)
        if n_gt == 0:
            continue

        pred_labeled = {}
        for m in models:
            avg_pred = np.mean([preds[m][i].astype(bool) for i in range(preds[m].shape[0])], axis=0) >= 0.5
            pred_labeled[m] = connected_components(avg_pred)

        for rid in range(1, n_gt + 1):
            region_mask = gt_labeled == rid
            size_px = int(region_mask.sum())
            if size_px < MIN_CC_SIZE:
                continue
            s = gt_slices[rid - 1]
            dominant_conf = int(np.bincount(alert[region_mask]).argmax())

            row = {
                "tile": tile_id, "region_id": rid, "size_px": size_px,
                "row_min": s[0].start, "row_max": s[0].stop,
                "col_min": s[1].start, "col_max": s[1].stop,
                "alert_conf": dominant_conf,
            }
            for m in models:
                p_labeled, n_p, p_slices = pred_labeled[m]
                overlap_labels = p_labeled[region_mask]
                overlap_labels = overlap_labels[overlap_labels > 0]
                if len(overlap_labels) > 0:
                    best_pid = int(np.bincount(overlap_labels).argmax())
                    pred_region = p_labeled == best_pid
                else:
                    pred_region = np.zeros_like(region_mask)
                tp = int((region_mask & pred_region).sum())
                fp = int((~region_mask & pred_region).sum())
                fn = int((region_mask & ~pred_region).sum())
                denom = 2 * tp + fp + fn
                row[f"dice_{m}"]     = 2 * tp / denom if denom else 0.0
                row[f"detected_{m}"] = float(tp > 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["size_bin"] = pd.cut(df["size_px"], bins=SIZE_BINS, labels=BIN_LABELS, right=False)
    return df


@st.cache_data(show_spinner=False)
def compute_fragmentation_df() -> pd.DataFrame:
    """Per-tile, per-model prediction-shape stats (component count, size, bbox fill)."""
    models = model_keys()
    rows = []
    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            for m in models:
                preds = data[f"pred__{m}"]
                avg_pred = np.mean([preds[i].astype(bool) for i in range(preds.shape[0])], axis=0) >= 0.5
                labeled, n, slices = connected_components(avg_pred)
                sizes, fills = [], []
                for rid in range(1, n + 1):
                    region = labeled == rid
                    size = int(region.sum())
                    if size < MIN_CC_SIZE:
                        continue
                    s = slices[rid - 1]
                    h = s[0].stop - s[0].start
                    w = s[1].stop - s[1].start
                    sizes.append(size)
                    fills.append(size / (h * w) if h * w else 0.0)
                rows.append({
                    "tile": tile_id, "model": m,
                    "n_components": len(sizes),
                    "mean_size": float(np.mean(sizes)) if sizes else 0.0,
                    "mean_bbox_fill": float(np.mean(fills)) if fills else np.nan,
                    "total_pred_px": int(avg_pred.sum()),
                })
    return pd.DataFrame(rows)


# ── Density / confidence stratification (all models) & ensembling ───────────

@st.cache_data(show_spinner=False)
def compute_density_stratified_dice() -> pd.DataFrame:
    """Micro-Dice (pooled TP/FP/FN) per model, stratified by tile-level GT loss density, seed-averaged."""
    models = model_keys()
    seed_counts = {m: len(model_info_all[m]["seeds"]) for m in models}
    sums = {m: {b: [np.zeros(3) for _ in range(seed_counts[m])] for b in DENSITY_LABELS} for m in models}
    tile_counts = {b: 0 for b in DENSITY_LABELS}

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            loss_frac = float(gt.mean())
            b = pd.cut([loss_frac], bins=DENSITY_BINS, labels=DENSITY_LABELS, right=False)[0]
            if pd.isna(b):
                continue
            tile_counts[b] += 1
            for m in models:
                preds = data[f"pred__{m}"]
                for i in range(preds.shape[0]):
                    p = preds[i].astype(bool)
                    sums[m][b][i][0] += int((p & gt).sum())
                    sums[m][b][i][1] += int((p & ~gt).sum())
                    sums[m][b][i][2] += int((~p & gt).sum())

    rows = []
    for m in models:
        for b in DENSITY_LABELS:
            seed_dices = []
            for tp, fp, fn in sums[m][b]:
                denom = 2 * tp + fp + fn
                seed_dices.append(2 * tp / denom if denom else np.nan)
            seed_dices = np.array(seed_dices, dtype=float)
            rows.append({
                "model": m, "bin": b,
                "mean_dice": float(np.nanmean(seed_dices)) * 100,
                "std_dice": float(np.nanstd(seed_dices)) * 100 if len(seed_dices) > 1 else 0.0,
                "n_tiles": tile_counts[b],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_confidence_stratified_dice_all_models() -> pd.DataFrame:
    """Micro-Dice per model, stratified by raw GLAD-S2 alert-confidence value, seed-averaged."""
    models = model_keys()
    seed_counts = {m: len(model_info_all[m]["seeds"]) for m in models}
    sums = {m: {c: [np.zeros(3) for _ in range(seed_counts[m])] for c in ALL_CONF_LEVELS} for m in models}
    gt_pixels_per_conf = {c: 0 for c in ALL_CONF_LEVELS}
    total_gt_pixels = 0

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            alert = data["alert_conf"]
            total_gt_pixels += int(gt.sum())
            conf_masks = {c: alert == c for c in ALL_CONF_LEVELS}
            for c in ALL_CONF_LEVELS:
                gt_pixels_per_conf[c] += int((gt & conf_masks[c]).sum())
            for m in models:
                preds = data[f"pred__{m}"]
                for i in range(preds.shape[0]):
                    p = preds[i].astype(bool)
                    correct_gt = p & gt
                    wrong_fp = p & ~gt
                    wrong_fn = ~p & gt
                    for c in ALL_CONF_LEVELS:
                        cm = conf_masks[c]
                        sums[m][c][i][0] += int((correct_gt & cm).sum())
                        sums[m][c][i][1] += int((wrong_fp & cm).sum())
                        sums[m][c][i][2] += int((wrong_fn & cm).sum())

    rows = []
    for m in models:
        for c in ALL_CONF_LEVELS:
            seed_dices = []
            for tp, fp, fn in sums[m][c]:
                denom = 2 * tp + fp + fn
                seed_dices.append(2 * tp / denom if denom else np.nan)
            seed_dices = np.array(seed_dices, dtype=float)
            rows.append({
                "model": m, "conf": c,
                "mean_dice": float(np.nanmean(seed_dices)) * 100,
                "std_dice": float(np.nanstd(seed_dices)) * 100 if len(seed_dices) > 1 else 0.0,
            })
    df = pd.DataFrame(rows)
    df["gt_share"] = df["conf"].map(lambda c: gt_pixels_per_conf[c] / total_gt_pixels if total_gt_pixels else 0.0)
    return df


@st.cache_data(show_spinner=False)
def compute_ensemble_dice(model_a: str, model_b: str) -> dict:
    """Per-seed, whole-dataset-pooled Dice for model_a alone, model_b alone, and a post-hoc
    mean-probability ensemble — paired seed-by-seed, then mean ± std across seeds. Same method
    as compute_model_micro_metrics / the project write-up's 'Across Seeds' charts, rather than
    averaging each model's probability across seeds before pooling."""
    n_pairs = min(len(get_model_info()[model_a]["seeds"]), len(get_model_info()[model_b]["seeds"]))
    sums_a = [np.zeros(3) for _ in range(n_pairs)]  # tp, fp, fn
    sums_b = [np.zeros(3) for _ in range(n_pairs)]
    sums_e = [np.zeros(3) for _ in range(n_pairs)]

    for tile_id in get_tiles_df().index:
        path = os.path.join(TILES_DIR, f"{tile_id}.npz")
        with np.load(path) as data:
            gt = data["gt"].astype(bool)
            prob_a = data[f"prob_change__{model_a}"].astype(np.float32)
            prob_b = data[f"prob_change__{model_b}"].astype(np.float32)

        for i in range(n_pairs):
            pred_a = prob_a[i] >= 0.5
            pred_b = prob_b[i] >= 0.5
            pred_e = ((prob_a[i] + prob_b[i]) / 2) >= 0.5
            for pred, sums in ((pred_a, sums_a), (pred_b, sums_b), (pred_e, sums_e)):
                sums[i][0] += int((pred & gt).sum())
                sums[i][1] += int((pred & ~gt).sum())
                sums[i][2] += int((~pred & gt).sum())

    def dice_stats(sums):
        dices = []
        for tp, fp, fn in sums:
            denom = 2 * tp + fp + fn
            dices.append(2 * tp / denom if denom else 0.0)
        dices = np.array(dices)
        return float(dices.mean()) * 100, float(dices.std(ddof=0)) * 100

    a_mean, a_std = dice_stats(sums_a)
    b_mean, b_std = dice_stats(sums_b)
    e_mean, e_std = dice_stats(sums_e)

    return {
        "a_dice": a_mean, "a_std": a_std,
        "b_dice": b_mean, "b_std": b_std,
        "ensemble_dice": e_mean, "ensemble_std": e_std,
    }


# ── Image helpers ─────────────────────────────────────────────────────────────

def mask_to_rgba(mask: np.ndarray, color, alpha=200):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha
    return rgba


def error_map_rgba(pred: np.ndarray, gt: np.ndarray):
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    correct = pred == gt
    rgba[correct] = (128, 128, 128, 120)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    rgba[tp] = (0, 200, 0, 220)
    rgba[fp] = (255, 165, 0, 220)
    rgba[fn] = (220, 0, 0, 220)
    return rgba


def legend_patches():
    return [
        mpatches.Patch(facecolor=[c / 255 for c in CLASS_COLORS[i]],
                       edgecolor="black", linewidth=0.4, label=CLASS_NAMES[i])
        for i in range(2)
    ]


def plot_cm(cm: np.ndarray, ax, title: str, normalised: bool = True):
    if normalised:
        data = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = lambda v: f"{v:.1%}"
        vmax = 1.0
    else:
        data = cm.astype(float)
        fmt = lambda v: f"{v:,.0f}"
        vmax = None

    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    threshold = data.max() * 0.55
    for i in range(2):
        for j in range(2):
            ax.text(j, i, fmt(data[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > threshold else "black")
    return im


# ── Sidebar ───────────────────────────────────────────────────────────────────

model_info_all = get_model_info()
models = model_keys()
tiles_df = get_tiles_df()

with st.sidebar:
    st.title("Settings")

    selected_model = st.selectbox("Active model", models, format_func=model_short)
    n_seeds_selected = len(model_info_all[selected_model]["seeds"])
    st.caption(
        f"{n_seeds_selected} seed{'s' if n_seeds_selected != 1 else ''} available "
        f"({', '.join(str(s) for s in model_info_all[selected_model]['seeds'])}). "
        "Metrics below are averaged across seeds; the tile viewer shows one seed at a time."
    )

    st.divider()
    st.subheader("Tile filters")
    sort_by = st.selectbox(
        "Sort tiles by",
        ["Name (A→Z)", "Error Rate (high→low)", "Dice (low→high)", "GT loss fraction (high→low)"],
    )
    min_error = st.slider("Min error rate", 0.0, 1.0, 0.0, 0.01,
                          help="Only show tiles whose pixel error rate ≥ this value")

    st.divider()
    st.subheader("Cloud grouping")
    st.caption("Cloud fraction = share of the 4 observation dates flagged cloudy/shadow/nodata per pixel.")
    cloud_thresh = st.slider("Cloud fraction thresholds", 0.0, 1.0, (0.25, 0.5), 0.25,
                              help="Low < left · left ≤ Medium < right · High ≥ right")

    st.divider()
    st.subheader("Alert-confidence grouping")
    st.caption("GLAD-S2 alert confidence values are 0 (no alert), 2, 3, 4 (increasing confidence).")
    conf_thresh = st.slider("Confidence thresholds", 0, 4, (2, 3), 1,
                             help="Low < left · left ≤ Medium < right · High ≥ right")

    st.divider()
    st.subheader("Colour legend")
    for i, (name, (r, g, b)) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        st.markdown(
            f'<span style="background:rgb({r},{g},{b});padding:2px 10px;'
            f'border-radius:4px;color:white;font-weight:600">&nbsp;{name}&nbsp;</span>',
            unsafe_allow_html=True,
        )


# ── Load & filter tiles ───────────────────────────────────────────────────────

metrics_df = compute_model_tile_metrics(selected_model)
filtered_df = metrics_df[metrics_df["error_rate"] >= min_error].copy()

if sort_by == "Error Rate (high→low)":
    filtered_df = filtered_df.sort_values("error_rate", ascending=False)
elif sort_by == "Dice (low→high)":
    filtered_df = filtered_df.sort_values("dice", ascending=True)
elif sort_by == "GT loss fraction (high→low)":
    filtered_df = filtered_df.sort_values("gt_loss_fraction", ascending=False)
else:
    filtered_df = filtered_df.sort_index()

tile_names = list(filtered_df.index)

# ── Page title ────────────────────────────────────────────────────────────────

st.title("Forest-Loss Detection — Error Analysis Tool")
st.caption(
    "Near-real-time forest-loss detection from four Harmonized Landsat/Sentinel-2 (HLS) "
    "observations (before → during → during → after). Ground truth is derived from GLAD-S2 "
    "alerts; 588 held-out 224×224 test tiles, 6 candidate models."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Confusion Matrix", "Tile Viewer", "Summary Stats", "Model Comparison",
    "Component-Size Analysis", "Prediction Shape", "Seasonality",
    "Stratified & Ensemble", "Multi-Model Gallery", "Report Highlights",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Confusion Matrix (overall + cloud/confidence-grouped)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header(f"Confusion Matrix — {model_short(selected_model)}")

    with st.expander("Overall confusion matrix (all pixels, seed-averaged)", expanded=True):
        cm_overall = compute_confusion_binary(selected_model)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw pixel counts (mean over seeds)")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            plot_cm(cm_overall, ax, "Rows = true, Cols = predicted", normalised=False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.subheader("Row-normalised (recall per class)")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            plot_cm(cm_overall, ax, "Each row sums to 1", normalised=True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Confusion matrices by cloud level")
    with st.spinner("Computing cloud-grouped confusion matrices…"):
        cloud_cms = compute_grouped_confusion(selected_model, "cloud", cloud_thresh[0], cloud_thresh[1])
    cols = st.columns(4)
    for col, group in zip(cols, ["Overall", "Low", "Medium", "High"]):
        cm_g = cloud_cms[group]
        total = float(cm_g.sum())
        with col:
            if total == 0:
                st.warning(f"**{group} cloud**\n\nNo pixels in this group.")
            else:
                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                plot_cm(cm_g, ax, f"{group} cloud\n({total:,.0f} px)", normalised=True)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Confusion matrices by alert-confidence level")
    with st.spinner("Computing confidence-grouped confusion matrices…"):
        conf_cms = compute_grouped_confusion(selected_model, "conf", conf_thresh[0], conf_thresh[1])
    cols = st.columns(4)
    for col, group in zip(cols, ["Overall", "Low", "Medium", "High"]):
        cm_g = conf_cms[group]
        total = float(cm_g.sum())
        with col:
            if total == 0:
                st.warning(f"**{group} confidence**\n\nNo pixels in this group.")
            else:
                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                plot_cm(cm_g, ax, f"{group} confidence\n({total:,.0f} px)", normalised=True)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Key insights")
    st.caption(
        "Computed the same way as the project write-up: pool TP/FP/FN/TN over every pixel in "
        "every tile for each seed, then report mean ± std across seeds — not an average of "
        "per-tile scores. These numbers are directly comparable to the write-up's slides."
    )
    micro = compute_model_micro_metrics(selected_model)
    recall_mean, recall_std = float(micro["recall"].mean()), float(micro["recall"].std(ddof=0))
    precision_mean, precision_std = float(micro["precision"].mean()), float(micro["precision"].std(ddof=0))
    dice_mean, dice_std = float(micro["dice"].mean()), float(micro["dice"].std(ddof=0))
    pred_share_mean = float(micro["pred_pos_share"].mean())
    true_share_mean = float(micro["true_pos_share"].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall (forest loss)", f"{recall_mean:.1%}", f"±{recall_std:.1%} across seeds")
    c2.metric("Precision (forest loss)", f"{precision_mean:.1%}", f"±{precision_std:.1%} across seeds")
    c3.metric("Dice", f"{dice_mean:.3f}", f"±{dice_std:.3f} across seeds")
    st.info(
        f"**Predicted loss share:** {pred_share_mean:.2%} of all pixels are predicted as "
        f"forest loss, vs. a true loss share of {true_share_mean:.2%}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tile Viewer
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Interactive Tile Viewer")
    st.caption("A new, per-tile exploratory view — not part of the original write-up.")

    if not tile_names:
        st.warning("No tiles match the current filters. Lower the min error rate threshold in the sidebar.")
    else:
        if "tile_idx" not in st.session_state:
            st.session_state.tile_idx = 0
        st.session_state.tile_idx = min(st.session_state.tile_idx, len(tile_names) - 1)

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 3, 1])
        with nav1:
            if st.button("Previous", width='stretch'):
                if st.session_state.tile_idx > 0:
                    st.session_state.tile_idx -= 1
        with nav2:
            if st.button("Next", width='stretch'):
                if st.session_state.tile_idx < len(tile_names) - 1:
                    st.session_state.tile_idx += 1
        with nav3:
            chosen = st.selectbox("Jump to tile", tile_names, index=st.session_state.tile_idx,
                                   label_visibility="collapsed")
            st.session_state.tile_idx = tile_names.index(chosen)
        with nav4:
            available_seeds = model_info_all[selected_model]["seeds"]
            chosen_seed = st.selectbox("Seed", available_seeds, label_visibility="collapsed")

        current_tile = tile_names[st.session_state.tile_idx]
        st.markdown(f"**Tile {st.session_state.tile_idx + 1} / {len(tile_names)}** — `{current_tile}`")

        tile = load_tile(current_tile)
        gt = tile["gt"].astype(bool)
        pred = get_pred(tile, selected_model, chosen_seed)
        tm = metrics_df.loc[current_tile]
        row_meta = tiles_df.loc[current_tile]

        m = binary_metrics(pred, gt)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Dice", f"{m['dice']:.3f}")
        c3.metric("Precision", f"{m['precision']:.3f}")
        c4.metric("Recall", f"{m['recall']:.3f}")
        c5.metric("GT loss fraction", f"{tm['gt_loss_fraction']:.2%}")

        st.caption(
            f"Observations — Before: {row_meta['t1_timestamp']} · During: {row_meta['t2_timestamp']} / "
            f"{row_meta['t3_timestamp']} · After: {row_meta['t4_timestamp']}  "
            f"(MGRS {row_meta['t1_mgrs']}, lat {row_meta['latitude']:.2f}, lon {row_meta['longitude']:.2f})"
        )

        st.subheader("Temporal RGB inputs")
        frames = [load_rgb_frame(current_tile, i) for i in range(4)]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        for ax, img, label_txt in zip(axes, frames, PHASE_LABELS):
            ax.imshow(img); ax.set_title(label_txt, fontweight="bold", fontsize=10); ax.axis("off")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Ground truth · prediction · error map")
        gt_rgba = mask_to_rgba(gt, CLASS_COLORS[1])
        pred_rgba = mask_to_rgba(pred, CLASS_COLORS[1])
        err_rgba = error_map_rgba(pred, gt)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        panels = [
            (frames[-1], "After (satellite)"),
            (gt_rgba, "Ground Truth (forest loss)"),
            (pred_rgba, f"Prediction — seed {chosen_seed}"),
            (err_rgba, "Error map (green=TP, red=FN, orange=FP)"),
        ]
        for ax, (img, title) in zip(axes, panels):
            is_overlay = img.ndim == 3 and img.shape[-1] == 4
            if is_overlay:
                ax.imshow(frames[-1])
            ax.imshow(img)
            ax.set_title(title, fontweight="bold", fontsize=9); ax.axis("off")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        with st.expander("Cloud info for this tile"):
            clarity = tile["clarity"]
            cl1, cl2 = st.columns([1, 2])
            with cl1:
                coverage = clarity.mean(axis=(1, 2)) * 100
                fig, ax = plt.subplots(figsize=(4, 3))
                bars = ax.bar(PHASE_LABELS, coverage, color="#5B9BD5", edgecolor="black", linewidth=0.4)
                ax.set_ylim(0, 100); ax.set_ylabel("% cloudy/nodata pixels")
                ax.set_title("Cloud/nodata coverage per observation", fontsize=9)
                for bar, val in zip(bars, coverage):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{val:.0f}%", ha="center", va="bottom", fontsize=7)
                plt.xticks(rotation=20, ha="right", fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with cl2:
                cloud_frac = clarity.astype(np.float32).mean(axis=0)
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                overlay = ax.imshow(cloud_frac, cmap="Blues", vmin=0, vmax=1)
                ax.set_title("Cloud fraction per pixel (across 4 obs)", fontsize=9); ax.axis("off")
                plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption(f"Mean cloud fraction across tile: **{float(cloud_frac.mean()):.1%}**")

        with st.expander("Alert-confidence map"):
            alert = tile["alert_conf"]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            overlay = ax.imshow(alert, cmap="OrRd", vmin=0, vmax=4)
            ax.set_title("GLAD-S2 alert confidence (0, 2, 3, 4)", fontsize=9); ax.axis("off")
            plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with st.expander("Boundary analysis"):
            st.caption("Are errors concentrated at the edges of loss patches or in their interior?")
            is_wrong = pred != gt
            dilated = binary_dilation(gt, iterations=1)
            boundary = dilated & ~gt
            boundary_errors = int(np.sum(is_wrong & boundary))
            interior_errors = int(np.sum(is_wrong & ~boundary))
            total_errors = boundary_errors + interior_errors
            if total_errors > 0:
                b_pct = boundary_errors / total_errors
                bc1, bc2 = st.columns(2)
                bc1.metric("Boundary errors", f"{boundary_errors:,}", f"{b_pct:.1%} of all wrong pixels")
                bc2.metric("Interior errors", f"{interior_errors:,}", f"{1 - b_pct:.1%} of all wrong pixels")
            else:
                st.success("No prediction errors on this tile!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Summary Statistics — {model_short(selected_model)}")
    st.caption(
        "New, per-tile cut not in the original write-up: one score computed per tile, then "
        "plotted as a distribution — not the same as the pooled 'Dice' reported in the "
        "Confusion Matrix and Model Comparison tabs."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dice score distribution across tiles")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(metrics_df["dice"], bins=25, color="#4C72B0", edgecolor="black", linewidth=0.4, alpha=0.85)
        mean_dice = float(metrics_df["dice"].mean())
        ax.axvline(mean_dice, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_dice:.3f}")
        ax.set_xlabel("Dice score"); ax.set_ylabel("Number of tiles"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Error rate distribution across tiles")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(metrics_df["error_rate"], bins=25, color="#C44E52", edgecolor="black", linewidth=0.4, alpha=0.85)
        mean_err = float(metrics_df["error_rate"].mean())
        ax.axvline(mean_err, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_err:.2%}")
        ax.set_xlabel("Error rate"); ax.set_ylabel("Number of tiles"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("GT loss fraction vs. Dice")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(metrics_df["gt_loss_fraction"], metrics_df["dice"], alpha=0.4, s=20, color="#55A868")
    ax.set_xlabel("GT loss fraction (per tile)"); ax.set_ylabel("Dice")
    ax.set_title("Do smaller disturbances score worse?")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("10 worst-performing tiles")
    worst = metrics_df.sort_values("dice").head(10)
    st.dataframe(
        worst[["accuracy", "dice", "precision", "recall", "gt_loss_fraction", "pred_loss_fraction"]]
        .style.format("{:.3f}"),
        width='stretch',
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Model Comparison")
    st.caption("Comparing all 6 models across the same 588 tiles (metrics averaged over each model's seeds).")

    METRIC_KEYS = ("accuracy", "dice", "precision", "recall", "iou")
    model_micro = {m: compute_model_micro_metrics(m) for m in models}
    model_stats = {m: {k: float(model_micro[m][k].mean()) for k in METRIC_KEYS} for m in models}
    model_stats_std = {m: {k: float(model_micro[m][k].std(ddof=0)) for k in METRIC_KEYS} for m in models}

    short_labels = [model_short(m) for m in models]

    st.subheader("Overall metrics by model")
    st.caption(
        "Same evaluation method as the project write-up: TP/FP/FN/TN pooled over the entire "
        "test set for each seed, then mean ± std across seeds (error bars below) — not a "
        "per-tile average. Directly comparable to the write-up's 'Across Seeds' slides."
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, key, label_txt in zip(axes, ["accuracy", "dice", "precision", "recall"],
                                  ["Accuracy", "Dice", "Precision", "Recall"]):
        vals = [model_stats[m][key] for m in models]
        errs = [model_stats_std[m][key] for m in models]
        bars = ax.bar(short_labels, vals, yerr=errs, capsize=3, color=MODEL_PALETTE, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1); ax.set_title(label_txt, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    best_dice_model = max(model_stats, key=lambda m: model_stats[m]["dice"])
    st.info(
        f"**Best overall Dice:** `{model_short(best_dice_model)}` "
        f"({model_stats[best_dice_model]['dice']:.3f} ± {model_stats_std[best_dice_model]['dice']:.3f})"
    )

    st.subheader("Summary table")
    table_rows = []
    for m in models:
        row = {"Model": model_short(m), "Seeds": len(model_info_all[m]["seeds"])}
        row.update({
            k.capitalize(): f"{model_stats[m][k]:.3f} ± {model_stats_std[m][k]:.3f}"
            for k in METRIC_KEYS
        })
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

    st.divider()
    st.subheader("Head-to-head: per-tile Dice (pick two models)")
    st.caption(
        "This view is per-tile (one Dice score per tile, plotted as a dot) — a new cut not in "
        "the original write-up, not the same 'Dice' as the pooled numbers above."
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        model_a = st.selectbox("Model A", models, index=0, format_func=model_short, key="cmp_a")
    with mc2:
        model_b = st.selectbox("Model B", models, index=1, format_func=model_short, key="cmp_b")

    if model_a != model_b:
        da = compute_model_tile_metrics(model_a)["dice"]
        db = compute_model_tile_metrics(model_b)["dice"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(da, db, alpha=0.4, s=20, color="#4C72B0")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Equal performance")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Dice — {model_short(model_a)}"); ax.set_ylabel(f"Dice — {model_short(model_b)}")
        ax.set_title("Each dot = one tile\nAbove diagonal → model B better")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

        a_wins = int((da > db).sum()); b_wins = int((db > da).sum()); ties = len(da) - a_wins - b_wins
        st.caption(f"{model_short(model_a)} wins on **{a_wins}** tiles · "
                   f"{model_short(model_b)} wins on **{b_wins}** tiles · Ties: **{ties}**")
    else:
        st.info("Select two different models to compare.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Component-Size Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Component-Size Analysis")
    st.caption(
        "For each 8-connected ground-truth forest-loss component, the predicted component with "
        "greatest overlap is matched and scored. Does detection improve for larger disturbances? "
        "New, component-level cut not in the original write-up."
    )

    with st.spinner("Computing per-component overlap across all tiles and models… (cached after first run)"):
        region_df = compute_region_df()

    if region_df.empty:
        st.error("No component data found.")
    else:
        r_tab1, r_tab2, r_tab3 = st.tabs(["Detection vs. Size", "By Alert-Confidence", "Component Viewer"])

        with r_tab1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, metric_prefix, title in zip(axes, ["detected_", "dice_"],
                                                 ["Detection rate (any overlap)", "Mean component Dice"]):
                for m, color in zip(models, MODEL_PALETTE):
                    col = f"{metric_prefix}{m}"
                    means = region_df.groupby("size_bin", observed=True)[col].mean().reindex(BIN_LABELS)
                    counts = region_df.groupby("size_bin", observed=True)[col].count().reindex(BIN_LABELS)
                    valid = means.notna()
                    xs = [i for i, v in enumerate(valid) if v]
                    ys = means[valid].values
                    if xs:
                        ax.plot(xs, ys, marker="o", linewidth=2, markersize=6, color=color, label=model_short(m))
                ax.set_xticks(range(len(BIN_LABELS))); ax.set_xticklabels(BIN_LABELS, rotation=20, ha="right")
                ax.set_ylim(0, 1); ax.set_xlabel("GT component size (pixels)"); ax.set_title(title, fontweight="bold")
                ax.grid(axis="y", linestyle=":", alpha=0.5); ax.legend(fontsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            counts_by_bin = region_df.groupby("size_bin", observed=True).size().reindex(BIN_LABELS).fillna(0).astype(int)
            st.caption("Components per size bin: " + " · ".join(f"{b}: {counts_by_bin[b]}" for b in BIN_LABELS))

            st.subheader("Best model per size bin (mean component Dice)")
            for b in BIN_LABELS:
                bdf = region_df[region_df["size_bin"] == b]
                if bdf.empty:
                    continue
                scored = {m: bdf[f"dice_{m}"].mean() for m in models}
                best_m = max(scored, key=scored.get)
                st.markdown(f"- **{b} px**: `{model_short(best_m)}` — mean Dice {scored[best_m]:.3f} "
                            f"({len(bdf)} components)")

        with r_tab2:
            st.caption("Stratified by dominant GLAD-S2 alert confidence within each GT component (2, 3, 4).")
            fig, axes = plt.subplots(1, len(CONF_LEVELS), figsize=(5 * len(CONF_LEVELS), 4.5), sharey=True)
            for ax, conf in zip(axes, CONF_LEVELS):
                cdf = region_df[region_df["alert_conf"] == conf]
                vals = [cdf[f"dice_{m}"].mean() if not cdf.empty else 0.0 for m in models]
                bars = ax.bar(short_labels := [model_short(m) for m in models], vals,
                              color=MODEL_PALETTE, edgecolor="black", linewidth=0.4)
                ax.set_ylim(0, 1); ax.set_title(f"Confidence {conf}  (n={len(cdf)})", fontweight="bold")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with r_tab3:
            st.caption("Components where models disagree most (highest Dice spread across models).")
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                filter_bin_r = st.selectbox("Size bin", ["All"] + BIN_LABELS, key="region_bin")
            with vc2:
                filter_conf_r = st.selectbox("Alert confidence", ["All"] + CONF_LEVELS, key="region_conf")
            with vc3:
                top_n = st.slider("# components to show", 3, 20, 8, 1, key="region_n")

            view_df = region_df.copy()
            if filter_bin_r != "All":
                view_df = view_df[view_df["size_bin"] == filter_bin_r]
            if filter_conf_r != "All":
                view_df = view_df[view_df["alert_conf"] == filter_conf_r]

            dice_cols = [f"dice_{m}" for m in models]
            view_df = view_df.copy()
            view_df["dice_spread"] = view_df[dice_cols].max(axis=1) - view_df[dice_cols].min(axis=1)
            top_regions = view_df.nlargest(top_n, "dice_spread")

            if top_regions.empty:
                st.warning("No components match the current filters.")
            else:
                tile_cache = {}

                def _get_tile_arrays(tile_id):
                    if tile_id not in tile_cache:
                        t = load_tile(tile_id)
                        tile_cache[tile_id] = t
                    return tile_cache[tile_id]

                for _, rrow in top_regions.iterrows():
                    tname = rrow["tile"]; rid = int(rrow["region_id"])
                    t = _get_tile_arrays(tname)
                    gt_t = t["gt"].astype(bool)
                    after_rgb = load_rgb_frame(tname, 3)
                    H, W = gt_t.shape

                    labeled_r, _, _ = connected_components(gt_t)
                    rmask = labeled_r == rid

                    r0 = max(0, int(rrow["row_min"]) - REGION_PAD); r1 = min(H, int(rrow["row_max"]) + REGION_PAD)
                    c0 = max(0, int(rrow["col_min"]) - REGION_PAD); c1 = min(W, int(rrow["col_max"]) + REGION_PAD)

                    rgb_crop = after_rgb[r0:r1, c0:c1]
                    mask_crop = rmask[r0:r1, c0:c1]
                    border = binary_dilation(mask_crop) & ~mask_crop
                    gt_rgba = mask_to_rgba(mask_crop, CLASS_COLORS[1])
                    gt_rgba[border] = (255, 255, 255, 255)

                    n_panels = 2 + len(models)
                    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.8))
                    axes[0].imshow(rgb_crop); axes[0].set_title("Satellite (after)", fontsize=8, fontweight="bold"); axes[0].axis("off")
                    axes[1].imshow(rgb_crop); axes[1].imshow(gt_rgba); axes[1].set_title("Ground Truth", fontsize=8, fontweight="bold"); axes[1].axis("off")

                    for ax, m, color in zip(axes[2:], models, MODEL_PALETTE):
                        avg_pred = np.mean([t[f"pred__{m}"][i].astype(bool) for i in range(t[f"pred__{m}"].shape[0])], axis=0) >= 0.5
                        pcrop = avg_pred[r0:r1, c0:c1]
                        prgba = mask_to_rgba(pcrop, CLASS_COLORS[1])
                        prgba[border] = (255, 255, 255, 255)
                        ax.imshow(rgb_crop); ax.imshow(prgba)
                        ax.set_title(f"{model_short(m)}\nDice={rrow[f'dice_{m}']:.2f}", fontsize=7, fontweight="bold", color=color)
                        ax.axis("off")

                    dice_vals = [rrow[f"dice_{m}"] for m in models]
                    fig.suptitle(
                        f"{tname} | component #{rid} | {int(rrow['size_px'])} px | "
                        f"conf={int(rrow['alert_conf'])} | Dice spread: {max(dice_vals) - min(dice_vals):.2f}",
                        fontsize=9, y=1.03,
                    )
                    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Prediction Shape / Fragmentation
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Prediction-Shape Personality")
    st.caption(
        "Do models make spatially different kinds of predictions even when overall Dice is similar? "
        "Component count, size, and bounding-box fill (compactness proxy) per tile, per model. "
        "New analysis, not in the original write-up."
    )

    with st.spinner("Computing prediction-shape statistics… (cached after first run)"):
        frag_df = compute_fragmentation_df()

    st.subheader("Component count per tile, by model")
    fig, ax = plt.subplots(figsize=(9, 5))
    data_box = [frag_df[frag_df["model"] == m]["n_components"] for m in models]
    bp = ax.boxplot(data_box, tick_labels=short_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], MODEL_PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("# connected components per tile")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mean component size (pixels)")
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [frag_df[frag_df["model"] == m]["mean_size"].mean() for m in models]
        bars = ax.bar(short_labels, vals, color=MODEL_PALETTE, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Mean bounding-box fill (compactness)")
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [frag_df[frag_df["model"] == m]["mean_bbox_fill"].mean() for m in models]
        bars = ax.bar(short_labels, vals, color=MODEL_PALETTE, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Interpretation")
    frag_summary = frag_df.groupby("model")["n_components"].mean()
    fill_summary = frag_df.groupby("model")["mean_bbox_fill"].mean()
    most_fragmented = frag_summary.idxmax()
    smoothest = fill_summary.idxmax()
    st.info(
        f"**Most fragmented predictions:** `{model_short(most_fragmented)}` "
        f"(avg {frag_summary[most_fragmented]:.1f} components/tile). "
        f"**Smoothest / most compact:** `{model_short(smoothest)}` "
        f"(avg bbox fill {fill_summary[smoothest]:.2f})."
    )

    st.subheader("Representative tiles: most-fragmented prediction per model")
    show_model = st.selectbox("Model", models, format_func=model_short, key="frag_gallery_model")
    top_frag_tiles = (frag_df[frag_df["model"] == show_model]
                       .sort_values("n_components", ascending=False).head(3))
    cols = st.columns(len(top_frag_tiles)) if len(top_frag_tiles) else []
    for col, (_, r) in zip(cols, top_frag_tiles.iterrows()):
        with col:
            after_rgb = load_rgb_frame(r["tile"], 3)
            t = load_tile(r["tile"])
            avg_pred = np.mean([t[f"pred__{show_model}"][i].astype(bool)
                                 for i in range(t[f"pred__{show_model}"].shape[0])], axis=0) >= 0.5
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(after_rgb); ax.imshow(mask_to_rgba(avg_pred, CLASS_COLORS[1]))
            ax.set_title(f"{r['tile']}\n{int(r['n_components'])} components", fontsize=9)
            ax.axis("off")
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Seasonality
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.header("Seasonality")
    st.caption(
        "Are errors and prediction shapes associated with the month of the 'after' observation? "
        "Reported as associations, not causal claims — month, location, cloudiness, and sample size "
        "are entangled in this test set. New analysis, not in the original write-up."
    )

    joined = metrics_df.join(tiles_df[["t4_month", "t4_year"]])
    joined_frag = frag_df.merge(tiles_df[["t4_month"]], left_on="tile", right_index=True)

    month_counts = joined.groupby("t4_month").size().reindex(range(1, 13), fill_value=0)
    st.caption("Tiles per month: " + " · ".join(
        f"{MONTH_NAMES[m]}: {int(month_counts[m])}" for m in range(1, 13) if month_counts[m] > 0
    ))

    st.subheader(f"Metrics by month of 'after' observation — {model_short(selected_model)}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ["dice", "precision", "recall"], ["Dice", "Precision", "Recall"]):
        monthly = joined.groupby("t4_month")[key].mean().reindex(range(1, 13))
        valid_months = [m for m in range(1, 13) if not np.isnan(monthly[m])]
        ax.plot(valid_months, [monthly[m] for m in valid_months], marker="o", color="#4C72B0")
        ax.set_xticks(valid_months); ax.set_xticklabels([MONTH_NAMES[m] for m in valid_months], rotation=45, ha="right")
        ax.set_ylim(0, 1); ax.set_title(title, fontweight="bold"); ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Predicted loss fraction and fragmentation by month (all models)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for m, color in zip(models, MODEL_PALETTE):
        mdf = compute_model_tile_metrics(m).join(tiles_df[["t4_month"]])
        monthly_loss = mdf.groupby("t4_month")["pred_loss_fraction"].mean().reindex(range(1, 13))
        valid_months = [mo for mo in range(1, 13) if not np.isnan(monthly_loss[mo])]
        axes[0].plot(valid_months, [monthly_loss[mo] for mo in valid_months],
                     marker="o", color=color, label=model_short(m))

        fdf = frag_df[frag_df["model"] == m].merge(tiles_df[["t4_month"]], left_on="tile", right_index=True)
        monthly_frag = fdf.groupby("t4_month")["n_components"].mean().reindex(range(1, 13))
        valid_months_f = [mo for mo in range(1, 13) if not np.isnan(monthly_frag[mo])]
        axes[1].plot(valid_months_f, [monthly_frag[mo] for mo in valid_months_f],
                     marker="o", color=color, label=model_short(m))

    axes[0].set_title("Predicted loss fraction by month", fontweight="bold")
    axes[1].set_title("Mean # components per tile by month", fontweight="bold")
    for ax in axes:
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES[1:], rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.legend(fontsize=7)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Easiest vs. hardest month")
    monthly_dice = joined.groupby("t4_month")["dice"].mean().dropna()
    if not monthly_dice.empty:
        best_month = monthly_dice.idxmax(); worst_month = monthly_dice.idxmin()
        st.info(
            f"For **{model_short(selected_model)}**, the easiest month is **{MONTH_NAMES[best_month]}** "
            f"(mean Dice {monthly_dice[best_month]:.3f}, n={int(month_counts[best_month])}) and the hardest "
            f"is **{MONTH_NAMES[worst_month]}** (mean Dice {monthly_dice[worst_month]:.3f}, "
            f"n={int(month_counts[worst_month])})."
        )

        example_tiles = {
            "Easiest": joined[joined["t4_month"] == best_month].sort_values("dice", ascending=False).head(2),
            "Hardest": joined[joined["t4_month"] == worst_month].sort_values("dice", ascending=True).head(2),
        }
        cols = st.columns(4)
        i = 0
        for label_txt, ex_df in example_tiles.items():
            for tile_id, r in ex_df.iterrows():
                with cols[i]:
                    after_rgb = load_rgb_frame(tile_id, 3)
                    fig, ax = plt.subplots(figsize=(3.5, 3.5))
                    ax.imshow(after_rgb)
                    ax.set_title(f"{label_txt}: {tile_id}\nDice={r['dice']:.2f}", fontsize=8)
                    ax.axis("off")
                    st.pyplot(fig); plt.close()
                i += 1


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Stratified & Ensemble Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab8:
    st.header("Stratified & Ensemble Analysis")
    st.caption(
        "Additional cuts from the project write-up: tile-level loss-density stratification, an "
        "all-model alert-confidence comparison, and a post-hoc probability-ensemble check."
    )

    s_tab1, s_tab2, s_tab3 = st.tabs(
        ["Tile Loss-Density", "Confidence (all models)", "Ensemble Analysis"]
    )

    with s_tab1:
        st.caption("Tiles binned by their overall GT forest-loss pixel fraction. Dice is micro-averaged "
                   "(pooled TP/FP/FN) within each bin, then averaged across seeds.")
        with st.spinner("Computing loss-density stratified Dice… (cached after first run)"):
            density_df = compute_density_stratified_dice()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(DENSITY_LABELS))
        width = 0.8 / len(models)
        for j, (m, color) in enumerate(zip(models, MODEL_PALETTE)):
            mdf = density_df[density_df["model"] == m].set_index("bin").reindex(DENSITY_LABELS)
            ax.bar(x + j * width, mdf["mean_dice"], width, yerr=mdf["std_dice"],
                   label=model_short(m), color=color, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + width * (len(models) - 1) / 2); ax.set_xticklabels(DENSITY_LABELS)
        ax.set_xlabel("Tile loss density"); ax.set_ylabel("Micro Dice (%)")
        ax.set_title("Tile Loss Density vs. Micro Dice", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        counts_row = density_df.drop_duplicates("bin").set_index("bin")["n_tiles"].reindex(DENSITY_LABELS)
        st.caption("Tiles per bin: " + " · ".join(f"{b}: n={int(counts_row[b])}" for b in DENSITY_LABELS))

    with s_tab2:
        st.caption("Same idea as the Confusion Matrix tab's confidence breakdown, but comparing all 6 "
                   "models on one chart instead of one model at a time.")
        with st.spinner("Computing confidence-stratified Dice across all models… (cached after first run)"):
            conf_all_df = compute_confidence_stratified_dice_all_models()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ALL_CONF_LEVELS))
        width = 0.8 / len(models)
        gt_shares = conf_all_df.drop_duplicates("conf").set_index("conf")["gt_share"].reindex(ALL_CONF_LEVELS)
        for j, (m, color) in enumerate(zip(models, MODEL_PALETTE)):
            mdf = conf_all_df[conf_all_df["model"] == m].set_index("conf").reindex(ALL_CONF_LEVELS)
            ax.bar(x + j * width, mdf["mean_dice"], width, yerr=mdf["std_dice"],
                   label=model_short(m), color=color, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"{CONF_DISPLAY[c]}\n{gt_shares[c]:.2%}" for c in ALL_CONF_LEVELS])
        ax.set_xlabel("Alert confidence  (share of GT loss pixels)"); ax.set_ylabel("Micro Dice (%)")
        ax.set_title("Alert Confidence vs. Micro Dice — all models", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with s_tab3:
        st.caption(
            "Post-hoc ensembling: average the seed-averaged forest-loss probability of two models, then "
            "threshold at 0.5 — compared against each model alone and, as a reference, the trained Prithvi "
            "LiteTA Fusion model (which learns its fusion end-to-end rather than combining after the fact)."
        )
        ec1, ec2 = st.columns(2)
        with ec1:
            default_a = models.index("prithvi_temporal_300m_concat") if "prithvi_temporal_300m_concat" in models else 0
            ens_a = st.selectbox("Model A", models, index=default_a, format_func=model_short, key="ens_a")
        with ec2:
            other_models = [m for m in models if m != ens_a]
            ens_b = st.selectbox("Model B", other_models, format_func=model_short, key="ens_b")

        with st.spinner("Computing ensemble Dice… (cached after first run)"):
            ens_result = compute_ensemble_dice(ens_a, ens_b)

        fusion_key = "prithvi_temporal_liteta_fusion_300m_concat"
        bars = {
            model_short(ens_a): (ens_result["a_dice"], ens_result["a_std"]),
            model_short(ens_b): (ens_result["b_dice"], ens_result["b_std"]),
            f"{model_short(ens_a)} + {model_short(ens_b)} Ensemble":
                (ens_result["ensemble_dice"], ens_result["ensemble_std"]),
        }
        if fusion_key in models and fusion_key not in (ens_a, ens_b):
            fusion_micro = compute_model_micro_metrics(fusion_key)
            bars["Prithvi LiteTA Fusion (trained)"] = (
                float(fusion_micro["dice"].mean()) * 100,
                float(fusion_micro["dice"].std(ddof=0)) * 100,
            )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = list(bars.keys())
        vals = [v[0] for v in bars.values()]
        errs = [v[1] for v in bars.values()]
        bar_colors = MODEL_PALETTE[:len(names)]
        b = ax.barh(names, vals, xerr=errs, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(b, vals):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9)
        ax.set_xlabel("Micro Dice (%)"); ax.set_title("Ensemble vs. individual models vs. trained fusion", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Multi-Model Qualitative Gallery
# ══════════════════════════════════════════════════════════════════════════════

with tab9:
    st.header("Multi-Model Qualitative Gallery")
    st.caption(
        "Side-by-side comparison of the four temporal inputs, ground truth, and several models' "
        "predictions on the same tile — mirrors the project write-up's qualitative comparison grid."
    )

    gal_models = st.multiselect("Models to display", models, default=models,
                                 format_func=model_short, key="gallery_models")
    n_gallery = st.slider("# tiles to show", 1, 10, 5, key="gallery_n")
    gallery_mode = st.radio("Tile selection", ["Highest model disagreement", "Manual"], horizontal=True)

    if gallery_mode == "Manual":
        gallery_tiles = st.multiselect("Pick tiles", tile_names,
                                        default=tile_names[:min(3, len(tile_names))], key="gallery_manual_tiles")
    elif gal_models:
        model_dice_by_tile = {m: compute_model_tile_metrics(m)["dice"] for m in gal_models}
        spread_df = pd.DataFrame(model_dice_by_tile)
        spread_df["spread"] = spread_df.max(axis=1) - spread_df.min(axis=1)
        gallery_tiles = list(spread_df.sort_values("spread", ascending=False).head(n_gallery).index)
    else:
        gallery_tiles = []

    if not gal_models:
        st.warning("Select at least one model.")
    elif not gallery_tiles:
        st.warning("No tiles to display.")
    else:
        for tname in gallery_tiles:
            t = load_tile(tname)
            gt_t = t["gt"].astype(bool)
            frames_t = [load_rgb_frame(tname, i) for i in range(4)]

            n_panels = 4 + 1 + len(gal_models)
            fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.9))
            for ax, img, lbl in zip(axes[:4], frames_t, PHASE_LABELS):
                ax.imshow(img); ax.set_title(lbl, fontsize=8); ax.axis("off")

            axes[4].imshow(frames_t[-1]); axes[4].imshow(mask_to_rgba(gt_t, CLASS_COLORS[1]))
            axes[4].set_title("Ground Truth", fontsize=8, fontweight="bold"); axes[4].axis("off")

            for ax, m in zip(axes[5:], gal_models):
                preds_m = t[f"pred__{m}"]
                avg_pred = np.mean([preds_m[i].astype(bool) for i in range(preds_m.shape[0])], axis=0) >= 0.5
                ax.imshow(frames_t[-1]); ax.imshow(mask_to_rgba(avg_pred, CLASS_COLORS[1]))
                dice_m = binary_metrics(avg_pred, gt_t)["dice"]
                ax.set_title(f"{model_short(m)}\nDice={dice_m:.2f}", fontsize=8)
                ax.axis("off")

            fig.suptitle(tname, fontsize=9, y=1.02)
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — Report Highlights (static reference material)
# ══════════════════════════════════════════════════════════════════════════════

with tab10:
    st.header("Report Highlights")
    st.caption(
        "These come from the project write-up's training logs and offline benchmarks, not from the "
        "588 held-out test tiles in `data/tiles` — so unlike the rest of this app, they can't be "
        "recomputed live and are shown here as reference images/tables."
    )

    REF_DIR = os.path.join(DATA_DIR, "reference")

    st.subheader("Prithvi LiteTA Fusion architecture")
    st.image(
        os.path.join(REF_DIR, "architecture_prithvi_liteta.jpg"),
        caption="A pretrained Prithvi temporal path augmented with a lightweight temporal-attention "
                "U-Net branch for local detail, combined via a gated residual fusion.",
    )

    st.divider()
    st.subheader("Inference time (best batch size per model)")
    inference_df = pd.DataFrame([
        {"Model": "Lite Temporal U-Net", "Batch": 32, "Latency (ms/batch)": 20.96,
         "Images/sec": 1526.57, "Peak GPU (MB)": 7550.5, "Params": 620_603},
        {"Model": "U-Net", "Batch": 32, "Latency (ms/batch)": 11.94,
         "Images/sec": 2679.77, "Peak GPU (MB)": 3968.2, "Params": 31_049_794},
        {"Model": "3D U-Net", "Batch": 32, "Latency (ms/batch)": 45.72,
         "Images/sec": 699.85, "Peak GPU (MB)": 15026.9, "Params": 5_785_383},
        {"Model": "U-TAE", "Batch": 32, "Latency (ms/batch)": 81.61,
         "Images/sec": 392.12, "Peak GPU (MB)": 8353.7, "Params": 1_079_718},
        {"Model": "Prithvi", "Batch": 32, "Latency (ms/batch)": 519.98,
         "Images/sec": 61.54, "Peak GPU (MB)": 17900.7, "Params": 317_414_126},
        {"Model": "Prithvi + Lite Temporal U-Net Fusion", "Batch": 32, "Latency (ms/batch)": 549.98,
         "Images/sec": 58.18, "Peak GPU (MB)": 20254.7, "Params": 317_936_168},
    ])
    st.dataframe(inference_df, width='stretch', hide_index=True)

    st.divider()
    st.subheader("Training curves — Prithvi LiteTA Fusion")
    st.image(
        os.path.join(REF_DIR, "training_curves_liteta.jpg"),
        caption="Train loss and validation Dice over training steps.",
    )

    st.divider()
    st.subheader("Dice-loss coefficient ablation")
    st.image(
        os.path.join(REF_DIR, "dice_loss_ablation.jpg"),
        caption="Prithvi + Branch sensitivity to the Dice-loss weight in the combined "
                "cross-entropy + Dice training loss.",
    )

    st.divider()
    st.subheader("Limited-data setting (20% vs. 100% training data)")
    st.image(
        os.path.join(REF_DIR, "limited_data_prithvi.jpg"),
        caption="Prithvi w/ U-Net maintains strong performance even as training data is reduced to 20%.",
    )
    st.info(
        "No 20%-training-data model runs are included in this test set's export "
        "(`export_metadata.json` reports `data_percentage: 1.0`), so this comparison can't be "
        "reproduced from `data/tiles`."
    )
