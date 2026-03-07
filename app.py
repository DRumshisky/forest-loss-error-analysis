import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Land Use — Error Analysis",
    page_icon="🛰️",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CLASS_NAMES = ["Everything Else", "Oil Palm", "Rubber", "Mining"]
# RGB tuples (0-255) matching README colour suggestions
CLASS_COLORS = [
    (128, 128, 128),   # gray
    (0,   180,   0),   # green
    (220, 200,   0),   # yellow
    (200,   0,   0),   # red
]
MODEL_PALETTE = ["#4C72B0", "#DD8452", "#55A868"]  # one colour per model


# ── Data helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def get_models():
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )


@st.cache_data
def get_tiles(model: str):
    model_dir = os.path.join(DATA_DIR, model)
    return sorted(f for f in os.listdir(model_dir) if f.endswith(".npz"))


@st.cache_data
def load_tile(model: str, tile_name: str):
    path = os.path.join(DATA_DIR, model, tile_name)
    tile = np.load(path)
    return tile["pred"], tile["gt"], tile["rgb"]


def dice_per_class(pred, gt):
    """Return list of 4 Dice scores (skipping gt == -1 pixels)."""
    mask = gt >= 0
    scores = []
    for c in range(4):
        tp = int(np.sum((pred == c) & (gt == c) & mask))
        fp = int(np.sum((pred == c) & (gt != c) & mask))
        fn = int(np.sum((pred != c) & (gt == c) & mask))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return scores


@st.cache_data
def compute_all_metrics(model: str):
    """Per-tile accuracy, error rate and Dice scores for every tile."""
    results = []
    for tile_name in get_tiles(model):
        pred, gt, _ = load_tile(model, tile_name)
        mask = gt >= 0
        if mask.sum() == 0:
            continue
        acc = float(np.mean(pred[mask] == gt[mask]))
        dice = dice_per_class(pred, gt)
        results.append({
            "tile":       tile_name,
            "accuracy":   acc,
            "error_rate": 1.0 - acc,
            "dice":       dice,
            "mean_dice":  float(np.mean(dice)),
        })
    return results


@st.cache_data
def compute_confusion_matrix(model: str):
    """Aggregate 4×4 confusion matrix (rows = true, cols = pred)."""
    cm = np.zeros((4, 4), dtype=np.int64)
    for tile_name in get_tiles(model):
        pred, gt, _ = load_tile(model, tile_name)
        mask = gt >= 0
        for true_c in range(4):
            for pred_c in range(4):
                cm[true_c, pred_c] += int(
                    np.sum((gt == true_c) & (pred == pred_c) & mask)
                )
    return cm


# ── Image-building helpers ────────────────────────────────────────────────────

def class_to_rgba(arr, alpha=210):
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    for c, color in enumerate(CLASS_COLORS):
        m = arr == c
        rgba[m, :3] = color
        rgba[m, 3] = alpha
    rgba[arr < 0] = 0
    return rgba


def error_map_rgba(pred, gt):
    """Gray where correct, red where wrong, transparent where ignored."""
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    valid   = gt >= 0
    correct = valid & (pred == gt)
    wrong   = valid & (pred != gt)
    rgba[correct] = (128, 128, 128, 150)
    rgba[wrong]   = (220,   0,   0, 230)
    return rgba


def class_error_map_rgba(pred, gt, class_id):
    """
    For a specific class:
      green  = true positive (correctly found)
      red    = false negative (missed)
      orange = false positive (wrong alarm)
      dark   = true negative (correctly not labelled)
    """
    rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    valid = gt >= 0
    tp = valid & (gt == class_id) & (pred == class_id)
    fn = valid & (gt == class_id) & (pred != class_id)
    fp = valid & (gt != class_id) & (pred == class_id)
    tn = valid & (gt != class_id) & (pred != class_id)
    rgba[tp] = (0,   200,   0, 210)
    rgba[fn] = (220,   0,   0, 210)
    rgba[fp] = (255, 165,   0, 210)
    rgba[tn] = ( 50,  50,  50,  60)
    return rgba


def legend_patches():
    return [
        mpatches.Patch(
            facecolor=[c / 255 for c in CLASS_COLORS[i]],
            edgecolor="black",
            linewidth=0.4,
            label=CLASS_NAMES[i],
        )
        for i in range(4)
    ]


def model_short(name: str) -> str:
    return name.replace("_datapercentage-1.0", "")


# ── Sidebar ───────────────────────────────────────────────────────────────────

models = get_models()

with st.sidebar:
    st.title("🛰️ Settings")
    selected_model = st.selectbox(
        "Active model",
        models,
        format_func=model_short,
    )
    st.divider()

    st.subheader("Tile filters")
    sort_by = st.selectbox(
        "Sort tiles by",
        ["Name (A→Z)", "Error Rate (high→low)", "Mean Dice (low→high)"],
    )
    min_error = st.slider(
        "Min error rate",
        0.0, 1.0, 0.0, 0.01,
        help="Only show tiles whose error rate ≥ this value",
    )
    st.divider()

    st.subheader("Colour legend")
    for i, (name, (r, g, b)) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        txt_color = "black" if i in (0, 1) else "white"
        st.markdown(
            f'<span style="background:rgb({r},{g},{b});padding:2px 10px;'
            f'border-radius:4px;color:{txt_color};font-weight:600">'
            f"&nbsp;{name}&nbsp;</span>",
            unsafe_allow_html=True,
        )

# ── Load & filter tiles ───────────────────────────────────────────────────────

all_metrics = compute_all_metrics(selected_model)
filtered = [m for m in all_metrics if m["error_rate"] >= min_error]

if sort_by == "Error Rate (high→low)":
    filtered.sort(key=lambda x: -x["error_rate"])
elif sort_by == "Mean Dice (low→high)":
    filtered.sort(key=lambda x: x["mean_dice"])

tile_names       = [m["tile"] for m in filtered]
tile_metrics_map = {m["tile"]: m for m in filtered}

# ── Page title ────────────────────────────────────────────────────────────────

st.title("Satellite Land Use — Error Analysis Tool")
st.caption(
    "Exploring model predictions for mining, oil palm, and rubber plantations "
    "in West Africa. Data: 224×224 pixel satellite tiles."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Confusion Matrix", "🗺️ Tile Viewer", "📈 Summary Stats", "⚖️ Model Comparison"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header(f"Confusion Matrix — {model_short(selected_model)}")
    cm = compute_confusion_matrix(selected_model)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw pixel counts")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted class", fontsize=10)
        ax.set_ylabel("True class", fontsize=10)
        ax.set_title("Rows = true class, Columns = predicted class")
        threshold = cm.max() * 0.55
        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black",
                )
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Row-normalised (recall per class)")
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted class", fontsize=10)
        ax.set_ylabel("True class", fontsize=10)
        ax.set_title("Each row sums to 1 (what % got predicted as what)")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, f"{cm_norm[i, j]:.1%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Insights ──────────────────────────────────────────────────────────────
    st.subheader("Key insights")

    per_class_recall = [
        cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        for i in range(4)
    ]
    best_idx  = int(np.argmax(per_class_recall))
    worst_idx = int(np.argmin(per_class_recall))

    cm_off_diag = cm.astype(float).copy()
    np.fill_diagonal(cm_off_diag, 0)
    ci, cj = np.unravel_index(np.argmax(cm_off_diag), cm_off_diag.shape)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Best classified class",
        CLASS_NAMES[best_idx],
        f"{per_class_recall[best_idx]:.1%} recall",
    )
    c2.metric(
        "Hardest class",
        CLASS_NAMES[worst_idx],
        f"{per_class_recall[worst_idx]:.1%} recall",
    )
    c3.metric(
        "Most confused pair",
        f"{CLASS_NAMES[ci]} → {CLASS_NAMES[cj]}",
        f"{int(cm_off_diag[ci, cj]):,} pixels mis-labelled",
    )

    pred_class0_pct = cm[:, 0].sum() / cm.sum()
    biased = pred_class0_pct > 0.70
    st.info(
        f"**'Everything Else' prediction share:** {pred_class0_pct:.1%} of all "
        "predicted pixels are labelled as class 0. "
        + ("⚠️ The model appears biased toward this majority class."
           if biased else "✅ No strong majority-class bias detected.")
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tile Viewer
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Interactive Tile Viewer")

    if not tile_names:
        st.warning(
            "No tiles match the current filters. "
            "Lower the min error rate threshold in the sidebar."
        )
    else:
        # ── Navigation ────────────────────────────────────────────────────────
        if "tile_idx" not in st.session_state:
            st.session_state.tile_idx = 0
        st.session_state.tile_idx = min(
            st.session_state.tile_idx, len(tile_names) - 1
        )

        nav1, nav2, nav3 = st.columns([1, 1, 4])
        with nav1:
            if st.button("⬅ Previous", use_container_width=True):
                if st.session_state.tile_idx > 0:
                    st.session_state.tile_idx -= 1
        with nav2:
            if st.button("Next ➡", use_container_width=True):
                if st.session_state.tile_idx < len(tile_names) - 1:
                    st.session_state.tile_idx += 1
        with nav3:
            chosen = st.selectbox(
                "Jump to tile",
                tile_names,
                index=st.session_state.tile_idx,
                label_visibility="collapsed",
            )
            st.session_state.tile_idx = tile_names.index(chosen)

        current_tile = tile_names[st.session_state.tile_idx]
        st.markdown(
            f"**Tile {st.session_state.tile_idx + 1} / {len(tile_names)}** "
            f"— `{current_tile}`"
        )

        pred, gt, rgb = load_tile(selected_model, current_tile)
        tm = tile_metrics_map[current_tile]

        # ── Per-tile metric strip ─────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",          f"{tm['accuracy']:.1%}")
        m2.metric("Mean Dice",         f"{tm['mean_dice']:.3f}")
        m3.metric("Dice – Oil Palm",   f"{tm['dice'][1]:.3f}")
        m4.metric("Dice – Rubber",     f"{tm['dice'][2]:.3f}")
        m5.metric("Dice – Mining",     f"{tm['dice'][3]:.3f}")
        m6.metric("Error Rate",        f"{tm['error_rate']:.1%}")

        # ── 4-panel image strip ───────────────────────────────────────────────
        gt_rgba   = class_to_rgba(gt)
        pred_rgba = class_to_rgba(pred)
        err_rgba  = error_map_rgba(pred, gt)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        panel_data = [
            (rgb,       "Satellite Image"),
            (gt_rgba,   "Ground Truth"),
            (pred_rgba, "Model Prediction"),
            (err_rgba,  "Error Map  (red = wrong)"),
        ]
        for ax, (img, title) in zip(axes, panel_data):
            ax.imshow(img)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.axis("off")

        fig.legend(
            handles=legend_patches(),
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.04),
            fontsize=9,
        )
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Per-class error maps (stretch goal) ───────────────────────────────
        with st.expander("🔍 Per-class error maps (stretch goal)"):
            st.caption(
                "**Green** = correctly detected · "
                "**Red** = missed (false negative) · "
                "**Orange** = false alarm (false positive)"
            )
            fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4.2))
            for ax, class_id, name in zip(axes2, range(4), CLASS_NAMES):
                ax.imshow(rgb)
                ax.imshow(class_error_map_rgba(pred, gt, class_id))
                ax.set_title(f"Class: {name}", fontweight="bold", fontsize=10)
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig2); plt.close()

        # ── Boundary analysis (stretch goal) ─────────────────────────────────
        with st.expander("📐 Boundary analysis (stretch goal)"):
            st.caption(
                "Are errors concentrated at class boundaries or in the "
                "interior of objects? "
                "Boundary pixels are those adjacent to a pixel of a different class."
            )
            from scipy.ndimage import binary_dilation  # type: ignore

            valid_mask = gt >= 0
            is_wrong   = valid_mask & (pred != gt)

            # Build a boundary mask: pixel is a boundary if any neighbour has a
            # different gt label (ignoring ignored pixels)
            boundary = np.zeros(gt.shape, dtype=bool)
            for c in range(4):
                class_mask = (gt == c) & valid_mask
                dilated    = binary_dilation(class_mask, iterations=1)
                # Boundary = pixels in the dilation that are NOT in the class
                boundary |= (dilated & valid_mask & ~class_mask)

            boundary_errors = int(np.sum(is_wrong & boundary))
            interior_errors = int(np.sum(is_wrong & ~boundary))
            total_errors    = boundary_errors + interior_errors
            if total_errors > 0:
                b_pct = boundary_errors / total_errors
                i_pct = interior_errors / total_errors
                c1, c2 = st.columns(2)
                c1.metric(
                    "Boundary errors",
                    f"{boundary_errors:,}",
                    f"{b_pct:.1%} of all wrong pixels",
                )
                c2.metric(
                    "Interior errors",
                    f"{interior_errors:,}",
                    f"{i_pct:.1%} of all wrong pixels",
                )
                if b_pct > 0.6:
                    st.info("Most errors occur at **class boundaries** — typical for segmentation models.")
                else:
                    st.info("Errors are spread across **interiors** as well as boundaries.")

                # Visualise
                overlay = np.zeros((*gt.shape, 4), dtype=np.uint8)
                overlay[is_wrong & boundary]  = (255, 165,   0, 230)  # orange = boundary error
                overlay[is_wrong & ~boundary] = (220,   0,   0, 200)  # red    = interior error
                overlay[boundary & ~is_wrong] = (100, 100, 255,  80)  # blue   = correct boundary
                fig3, (ax_rgb, ax_ov) = plt.subplots(1, 2, figsize=(10, 4))
                ax_rgb.imshow(rgb); ax_rgb.set_title("Satellite"); ax_rgb.axis("off")
                ax_ov.imshow(rgb);  ax_ov.imshow(overlay)
                ax_ov.set_title("Boundary errors (orange) vs interior errors (red)")
                ax_ov.axis("off")
                plt.tight_layout()
                st.pyplot(fig3); plt.close()
            else:
                st.success("No prediction errors on this tile!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Summary Statistics — {model_short(selected_model)}")

    per_class_dice_lists = [
        [m["dice"][c] for m in all_metrics] for c in range(4)
    ]
    mean_dice_per_class = [float(np.mean(d)) for d in per_class_dice_lists]
    error_rates         = [m["error_rate"] for m in all_metrics]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-class mean Dice score")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_f = [[c / 255 for c in col] for col in CLASS_COLORS]
        bars = ax.bar(CLASS_NAMES, mean_dice_per_class, color=colors_f,
                      edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Dice Score")
        ax.set_title("Averaged across all tiles")
        for bar, val in zip(bars, mean_dice_per_class):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Error rate distribution across tiles")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(error_rates, bins=25, color="#4C72B0", edgecolor="black",
                linewidth=0.4, alpha=0.85)
        mean_err = float(np.mean(error_rates))
        ax.axvline(mean_err, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_err:.1%}")
        ax.set_xlabel("Error Rate")
        ax.set_ylabel("Number of tiles")
        ax.set_title("Distribution of per-tile error rates")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Per-class accuracy bar chart ──────────────────────────────────────────
    st.subheader("Per-class accuracy (confusion-matrix diagonal)")
    cm_s = compute_confusion_matrix(selected_model)
    per_class_acc = [
        cm_s[i, i] / cm_s[i].sum() if cm_s[i].sum() > 0 else 0.0
        for i in range(4)
    ]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors_f,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall (true-positive rate)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    for bar, val in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}", ha="center", va="bottom", fontsize=9,
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Worst-tile table ──────────────────────────────────────────────────────
    st.subheader("10 worst-performing tiles")
    worst = sorted(all_metrics, key=lambda x: -x["error_rate"])[:10]
    df = pd.DataFrame([
        {
            "Tile":             m["tile"],
            "Error Rate":       f"{m['error_rate']:.1%}",
            "Accuracy":         f"{m['accuracy']:.1%}",
            "Mean Dice":        f"{m['mean_dice']:.3f}",
            "Dice – Oil Palm":  f"{m['dice'][1]:.3f}",
            "Dice – Rubber":    f"{m['dice'][2]:.3f}",
            "Dice – Mining":    f"{m['dice'][3]:.3f}",
        }
        for m in worst
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Model Comparison")
    st.caption("Comparing all three models across the same 202 tiles.")

    # Pre-compute stats for every model
    model_stats = {}
    for model in models:
        metrics_m = compute_all_metrics(model)
        per_class_d = [
            float(np.mean([m["dice"][c] for m in metrics_m]))
            for c in range(4)
        ]
        model_stats[model] = {
            "per_class_dice": per_class_d,
            "overall_acc":    float(np.mean([m["accuracy"]   for m in metrics_m])),
            "overall_dice":   float(np.mean([m["mean_dice"]  for m in metrics_m])),
        }

    short_labels = [model_short(m) for m in models]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall accuracy")
        fig, ax = plt.subplots(figsize=(6, 4))
        accs = [model_stats[m]["overall_acc"] for m in models]
        bars = ax.bar(short_labels, accs, color=MODEL_PALETTE, edgecolor="black",
                      linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean pixel accuracy")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Overall mean Dice")
        fig, ax = plt.subplots(figsize=(6, 4))
        dices = [model_stats[m]["overall_dice"] for m in models]
        bars = ax.bar(short_labels, dices, color=MODEL_PALETTE, edgecolor="black",
                      linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Dice score")
        for bar, val in zip(bars, dices):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Per-class Dice by model ───────────────────────────────────────────────
    st.subheader("Per-class Dice score by model")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ci, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
        vals = [model_stats[m]["per_class_dice"][ci] for m in models]
        bars = ax.bar(short_labels, vals, color=MODEL_PALETTE, edgecolor="black",
                      linewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_title(class_name, fontweight="bold")
        ax.set_ylabel("Dice Score" if ci == 0 else "")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    plt.suptitle("Per-class Dice — which model is best per class?",
                 fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Best model per class ──────────────────────────────────────────────────
    st.subheader("Best model per class")
    for ci, class_name in enumerate(CLASS_NAMES):
        scored = [
            (model_stats[m]["per_class_dice"][ci], model_short(m))
            for m in models
        ]
        best_dice, best_name = max(scored)
        st.markdown(f"- **{class_name}**: `{best_name}` — Dice {best_dice:.3f}")

    # ── Head-to-head tile-level scatter ──────────────────────────────────────
    st.subheader("Head-to-head: mean Dice per tile (pick two models)")
    mc1, mc2 = st.columns(2)
    with mc1:
        model_a = st.selectbox("Model A", models, index=0, format_func=model_short,
                               key="cmp_a")
    with mc2:
        model_b = st.selectbox("Model B", models, index=1, format_func=model_short,
                               key="cmp_b")

    if model_a != model_b:
        tiles_a  = {m["tile"]: m["mean_dice"] for m in compute_all_metrics(model_a)}
        tiles_b  = {m["tile"]: m["mean_dice"] for m in compute_all_metrics(model_b)}
        common   = sorted(set(tiles_a) & set(tiles_b))
        da = [tiles_a[t] for t in common]
        db = [tiles_b[t] for t in common]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(da, db, alpha=0.4, s=20, color="#4C72B0")
        lim = (0, 1)
        ax.plot(lim, lim, "k--", linewidth=1, label="Equal performance")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(f"Mean Dice — {model_short(model_a)}")
        ax.set_ylabel(f"Mean Dice — {model_short(model_b)}")
        ax.set_title("Each dot = one tile\nAbove diagonal → model B better")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        a_wins = sum(1 for a, b in zip(da, db) if a > b)
        b_wins = sum(1 for a, b in zip(da, db) if b > a)
        ties   = len(common) - a_wins - b_wins
        st.caption(
            f"{model_short(model_a)} wins on **{a_wins}** tiles · "
            f"{model_short(model_b)} wins on **{b_wins}** tiles · "
            f"Ties: **{ties}**"
        )
    else:
        st.info("Select two different models to compare.")
