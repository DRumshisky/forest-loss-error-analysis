# Forest-Loss Detection — Error Analysis Project

## Background

This project analyzes near-real-time **forest-loss detection** models. Each model looks at
four Harmonized Landsat/Sentinel-2 (HLS) satellite observations of a location — **before**,
**during**, **during**, and **after** — and predicts which pixels experienced forest loss in
that window. Ground truth comes from GLAD-S2 forest-loss alerts.

Your job is to **analyze how well the models are doing** and **explore where and why they
disagree or make mistakes**. The web app (`app2.py`) is an interactive Streamlit tool built
for exactly that.

## The Data

Source: [huggingface.co/datasets/mqraitem/Deforest-Analysis](https://huggingface.co/datasets/mqraitem/Deforest-Analysis)
— 588 held-out 224×224 test tiles, 6 candidate models.

```text
data/
  tiles/<tile_id>.npz       one self-contained numerical record per test tile
  rgb/<tile_id>_*.png       four display-only RGB renders per test tile
  test_tiles.csv            dates, phases, location, and tile-level metadata
  models.json               model names, display names, seed IDs, and source runs
  export_metadata.json      fixed split and prediction conventions
```

Each `data/tiles/<tile_id>.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `spectral` | (6, 4, 224, 224) int16 | Raw HLS reflectance. Bands: blue, green, red, NIR, SWIR1, SWIR2. Dates: before, during, during, after. Nodata = -32768. |
| `clarity` | (4, 224, 224) uint8 | `1` = cloud/shadow/snow/cirrus/missing data per HLS Fmask, per observation date. |
| `gt` | (224, 224) uint8 | Forest-loss ground truth (binary). Components smaller than 10px removed. |
| `alert_conf` | (224, 224) uint8 | GLAD-S2 alert confidence (values 0, 2, 3, 4). |
| `logits__<model>` | (n_seeds, 2, 224, 224) float16 | Raw two-class logits (index 0 = no loss, 1 = loss). |
| `prob_change__<model>` | (n_seeds, 224, 224) float16 | Softmax probability of forest loss. |
| `pred__<model>` | (n_seeds, 224, 224) uint8 | Binary prediction mask (argmax over logits). |
| `seeds__<model>` | (n_seeds,) int8 | Seed IDs aligned with the arrays above. |

Models with 3 evaluated seeds: `3dunet`, `original_unet`, `utae`, `prithvi_temporal_liteta_fusion_300m_concat`.
Models with 1 seed: `local_unet`, `prithvi_temporal_300m_concat`. See `data/models.json` for display names.

`data/test_tiles.csv` has one row per tile: `tile_id`, `latitude`/`longitude`, `loss_pixels`,
`loss_fraction`, and per-observation `t{1..4}_timestamp/phase/year/month/day/doy/mgrs`.

### Loading the data

```python
import numpy as np

tile = np.load("data/tiles/<tile_id>.npz")
gt = tile["gt"]                                   # (224, 224) binary forest-loss target
seeds = tile["seeds__utae"]                       # e.g. [0, 1, 2]
pred = tile["pred__utae"][0]                      # first seed's prediction mask
```

## The App (`app2.py`)

```bash
streamlit run app2.py
```

Tabs:

1. **Confusion Matrix** — pixel-pooled binary confusion matrix (seed-averaged), plus
   breakdowns by cloud fraction and by GLAD alert-confidence level.
2. **Tile Viewer** — browse tiles; view the four temporal RGB inputs, ground truth,
   a chosen model/seed's prediction, and the TP/FP/FN error map, plus cloud and
   alert-confidence overlays and a boundary-vs-interior error breakdown.
3. **Summary Stats** — Dice/error-rate distributions across tiles, GT loss fraction vs.
   Dice, and the 10 worst-performing tiles for the active model.
4. **Model Comparison** — accuracy/Dice/precision/recall across all 6 models, and a
   per-tile head-to-head scatter for any two models.
5. **Component-Size Analysis** — 8-connected GT components matched to the
   best-overlapping predicted component, binned by size (10–24, 25–99, 100–499, 500+ px)
   and by GLAD alert-confidence level, with a visual component viewer.
6. **Prediction Shape** — per-model "personality": component count, size, and
   bounding-box fill (fragmented vs. smooth/compact predictions), with a gallery of
   each model's most-fragmented tiles.
7. **Seasonality** — Dice/precision/recall, predicted loss fraction, and fragmentation
   aggregated by the month of the "after" observation, with easiest/hardest month
   examples.

## Metrics

For a binary mask, pooling true positives (TP), false positives (FP), and false
negatives (FN) across pixels:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
Dice      = 2·TP / (2·TP + FP + FN)
IoU       = TP / (TP + FP + FN)
```

Models with multiple seeds have their per-tile metrics computed per seed and then
averaged, so an ensemble's score isn't dominated by one lucky/unlucky run.

## Notes and caveats

- Do not use this test set to select a model, tune a threshold, or claim a model was
  "selected from these analyses" — it's a fixed held-out split.
- Connected-component tables are *not* precomputed upstream; the app builds them at
  runtime with 8-pixel connectivity, matching the evaluation protocol.
- Seasonal associations are just that — associations. Month, location, cloudiness, and
  sample size are entangled in this test set, so treat seasonal findings as suggestive,
  not causal.
