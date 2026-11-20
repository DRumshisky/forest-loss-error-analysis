# Forest-Loss Detection — Error Analysis

An interactive [Streamlit](https://streamlit.io) app for analyzing near-real-time
**forest-loss detection** models. Each model takes four Harmonized Landsat/Sentinel-2 (HLS)
observations of a location (before, during, during, after) and predicts which pixels
experienced forest loss. Ground truth comes from GLAD-S2 alerts.

The app compares 6 models (3D U-Net, U-Net, local U-Net, U-TAE, and two Prithvi variants)
on 588 held-out 224×224 test tiles. Use it to see where models make mistakes and how they
differ.

**Features:** confusion matrices (overall and by cloud / alert-confidence level), a tile
viewer with TP/FP/FN error maps, per-tile summary stats, model-vs-model comparison,
component-size and prediction-shape analysis, seasonality, ensembles, a multi-model
gallery, and report highlights.

## How to open it

**Requirements:** Python 3.9+ and about 3 GB of free disk space.

```bash
# 1. Clone and enter the repo
git clone https://github.com/DRumshis/forest-loss-error-analysis.git
cd forest-loss-error-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app2.py
```

Streamlit prints a local URL (usually http://localhost:8501). Open it in your browser.

**First launch:** if `data/` isn't present, the app downloads the ~2.8 GB test set from
[Hugging Face](https://huggingface.co/datasets/mqraitem/Deforest-Analysis) automatically.
This takes a few minutes and only happens once.

## Project layout

```text
app2.py            the Streamlit app (downloads data on first run)
app.py             earlier version that expects data/ to already exist
reference/         static figures shown in the Report Highlights tab
data/              test set (tiles, RGB renders, metadata); downloaded on demand
deploy_space.sh    deploys the app to a Hugging Face Space
```

## Data

Source: [mqraitem/Deforest-Analysis](https://huggingface.co/datasets/mqraitem/Deforest-Analysis).
Each `data/tiles/<tile_id>.npz` holds the spectral bands, cloud mask, ground truth, GLAD
alert confidence, and each model's logits, probabilities, and predictions per seed.

```python
import numpy as np

tile = np.load("data/tiles/<tile_id>.npz")
gt = tile["gt"]               # (224, 224) binary forest-loss target
pred = tile["pred__utae"][0]  # first seed's prediction mask
```

## Notes

- The test set is a fixed held-out split. Don't use it to select a model or tune thresholds.
- Seasonal findings are associations only. Month, location, and cloudiness are entangled.
