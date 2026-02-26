# Satellite Land Use Classification — Error Analysis Project

## Background

Researchers use satellite images to automatically detect **land use changes** in West Africa — specifically, they want to find areas of **mining**, **oil palm plantations**, and **rubber plantations** from space.

A machine learning model looks at satellite images taken over several months and predicts what type of land each pixel belongs to. Your job is to **analyze how well the model is doing** and **build a tool to explore its mistakes**.

## The Data

Inside `data/<model_name>/`, there is one `.npz` file per satellite tile. Each file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `pred` | (224, 224) | Model's predicted class for each pixel |
| `gt` | (224, 224) | Ground truth (correct answer) for each pixel |
| `rgb` | (224, 224, 3) | RGB satellite image (uint8, 0–255) |

### Classes

| Value | Class |
|-------|-------|
| 0 | Everything else (forest, water, buildings, etc.) |
| 1 | Oil palm plantation |
| 2 | Rubber plantation |
| 3 | Mining |
| -1 | Ignore (no label available — skip these pixels) |

### Loading the data

```python
import numpy as np
import os

data_dir = "data/<model_name>"  # pick one of the model folders

# Load one tile
tile = np.load(os.path.join(data_dir, "<some_tile>.npz"))
pred = tile["pred"]   # (224, 224) — what the model predicted
gt = tile["gt"]       # (224, 224) — the correct answer
rgb = tile["rgb"]     # (224, 224, 3) — satellite photo

# Load all tiles
tiles = []
for f in sorted(os.listdir(data_dir)):
    if f.endswith(".npz"):
        tiles.append(np.load(os.path.join(data_dir, f)))
```

## Your Project

Build an **error analysis tool** — something that helps a researcher understand where and why the model makes mistakes. Here are the two main pieces:

### Part 1: Confusion Matrix

A confusion matrix shows how often the model confuses one class for another.

- For every pixel where `gt >= 0` (skip -1 pixels), compare `pred` vs `gt`.
- Build a 4×4 matrix: rows = true class, columns = predicted class.
- Each cell counts how many pixels had that (true, predicted) combination.
- Visualize it as a heatmap with labels.

Questions to answer:
- Which class does the model get right most often?
- Which classes get confused with each other?
- Is the model biased toward predicting "everything else" (class 0)?

### Part 2: Interactive Error Viewer

Build a visual tool (a webpage, a Streamlit app, or a Jupyter notebook) that lets you browse tiles and see:

1. **The satellite image** (rgb)
2. **Ground truth** — color-coded by class
3. **Model prediction** — color-coded by class
4. **Error map** — highlight pixels where pred ≠ gt (e.g., correct = transparent/gray, wrong = red)

Suggested colors:
- Class 0 (everything else): gray
- Class 1 (oil palm): green
- Class 2 (rubber): yellow
- Class 3 (mining): red

Nice-to-haves:
- Let the user click through tiles with Previous/Next buttons
- Show per-tile accuracy or Dice score
- Filter to only show tiles where the model does poorly
- Sort tiles by error rate

### Stretch Goals (if you finish early)

- **Per-class error maps**: separate error maps for each class (e.g., "where did the model miss mining?")
- **Boundary analysis**: are most errors at the edges of objects or in the middle?
- **Model comparison**: if `data/` has multiple model folders, compare them side by side. Which model is better at which class?
- **Summary statistics**: bar charts of per-class accuracy, per-class Dice score, etc.

## Dice Score (optional reading)

Dice score is the main metric used in this project. For a single class:

```
Dice = 2 * TP / (2 * TP + FP + FN)
```

Where:
- **TP** (true positive) = pixels correctly predicted as this class
- **FP** (false positive) = pixels wrongly predicted as this class
- **FN** (false negative) = pixels of this class that the model missed

Dice = 1.0 is perfect, Dice = 0.0 is completely wrong. Compute it per class, then average.

## Tips

- Use a coding assistant (like Claude) to help you write code — that's encouraged!
- Start small: load one tile, display it, then build up.
- `matplotlib` is great for quick plots. `Streamlit` or plain HTML/JS are great for interactive tools.
- Always filter out pixels where `gt == -1` before computing any metrics.
