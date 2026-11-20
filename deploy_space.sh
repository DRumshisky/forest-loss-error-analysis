#!/usr/bin/env bash
# Usage: ./deploy_space.sh <hf-username>/<space-name> [--private]
set -euo pipefail
SPACE="$1"; PRIV="${2:-}"
STAGE="$(mktemp -d)"
cp app2.py requirements.txt "$STAGE/"
cp -r reference "$STAGE/"
cat > "$STAGE/Dockerfile" <<'EOD'
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app2.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
EOD
cat > "$STAGE/README.md" <<'EOR'
---
title: Forest-Loss Detection Error Analysis
emoji: 🌳
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# Forest-Loss Detection — Error Analysis

An interactive Streamlit app for exploring where and why six deep-learning models
(U-Net, 3D U-Net, U-TAE, and Prithvi-based variants) succeed or fail at detecting
forest loss from Harmonized Landsat/Sentinel-2 imagery.

Data: [mqraitem/Deforest-Analysis](https://huggingface.co/datasets/mqraitem/Deforest-Analysis)
(588 held-out 224×224 tiles; ground truth from GLAD-S2 alerts).
EOR
hf repo create "$SPACE" --repo-type space --space-sdk docker $PRIV --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space
echo "Live at: https://huggingface.co/spaces/$SPACE"
