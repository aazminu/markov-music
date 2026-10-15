#!/usr/bin/env bash
# Fetches the Kaggle Bach Chorales dataset.
# Prereqs:
#   1. Kaggle CLI installed (`pip install kaggle`)
#   2. `~/.kaggle/kaggle.json` present with your API token
#   3. `unzip` available on PATH
set -euo pipefail

DATA_DIR="data/bach_chorales"
ZIP_NAME="bach-chorales-2.zip"
ZIP_PATH="$DATA_DIR/$ZIP_NAME"

mkdir -p "$DATA_DIR"

echo "[fetch_bach_chorales] Downloading dataset into $DATA_DIR"
kaggle datasets download pranjalsriv/bach-chorales-2 -p "$DATA_DIR" --force

echo "[fetch_bach_chorales] Unzipping $ZIP_PATH"
unzip -o "$ZIP_PATH" -d "$DATA_DIR"

echo "[fetch_bach_chorales] Cleaning up zip archive"
rm -f "$ZIP_PATH"

echo "[fetch_bach_chorales] Done. Contents now in $DATA_DIR"