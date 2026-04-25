#!/usr/bin/env bash
set -euo pipefail

cd ~/repos/hypercorpus
export PATH="$HOME/.local/bin:$PATH"

EXPORT_DIR="data/mdr-iirc-export"
TRAIN_DIR="data/mdr-iirc-trained"
INDEX_DIR="data/mdr-iirc-index"
LOG_DIR="logs"

mkdir -p "$LOG_DIR" "$TRAIN_DIR" "$INDEX_DIR"

echo "[$(date)] === MDR Pipeline Start ==="

# --- Step 1: Wait for export to finish ---
EXPORT_MANIFEST="$EXPORT_DIR/mdr_export_manifest.json"
echo "[$(date)] Waiting for export manifest: $EXPORT_MANIFEST"
while [ ! -f "$EXPORT_MANIFEST" ]; do
	sleep 30
	echo "[$(date)] Still waiting for export..."
done
echo "[$(date)] Export complete!"
ls -lh "$EXPORT_DIR/"

# --- Step 2: Install MDR dependencies ---
echo "[$(date)] === Installing MDR dependencies ==="
uv pip install transformers==2.11.0 tensorboard numpy tqdm ujson torch faiss-cpu 2>&1 || {
	echo "[$(date)] WARNING: Some MDR deps failed to install. Trying without version pin..."
	uv pip install transformers tensorboard numpy tqdm ujson torch faiss-cpu 2>&1 || true
}

# --- Step 3: Train MDR ---
echo "[$(date)] === Training MDR ==="
uv run hypercorpus baselines train-mdr \
	--export-manifest "$EXPORT_DIR/mdr_export_manifest.json" \
	--output-dir "$TRAIN_DIR" \
	--model-name roberta-base \
	2>&1 | tee "$LOG_DIR/mdr-train.log"
echo "[$(date)] Training complete!"

# --- Step 4: Build index ---
echo "[$(date)] === Building MDR Index ==="
uv run hypercorpus baselines build-mdr-index \
	--export-manifest "$EXPORT_DIR/mdr_export_manifest.json" \
	--train-manifest "$TRAIN_DIR/mdr_train_manifest.json" \
	--output-dir "$INDEX_DIR" \
	2>&1 | tee "$LOG_DIR/mdr-index.log"
echo "[$(date)] Index build complete!"

# --- Step 5: Run evaluation ---
CASE_IDS_FILE="runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt"
SELECTORS="external__mdr__iirc_finetuned,top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop,gold_support_context"

echo "[$(date)] === Running MDR Evaluation ==="
if [ -f "$CASE_IDS_FILE" ]; then
	echo "[$(date)] Using case IDs from $CASE_IDS_FILE ($(wc -l < "$CASE_IDS_FILE") cases)"
	uv run hypercorpus experiments run-iirc-store \
		--store dataset/iirc/store \
		--selectors "$SELECTORS" \
		--mdr-artifact-manifest "$INDEX_DIR/mdr_artifact_manifest.json" \
		--case-ids-file "$CASE_IDS_FILE" \
		--budget-ratios "0.01,0.02,0.05,0.10,1.0" \
		--exp-name iirc-mdr-trained-v1 \
		--no-e2e --no-export-graphrag-inputs \
		2>&1 | tee "$LOG_DIR/iirc-mdr-trained-v1.log"
else
	echo "[$(date)] Case IDs file not found, using --case-limit 100"
	uv run hypercorpus experiments run-iirc-store \
		--store dataset/iirc/store \
		--selectors "$SELECTORS" \
		--mdr-artifact-manifest "$INDEX_DIR/mdr_artifact_manifest.json" \
		--case-limit 100 \
		--budget-ratios "0.01,0.02,0.05,0.10,1.0" \
		--exp-name iirc-mdr-trained-v1 \
		--no-e2e --no-export-graphrag-inputs \
		2>&1 | tee "$LOG_DIR/iirc-mdr-trained-v1.log"
fi

echo "[$(date)] === MDR Pipeline Complete ==="
