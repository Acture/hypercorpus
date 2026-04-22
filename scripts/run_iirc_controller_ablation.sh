#!/usr/bin/env bash
#
# Run the IIRC controller on 100 cases with a specific LLM model.
# Includes dense + gold_support_context in the same surface for comparison.
#
# Usage:
#   bash scripts/run_iirc_controller_ablation.sh                  # gpt-5, all chunks
#   SELECTOR_MODEL=gpt-4.1 bash scripts/run_iirc_controller_ablation.sh
#   SELECTOR_MODEL=gpt-4.1-mini bash scripts/run_iirc_controller_ablation.sh
#   START_CHUNK=0 END_CHUNK=0 bash scripts/run_iirc_controller_ablation.sh  # chunk 0 only (20 cases, quick sanity check)
#   RESUME=1 bash scripts/run_iirc_controller_ablation.sh         # resume interrupted run

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"; }

require_file() {
	if [[ ! -f "$1" ]]; then
		printf 'Missing required file: %s\n' "$1" >&2
		exit 1
	fi
}

if [[ "${RESUME:-0}" == "1" && "${RESTART:-0}" == "1" ]]; then
	printf 'RESUME=1 and RESTART=1 are mutually exclusive.\n' >&2
	exit 2
fi

# ── Configuration ────────────────────────────────────────────────
selector_model="${SELECTOR_MODEL:-gpt-5}"
# Derive exp name from model: gpt-5 → gpt5, gpt-4.1-mini → gpt41mini
model_slug="$(printf '%s' "$selector_model" | tr -d '.-')"
exp_name="${EXP_NAME:-iirc-controller-100-${model_slug}-v1}"

store_uri="${STORE_URI:-dataset/iirc/store}"
split="${SPLIT:-dev}"
case_ids_file="${CASE_IDS_FILE:-runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt}"
chunk_size="${CHUNK_SIZE:-20}"
start_chunk="${START_CHUNK:-0}"
end_chunk="${END_CHUNK:-4}"
selector_provider="${SELECTOR_PROVIDER:-copilot}"
sentence_transformer_device="${SENTENCE_TRANSFORMER_DEVICE:-mps}"

# Controller + dense baseline + oracle, comma-separated
selectors="top_1_seed__sentence_transformer__hop_adaptive__constrained_multipath__link_context_llm_controller__lookahead_2"
selectors="${selectors},top_1_seed__sentence_transformer__hop_0__dense__budget_fill_relative_drop"
selectors="${selectors},gold_support_context"

budget_ratios="0.01,0.02,0.05,0.10,1.0"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S="${HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S:-30}"
export HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S="${HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S:-900}"
export HYPERCORPUS_BUDGET_FILL_TIMEOUT_S="${HYPERCORPUS_BUDGET_FILL_TIMEOUT_S:-240}"
export HYPERCORPUS_COPILOT_TIMEOUT_S="${HYPERCORPUS_COPILOT_TIMEOUT_S:-300}"

# ── Validation ───────────────────────────────────────────────────
cd "$repo_root"
require_file "$case_ids_file"

run_args=(
	run
	hypercorpus
	experiments
	run-iirc-store
	--store "$store_uri"
	--exp-name "$exp_name"
	--split "$split"
	--selectors "$selectors"
	--budget-ratios "$budget_ratios"
	--case-ids-file "$case_ids_file"
	--chunk-size "$chunk_size"
	--selector-provider "$selector_provider"
	--selector-model "$selector_model"
	--sentence-transformer-device "$sentence_transformer_device"
	--no-e2e
	--no-export-graphrag-inputs
)

if [[ "${RESUME:-0}" == "1" ]]; then
	run_args+=(--resume)
fi
if [[ "${RESTART:-0}" == "1" ]]; then
	run_args+=(--restart)
fi

# ── Run ──────────────────────────────────────────────────────────
log "START exp_name=${exp_name} model=${selector_model} provider=${selector_provider}"
log "CONFIG chunks=${start_chunk}-${end_chunk} (${chunk_size}/chunk) device=${sentence_transformer_device}"
log "CONFIG heartbeat=${HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S}s stage_timeout=${HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S}s budget_fill_timeout=${HYPERCORPUS_BUDGET_FILL_TIMEOUT_S}s"

for ((idx = start_chunk; idx <= end_chunk; idx += 1)); do
	log "CHUNK_START idx=${idx}"
	chunk_args=("${run_args[@]}" --chunk-index "$idx")
	# Only pass --resume when the chunk already has artifacts
	chunk_dir="runs/${exp_name}/chunks/chunk-$(printf '%05d' "$idx")"
	if [[ "${RESUME:-0}" == "1" && -f "${chunk_dir}/run_state.json" ]]; then
		chunk_args+=(--resume)
	fi
	uv "${chunk_args[@]}"
	log "CHUNK_DONE idx=${idx}"
done

log "MERGE_START run_dir=runs/${exp_name}"
uv run hypercorpus experiments merge-iirc-results --run-dir "runs/${exp_name}"
log "MERGE_DONE run_dir=runs/${exp_name}"
log "DONE exp_name=${exp_name}"
