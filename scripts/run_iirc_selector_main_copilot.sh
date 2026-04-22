#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

log() {
	printf '[%s] %s\n' "$(date -Is)" "$*"
}

require_file() {
	local path="$1"
	if [[ ! -f "$path" ]]; then
		printf 'Missing required file: %s\n' "$path" >&2
		exit 1
	fi
}

if [[ "${RESUME:-0}" == "1" && "${RESTART:-0}" == "1" ]]; then
	printf 'RESUME=1 and RESTART=1 are mutually exclusive.\n' >&2
	exit 2
fi

exp_name="${EXP_NAME:-iirc-selector-main-copilot-v1}"
store_uri="${STORE_URI:-dataset/iirc/store}"
split="${SPLIT:-dev}"
case_ids_file="${CASE_IDS_FILE:-runs/iirc-sample-s100-dense-v1/chunks/chunk-00000/evaluated_case_ids.txt}"
chunk_size="${CHUNK_SIZE:-20}"
start_chunk="${START_CHUNK:-0}"
end_chunk="${END_CHUNK:-4}"
selector_provider="${SELECTOR_PROVIDER:-copilot}"
selector_model="${SELECTOR_MODEL:-gpt-4.1}"
sentence_transformer_device="${SENTENCE_TRANSFORMER_DEVICE:-cuda}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S="${HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S:-30}"
export HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S="${HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S:-900}"
export HYPERCORPUS_BUDGET_FILL_TIMEOUT_S="${HYPERCORPUS_BUDGET_FILL_TIMEOUT_S:-240}"

cd "$repo_root"
require_file "$case_ids_file"

run_args=(
	run
	hypercorpus
	experiments
	run-iirc-store
	--store
	"$store_uri"
	--exp-name
	"$exp_name"
	--split
	"$split"
	--study-preset
	iirc_selector_main
	--case-ids-file
	"$case_ids_file"
	--chunk-size
	"$chunk_size"
	--selector-provider
	"$selector_provider"
	--selector-model
	"$selector_model"
	--sentence-transformer-device
	"$sentence_transformer_device"
	--no-e2e
	--no-export-graphrag-inputs
)

if [[ "${RESUME:-0}" == "1" ]]; then
	run_args+=(--resume)
fi
if [[ "${RESTART:-0}" == "1" ]]; then
	run_args+=(--restart)
fi

log "START exp_name=${exp_name} store=${store_uri} split=${split}"
log "CONFIG chunk_size=${chunk_size} chunk_range=${start_chunk}-${end_chunk} provider=${selector_provider} model=${selector_model}"
log "CONFIG heartbeat_s=${HYPERCORPUS_SELECTION_HEARTBEAT_INTERVAL_S} stage_timeout_s=${HYPERCORPUS_SELECTION_STAGE_TIMEOUT_S} budget_fill_timeout_s=${HYPERCORPUS_BUDGET_FILL_TIMEOUT_S}"

for ((idx = start_chunk; idx <= end_chunk; idx += 1)); do
	log "CHUNK_START idx=${idx}"
	uv "${run_args[@]}" --chunk-index "$idx"
	log "CHUNK_DONE idx=${idx}"
done

log "MERGE_START run_dir=runs/${exp_name}"
uv run hypercorpus experiments merge-iirc-results --run-dir "runs/${exp_name}"
log "MERGE_DONE run_dir=runs/${exp_name}"
log "DONE exp_name=${exp_name}"
