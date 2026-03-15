#!/usr/bin/env bash

set -euo pipefail

# Configuration
N="${1:-1}"
MODEL_PATH="artifacts/results/best_model.pt"
DATASET_DIR="dataset"
EVAL_DIR="artifacts/eval"
RAGAS_DIR="artifacts/ragas"
RAGAS_JSONL="${RAGAS_DIR}/cases.jsonl"
LLM_MODEL="ministral-3:3b"
CASE_COUNTER=1

# Stego class name patterns — must match embedded classname in filenames
# Filename format: image_<id>_<classname>_<n>.png
declare -A STEGO_CLASSES=(
    [js]="*_js_*"
    [html]="*_html_*"
    [ps]="*_ps_*"
    [eth]="*_eth_*"
    [url]="*_url_*"
)

log() { echo "[run.sh] $*"; }
die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }

# Find up to N images matching a glob, preferring test/ then val/ then train/
find_images_glob() {
    local dir="$1" pattern="$2" count="$3"
    find "$dir" -maxdepth 1 -type f \
        \( -iname "*.png" -o -iname "*.jpg" \) \
        -iname "$pattern" 2>/dev/null \
        | sort | head -n "$count"
}

run_case() {
    local img="$1" case_id="$2"
    log "  → ${case_id}: $(basename "$img")"
    python3 pipeline.py \
        --image      "$img"        \
        --model-path "$MODEL_PATH" \
        --output-dir "${EVAL_DIR}/${case_id}" \
        --case-id    "$case_id"
    echo "${EVAL_DIR}/${case_id}"
}

# Train if needed
if [[ ! -f "$MODEL_PATH" ]]; then
    log "No checkpoint at ${MODEL_PATH} — starting training …"
    python3 -m binary_classifier.train
    [[ -f "$MODEL_PATH" ]] || die "Training finished but ${MODEL_PATH} not found."
    log "Training complete."
else
    log "Checkpoint found — skipping training."
fi

# Verify Ollama
log "Checking Ollama …"
curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 \
    || die "Ollama is not running. Start with: ollama serve"

if ! ollama list 2>/dev/null | grep -q "${LLM_MODEL}"; then
    log "Pulling ${LLM_MODEL} …"
    ollama pull "${LLM_MODEL}" || die "Failed to pull ${LLM_MODEL}."
fi
log "Ollama OK — ${LLM_MODEL} available."

# Clean images
log "=== CLASS: clean (${N} image(s)) ==="
found=0
for split in test val train; do
    clean_dir="${DATASET_DIR}/${split}/clean"
    [[ -d "$clean_dir" ]] || continue
    while IFS= read -r img; do
        case_id=$(printf "CASE-%04d" "$CASE_COUNTER")
        run_case "$img" "$case_id"
        CASE_COUNTER=$((CASE_COUNTER + 1))
        found=$((found + 1))
        [[ $found -ge $N ]] && break 2
    done < <(find_images_glob "$clean_dir" "*" "$N")
done
[[ $found -eq 0 ]] && log "WARNING: No clean images found in ${DATASET_DIR}/*/clean/"

# Stego images per class
for class_name in js html ps eth url; do
    pattern="${STEGO_CLASSES[$class_name]}"
    log "=== CLASS: ${class_name} (${N} image(s), pattern: ${pattern}) ==="
    found=0
    for split in test val train; do
        stego_dir="${DATASET_DIR}/${split}/stego"
        [[ -d "$stego_dir" ]] || continue
        while IFS= read -r img; do
            case_id=$(printf "CASE-%04d" "$CASE_COUNTER")
            run_case "$img" "$case_id"
            CASE_COUNTER=$((CASE_COUNTER + 1))
            found=$((found + 1))
            [[ $found -ge $N ]] && break 2
        done < <(find_images_glob "$stego_dir" "$pattern" "$N")
    done
    [[ $found -eq 0 ]] && \
        log "WARNING: No ${class_name} images found — check dataset layout."
done

TOTAL=$((CASE_COUNTER - 1))
log "Pipeline complete: ${TOTAL} case(s) → ${EVAL_DIR}"

# Convert reports to RAGAS JSONL
log "Converting forensic reports to RAGAS format …"
mkdir -p "$RAGAS_DIR"
python3 -m evals.ragas_eval convert \
    --report-dir "$EVAL_DIR" \
    --output     "$RAGAS_JSONL"

CASE_COUNT=$(grep -c '' "$RAGAS_JSONL" 2>/dev/null || echo 0)
log "RAGAS cases: ${CASE_COUNT} written to ${RAGAS_JSONL}"

if [[ "$CASE_COUNT" -eq 0 ]]; then
    log "No stego cases to evaluate — skipping RAGAS."
    exit 0
fi

# RAGAS evaluation
log "Running RAGAS evaluation (judge: ${LLM_MODEL}) …"
python3 -m evals.ragas_eval eval \
    --cases-jsonl "$RAGAS_JSONL" \
    --output-dir  "$RAGAS_DIR"   \
    --model       "$LLM_MODEL"

log "=================================================================="
log "Done."
log "  Pipeline outputs : ${EVAL_DIR}/"
log "  RAGAS results    : ${RAGAS_DIR}/ragas_results.csv"
log "  RAGAS summary    : ${RAGAS_DIR}/ragas_summary.json"
log "=================================================================="
