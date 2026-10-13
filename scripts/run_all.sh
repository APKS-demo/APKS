#!/bin/bash
# ============================================================
# APKS (Adaptive Parametric Knowledge Suppression) full pipeline
# ------------------------------------------------------------
# Train detector once and freeze -> offline-annotate training set with per-sample lambda ->
# LoRA fine-tune with those fixed lambdas (detector not involved) -> at inference, score risk
# on the BASE model with the frozen detector, then regenerate with the fine-tuned model.
#
# Stages:
#   [1/4] Train risk detector (base-model activations)          step1_train_detector.py
#   [2/4] Annotate fine-tuning set with per-sample lambda       step2_annotate_lambda.py
#   [3/4] LoRA fine-tuning with per-sample lambda               step3_finetune_lora.sh
#   [4/4] Two-pass inference evaluation                         step4_evaluate_dynamic.py
#
# Toggle stages below; e.g. set everything false except RUN_EVAL to re-run evaluation only.
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ==================== Stage switches ====================
RUN_TRAIN_DETECTOR=false
RUN_ANNOTATE=false
RUN_FINETUNE=true
RUN_EVAL=true

# ==================== Configuration ====================
LLM_NAME=".../Meta-Llama-3-8B-Instruct"
SAE_NAME=".../sae-llama-3-8b-32x"
HOOKPOINT="layers.21"
INHIBIT_LAYERS="21 22 23 24 25 26"
KAPPA=1.0                  # lambda mapping: lambda=max(0,1-kappa*r)
MAX_NEW_TOKENS=32
SCHEMA="base"
TOP_K=1000
NUM_GPUS=2

# ---- Paths ----
DETECTOR_TRAIN_DATA="data/draw_acctivations_shuf1k.jsonl"          # [1] has is_faithful labels
FT_RAW_DATA="data/ParamMute-Training-Data/pip_kag_train.jsonl"     # [2] rag_input/raw_input/output
FT_LAMBDA_DATA="outputs/pip_kag_train_with_lambda.jsonl"           # [2]->[3]
EVAL_DATA_DIR="data/CoConflictQA/test"                             # [4]
DETECTOR_DIR="outputs/detector_model"                                # [1] output
ADAPTER_DIR="outputs/apks_lora"                                    # [3] output -> [4] input

GLOBAL_LOG="outputs/run_all_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs

echo "============================================" | tee "$GLOBAL_LOG"
echo "  APKS full pipeline" | tee -a "$GLOBAL_LOG"
echo "  Time:      $(date)" | tee -a "$GLOBAL_LOG"
echo "  Base model: $LLM_NAME" | tee -a "$GLOBAL_LOG"
echo "  SAE:        $SAE_NAME" | tee -a "$GLOBAL_LOG"
echo "  Hookpoint:  $HOOKPOINT" | tee -a "$GLOBAL_LOG"
echo "  Inhibit:    $INHIBIT_LAYERS" | tee -a "$GLOBAL_LOG"
echo "  kappa:      $KAPPA" | tee -a "$GLOBAL_LOG"
echo "  Stages:     detector=$RUN_TRAIN_DETECTOR annotate=$RUN_ANNOTATE finetune=$RUN_FINETUNE eval=$RUN_EVAL" | tee -a "$GLOBAL_LOG"
echo "  Log:        $GLOBAL_LOG" | tee -a "$GLOBAL_LOG"
echo "============================================" | tee -a "$GLOBAL_LOG"

# ==================== [1/4] Train Risk Detector ====================
if [ "$RUN_TRAIN_DETECTOR" = true ]; then
    echo "" | tee -a "$GLOBAL_LOG"
    echo ">>> [1/4] Training Risk Detector..." | tee -a "$GLOBAL_LOG"
    echo "    Start: $(date)" | tee -a "$GLOBAL_LOG"
    python scripts/step1_train_detector.py \
        --llm_name "$LLM_NAME" \
        --sae_name "$SAE_NAME" \
        --hookpoint "$HOOKPOINT" \
        --data_path "$DETECTOR_TRAIN_DATA" \
        --output_dir "$DETECTOR_DIR" \
        --top_k "$TOP_K" \
        2>&1 | tee -a "$GLOBAL_LOG"
    echo "    End: $(date)" | tee -a "$GLOBAL_LOG"
else
    echo ">>> [1/4] Skipped (RUN_TRAIN_DETECTOR=false)" | tee -a "$GLOBAL_LOG"
fi

# ==================== [2/4] Annotate training set with per-sample lambda ====================
if [ "$RUN_ANNOTATE" = true ]; then
    echo "" | tee -a "$GLOBAL_LOG"
    echo ">>> [2/4] Annotating training set with per-sample lambda (offline, frozen detector)..." | tee -a "$GLOBAL_LOG"
    echo "    Start: $(date)" | tee -a "$GLOBAL_LOG"
    python scripts/step2_annotate_lambda.py \
        --llm_name "$LLM_NAME" \
        --sae_name "$SAE_NAME" \
        --hookpoint "$HOOKPOINT" \
        --detector_dir "$DETECTOR_DIR" \
        --train_data_path "$FT_RAW_DATA" \
        --output_path "$FT_LAMBDA_DATA" \
        --kappa "$KAPPA" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --schema "$SCHEMA" \
        2>&1 | tee -a "$GLOBAL_LOG"
    echo "    End: $(date)" | tee -a "$GLOBAL_LOG"
else
    echo ">>> [2/4] Skipped (RUN_ANNOTATE=false)" | tee -a "$GLOBAL_LOG"
fi

# ==================== [3/4] LoRA fine-tuning with per-sample lambda ====================
if [ "$RUN_FINETUNE" = true ]; then
    echo "" | tee -a "$GLOBAL_LOG"
    echo ">>> [3/4] LoRA fine-tuning with per-sample lambda (adaptive_input_contrastive)..." | tee -a "$GLOBAL_LOG"
    echo "    Start: $(date)" | tee -a "$GLOBAL_LOG"
    NUM_GPUS="$NUM_GPUS" \
    BASE_MODEL="$LLM_NAME" \
    TRAIN_FILE="$FT_LAMBDA_DATA" \
    OUTPUT_DIR="$ADAPTER_DIR" \
    INHIBIT_LAYERS="$INHIBIT_LAYERS" \
        bash scripts/step3_finetune_lora.sh \
        2>&1 | tee -a "$GLOBAL_LOG"
    echo "    End: $(date)" | tee -a "$GLOBAL_LOG"
else
    echo ">>> [3/4] Skipped (RUN_FINETUNE=false)" | tee -a "$GLOBAL_LOG"
fi

# ==================== [4/4] Two-pass inference evaluation ====================
if [ "$RUN_EVAL" = true ]; then
    echo "" | tee -a "$GLOBAL_LOG"
    echo ">>> [4/4] Two-pass inference evaluation (APKS vs fixed lambda=0 vs vanilla)..." | tee -a "$GLOBAL_LOG"
    echo "    Start: $(date)" | tee -a "$GLOBAL_LOG"
    python scripts/step4_evaluate_dynamic.py \
        --model_name "$LLM_NAME" \
        --sae_name "$SAE_NAME" \
        --hookpoint "$HOOKPOINT" \
        --detector_dir "$DETECTOR_DIR" \
        --train_data_path "$DETECTOR_TRAIN_DATA" \
        --eval_data_dir "$EVAL_DATA_DIR" \
        --inhibit_layer_list $INHIBIT_LAYERS \
        --adapter_dir "$ADAPTER_DIR" \
        --kappa "$KAPPA" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --schema "$SCHEMA" \
        --output_dir outputs/dynamic_eval \
        2>&1 | tee -a "$GLOBAL_LOG"
    echo "    End: $(date)" | tee -a "$GLOBAL_LOG"
else
    echo ">>> [4/4] Skipped (RUN_EVAL=false)" | tee -a "$GLOBAL_LOG"
fi

echo "" | tee -a "$GLOBAL_LOG"
echo "============================================" | tee -a "$GLOBAL_LOG"
echo "  All done. $(date)" | tee -a "$GLOBAL_LOG"
echo "  Results in outputs/" | tee -a "$GLOBAL_LOG"
echo "  Log: $GLOBAL_LOG" | tee -a "$GLOBAL_LOG"
echo "============================================" | tee -a "$GLOBAL_LOG"
