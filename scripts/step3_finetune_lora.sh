#!/bin/bash
# ============================================================
# Step 3: LoRA fine-tuning with per-sample adaptive lambda.
# Under each sample's own lambda_i, do ParamMute-style knowledge-alignment:
#   loss = alpha * L_KAT + beta * L_KPO
#   L_KAT: log p(answer | context)              (rag_loss)
#   L_KPO: relu(rag_loss - raw_loss + margin)   (context prior)
#   Difference vs ParamMute: inhibit_strength is fixed 0 -> per-sample lambda_i.
#
# Prerequisites:
#   1) step1_train_detector.py finished (outputs/detector_model/raglens_clf.pkl)
#   2) step2_annotate_lambda.py annotated the training set (jsonl with lambda field)
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ==================== Config (overridable via env from run_all.sh) ====================
NUM_GPUS="${NUM_GPUS:-2}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-outputs/pip_kag_train_with_lambda.jsonl}"   # step2 output
OUTPUT_DIR="${OUTPUT_DIR:-outputs/apks_lora}"
INHIBIT_LAYERS="${INHIBIT_LAYERS:-21 22 23 24 25 26}"

mkdir -p "$OUTPUT_DIR"

RESUME_ARG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
    echo ">>> Found checkpoint: $LATEST_CKPT, resuming training"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
    echo ">>> No checkpoint found, starting from scratch"
fi

torchrun --nproc_per_node="$NUM_GPUS" src/train.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --max_len 1024 \
    --train_mode adaptive_input_contrastive \
    --model_type LlamaForInputContrastivew_act_inhibit \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --inhibit_strength 0 \
    --inhibit_layer_list $INHIBIT_LAYERS \
    --alpha 0.5 \
    --beta 0.5 \
    --initial_margin 1 \
    --final_margin 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.00001 \
    --bf16 True \
    --save_steps 300 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to none \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_ARG

echo "Fine-tuning complete. LoRA adapter saved to $OUTPUT_DIR"
echo "Next: step4 loads base + this LoRA adapter (adapter off = score risk, adapter on = generate)"
