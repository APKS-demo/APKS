# APKS: Adaptive Parametric Knowledge Suppression for Faithful Retrieval-Augmented Generation

Official implementation of **Adaptive Parametric Knowledge Suppression for Faithful Retrieval-Augmented Generation**.

APKS improves faithfulness in retrieval-augmented generation by adaptively suppressing unfaithfulness-associated feed-forward networks (UA-FFNs) based on a per-instance risk estimated by an interpretable sparse-autoencoder detector.

## Overview

- **Risk detector**: estimates the unfaithfulness risk of a response from sparse-autoencoder features.
- **Adaptive suppression**: maps the risk to a per-instance suppression coefficient for UA-FFNs.
- **Knowledge-enhanced fine-tuning**: fine-tunes the model under the same adaptive suppression used at inference.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

All datasets are from the [ParamMute](https://github.com/zjunlp/ParamMute) repository. Download or prepare the datasets and place them under `data/`:

```
data/
├── CoConflictQA/
│   └── test/           # 6-subset test set (NQ, NewsQA, SQuAD, SearchQA, TriviaQA-web, HotpotQA)
├── draw_acctivations_shuf1k.jsonl   # Risk detector training data, repurposed from ParamMute's UA-FFN identification dataset
└── ParamMute-Training-Data/
    └── pip_kag_train.jsonl           # LoRA fine-tuning corpus
```

## Usage

### 1. Train the risk detector

```bash
python scripts/step1_train_detector.py \
  --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
  --sae_name EleutherAI/sae-llama-3-8b-32x \
  --hookpoint layers.21 \
  --data_path data/draw_acctivations_shuf1k.jsonl \
  --output_dir outputs/detector_model
```

### 2. Annotate training set with per-sample λ

```bash
python scripts/step2_annotate_lambda.py \
  --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
  --sae_name EleutherAI/sae-llama-3-8b-32x \
  --hookpoint layers.21 \
  --detector_dir outputs/detector_model \
  --train_data_path data/ParamMute-Training-Data/pip_kag_train.jsonl \
  --output_path outputs/pip_kag_train_with_lambda.jsonl
```

### 3. Fine-tune the model with adaptive suppression

Output saved to `outputs/apks_lora/`.

```bash
bash scripts/step3_finetune_lora.sh
```

### 4. Evaluate (two-pass inference with dynamic λ)

```bash
python scripts/step4_evaluate_dynamic.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --sae_name EleutherAI/sae-llama-3-8b-32x \
  --hookpoint layers.21 \
  --detector_dir outputs/detector_model \
  --eval_data_dir data/CoConflictQA/test \
  --adapter_dir outputs/apks_lora \
  --kappa 1.0 \
  --output_dir outputs/dynamic_eval
```

Or run the full pipeline with `scripts/run_all.sh` (toggle stages at the top of the script).

## Results

APKS improves context recall on all subsets of CoConflictQA. See the paper for details.
