"""
Step 1: Train the risk detector.
Trains on draw_acctivations_shuf1k.jsonl to predict hallucination probability from SAE features.

Usage:
    python scripts/step1_train_raglens.py \
        --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
        --sae_name EleutherAI/sae-llama-3-8b-32x \
        --hookpoint layers.21.mlp \
        --data_path data/draw_acctivations_shuf1k.jsonl \
        --output_dir outputs/detector_model \
        --top_k 1000
"""

import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sparsify import Sae
from detector import RiskDetector
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"step1_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Step1] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {log_file}")


def load_data(data_path):
    """Load ParamMute data and convert to risk-detector format."""
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    inputs = [d["prompt_w_context"] for d in raw_data]
    outputs = [d["pred"] for d in raw_data]
    # Flip label: ParamMute is_faithful=0 means unfaithful, risk-detector label=1 means hallucination
    labels = [1 - d["is_faithful"] for d in raw_data]

    logging.info(f"Loaded {len(inputs)} samples")
    logging.info(f"  Hallucination (label=1): {sum(labels)}")
    logging.info(f"  Faithful (label=0): {len(labels) - sum(labels)}")

    return inputs, outputs, labels


def train_and_evaluate(args):
    setup_logging(args.output_dir)

    logging.info("=" * 60)
    logging.info("Step 1: Train risk hallucination detector")
    logging.info("=" * 60)
    logging.info(f"Args: {vars(args)}")

    inputs, outputs, labels = load_data(args.data_path)

    split_idx = int(len(inputs) * 0.8)
    train_inputs, test_inputs = inputs[:split_idx], inputs[split_idx:]
    train_outputs, test_outputs = outputs[:split_idx], outputs[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    logging.info(f"Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    logging.info(f"Loading LLM: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    logging.info(f"Loading SAE: {args.sae_name}, hookpoint: {args.hookpoint}")
    sae = Sae.load_from_disk(args.sae_name + '/' + args.hookpoint)
    sae.cfg.transcode = "transcoder" in args.sae_name
    sae = sae.to(model.device)
    sae.eval()

    logging.info(f"Training risk (top_k={args.top_k})...")
    _detector = RiskDetector(
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hookpoint=args.hookpoint,
        top_k=args.top_k,
    )

    feature_save_path = os.path.join(args.output_dir, "train_features.npy")

    detector.fit(
        inputs=train_inputs,
        outputs=train_outputs,
        labels=train_labels,
        save_path=feature_save_path,
    )

    logging.info("Evaluating on test set...")
    preds = detector.predict(inputs=test_inputs, outputs=test_outputs)
    probas = detector.predict_proba(inputs=test_inputs, outputs=test_outputs)

    balanced_acc = balanced_accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average="macro")

    logging.info(f"{'='*40}")
    logging.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    logging.info(f"  Macro F1:          {macro_f1:.4f}")
    logging.info(f"{'='*40}")

    results = {
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "llm_name": args.llm_name,
        "sae_name": args.sae_name,
        "hookpoint": args.hookpoint,
        "top_k": args.top_k,
        "train_size": len(train_inputs),
        "test_size": len(test_inputs),
        "test_probas": probas.tolist(),
        "test_labels": test_labels,
        "test_preds": preds.tolist(),
    }

    results_path = os.path.join(args.output_dir, "detector_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    indices_path = os.path.join(args.output_dir, "top_k_indices.npy")
    np.save(indices_path, detector.top_k_indices)
    logging.info(f"Top-K feature indices saved to {indices_path}")

    raglens_save_path = os.path.join(args.output_dir, "detector_clf.pkl")
    with open(raglens_save_path, "wb") as f:
        pickle.dump({"clf": detector.clf, "top_k_indices": detector.top_k_indices}, f)
    logging.info(f"RiskDetector GAM model saved to {raglens_save_path}")

    logging.info("Step 1 complete.")
    logging.info("Next: run step2_annotate_lambda.py to annotate training data with per-sample lambda.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Train risk hallucination detector")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--sae_name", type=str, default="EleutherAI/sae-llama-3-8b-32x")
    parser.add_argument("--hookpoint", type=str, default="layers.21.mlp")
    parser.add_argument("--data_path", type=str, default="data/draw_acctivations_shuf1k.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/detector_model")
    parser.add_argument("--top_k", type=int, default=1000)
    args = parser.parse_args()
    train_and_evaluate(args)
