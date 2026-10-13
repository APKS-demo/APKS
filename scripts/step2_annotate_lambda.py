"""
Step 2: Annotate training set with per-sample lambda (offline preprocessing before APKS fine-tuning).

For each sample in the fine-tuning corpus, use the frozen base model + the step-1 detector
to assign a suppression coefficient lambda_i:

    draft_i = base_model(context_i, question_i)
    r_i     = RiskDetector.predict_proba(raglens_prompt_i, draft_i)
    lambda_i = max(0, 1 - kappa * r_i)

Output appends a `lambda` field to each record, consumed by InputContrastiveLambdaDataset.

Design:
  * The detector always runs on the BASE model — both during annotation (here) and inference
    (step4), so there is no distribution drift between detector/SAE contexts.
  * Lambda construction (draft via base chat template, risk via risk-detector prompt) is identical
    to step4, ensuring training-time lambda distribution == inference-time lambda distribution.

Usage:
    python scripts/step2_annotate_lambda.py \
        --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
        --sae_name EleutherAI/sae-llama-3-8b-32x \
        --hookpoint layers.21.mlp \
        --detector_dir outputs/detector_model \
        --train_data_path data/ParamMute-Training-Data/pip_kag_train.jsonl \
        --output_path outputs/pip_kag_train_with_lambda.jsonl \
        --kappa 1.0 \
        --max_new_tokens 32
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
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sparsify import Sae
from detector import RiskDetector
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"annotate_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Annotate] %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_file}")


def build_prompt(query, context, schema="base"):
    """Draft-generation prompt matching step3 build_prompt base schema."""
    if schema == "base":
        return f"{context}\nQ: {query}\nA: "
    elif schema == "opin":
        context = context.replace('"', "")
        return f'Bob said "{context}"\nQ: {query[:-1]} in Bob\'s opinion?\nA: '
    elif schema == "attr":
        return f"{context}\nQ:{query[:-1]} based on the given tex?\nA:"
    elif schema == "instr":
        return f"Instruction: read the given information and answer the corresponding question.\n\n{context}\nQ: {query}\nA: "
    return f"{context}\nQ: {query}\nA: "


def build_risk_prompt(query, context):
    """RiskDetector prompt, identical to step1 training-data prompt_w_context."""
    return (
        f"Answer the question with as few words as possible "
        f"through the information given in the context.\n\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Answer: "
    )


# Draft prompt must be byte-identical to src/dataset.py user_format so the fine-tuned
# model sees the template at inference time that it saw during training.
# content = rag_input = "{context}\nQ: {question}", tokenize with add_special_tokens=False (no BOS).
_LLAMA3_USER_FMT = (
    "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

def build_gen_prompt(query, context):
    content = f"{context}\nQ: {query}"
    return _LLAMA3_USER_FMT.format(content=content)


def resolve_context_question(d):
    """Parse (context, question) from a training record.

    1) Prefer explicit context/question fields (CoConflictQA-style).
    2) Otherwise reverse-engineer from ParamMute training format (validated:
       rag_input always ends with raw_input):
         raw_input = "Q: {question}"
         rag_input = "{context}\\nQ: {question}"
       => question = raw_input with "Q: " prefix stripped
          context  = rag_input with trailing raw_input removed
    3) Return (None, None) if neither works; caller falls back to rag_input.
    """
    ctx = d.get("context")
    q = d.get("question")
    if ctx is not None and q is not None:
        return ctx, q

    rag = d.get("rag_input")
    raw = d.get("raw_input")
    if rag is not None and raw is not None:
        raw_s = raw.strip()
        question = raw_s[len("Q:"):].strip() if raw_s.startswith("Q:") else raw_s
        rag_s = rag.rstrip()
        if rag_s.endswith(raw_s):
            context = rag_s[: len(rag_s) - len(raw_s)].rstrip()
        else:
            idx = rag_s.rfind("\nQ:")
            context = rag_s[:idx].rstrip() if idx != -1 else rag_s
        return context, question

    return None, None


def generate_draft(model, tokenizer, prompt, max_new_tokens=32):
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
        )
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main(args):
    output_dir = os.path.dirname(args.output_path) or "."
    setup_logging(output_dir)
    logging.info("=" * 60)
    logging.info("Annotate training set with per-sample lambda")
    logging.info("=" * 60)
    logging.info(f"Args: {vars(args)}")

    logging.info(f"Loading base LLM: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    logging.info(f"Loading SAE: {args.sae_name}, hookpoint: {args.hookpoint}")
    sae = Sae.load_from_disk(args.sae_name + '/' + args.hookpoint)
    sae.cfg.transcode = "transcoder" in args.sae_name
    sae = sae.to(model.device)
    sae.eval()

    _detector = RiskDetector(tokenizer=tokenizer, model=model, sae=sae, hookpoint=args.hookpoint)
    raglens_pkl_path = os.path.join(args.detector_dir, "detector_clf.pkl")
    if not os.path.exists(raglens_pkl_path):
        raise FileNotFoundError(
            f"Detector not found at {raglens_pkl_path}. Run step1_train_raglens.py first."
        )
    logging.info(f"Loading RiskDetector GAM from {raglens_pkl_path}")
    with open(raglens_pkl_path, "rb") as f:
        saved = pickle.load(f)
    detector.clf = saved["clf"]
    detector.top_k_indices = saved["top_k_indices"]

    with open(args.train_data_path, "r") as f:
        data = [json.loads(line) for line in f]
    logging.info(f"Training samples: {len(data)}")

    sample0 = data[0]
    logging.info(f"Record fields: {sorted(sample0.keys())}")
    ctx0, q0 = resolve_context_question(sample0)
    if ctx0 is None:
        raise KeyError(
            "Training data has neither context/question nor parseable rag_input/raw_input. "
            "Cannot compute lambda."
        )

    lambdas, risks = [], []
    n_warn = 0
    with open(args.output_path, "w") as fout:
        for d in tqdm(data, desc="annotate lambda"):
            context, question = resolve_context_question(d)
            if context is not None:
                gen_prompt = build_gen_prompt(question, context)
                raglens_prompt = build_risk_prompt(question, context)
            else:
                gen_prompt = _LLAMA3_USER_FMT.format(content=d["rag_input"])
                raglens_prompt = d["rag_input"]
                n_warn += 1

            draft = generate_draft(model, tokenizer, gen_prompt, max_new_tokens=args.max_new_tokens)
            risk = float(detector.predict_proba(inputs=[raglens_prompt], outputs=[draft])[0])
            lam = max(0.0, 1.0 - args.kappa * risk)

            out = dict(d)
            out["lambda"] = round(lam, 6)
            out["risk_score"] = round(risk, 6)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            lambdas.append(lam)
            risks.append(risk)

    logging.info(f"{'='*50}")
    logging.info(f"Done: {len(lambdas)} samples annotated")
    logging.info(f"Risk r:  mean={np.mean(risks):.4f} std={np.std(risks):.4f} "
                 f"min={np.min(risks):.4f} max={np.max(risks):.4f}")
    logging.info(f"Lambda:  mean={np.mean(lambdas):.4f} std={np.std(lambdas):.4f} "
                 f"min={np.min(lambdas):.4f} max={np.max(lambdas):.4f}")
    if n_warn:
        logging.warning(f"{n_warn} samples fell back to raw rag_input")
    logging.info(f"Lambda-annotated data saved to {args.output_path}")
    logging.info("Next: run scripts/step3_finetune_lora.sh for adaptive_input_contrastive fine-tuning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate training set with per-sample lambda")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--sae_name", type=str, default="EleutherAI/sae-llama-3-8b-32x")
    parser.add_argument("--hookpoint", type=str, default="layers.21.mlp")
    parser.add_argument("--detector_dir", type=str, default="outputs/detector_model")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/pip_kag_train_with_lambda.jsonl")
    parser.add_argument("--kappa", type=float, default=1.0, help="Lambda mapping: lambda=max(0,1-kappa*r)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--schema", type=str, default="base")
    args = parser.parse_args()
    main(args)
