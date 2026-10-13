"""
Step 4: Two-pass inference evaluation with dynamic lambda (main experiment).
For each sample:
  Pass 1: base model (lambda=1) -> draft response
  RiskDetector scores hallucination risk r -> lambda = max(0, 1 - kappa * r)
  Pass 2: fine-tuned model with per-sample lambda -> final answer
Reports ConR / MemR / MR, comparing APKS dynamic lambda vs fixed lambda=0 vs vanilla.

Usage:
    python scripts/step4_evaluate_dynamic.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --sae_name EleutherAI/sae-llama-3-8b-32x \
        --hookpoint layers.21.mlp \
        --detector_dir outputs/detector_model \
        --eval_data_dir data/CoConflictQA/test \
        --inhibit_layer_list 21 22 23 24 25 26 \
        --adapter_dir outputs/apks_lora \
        --kappa 1.0 \
        --max_new_tokens 32 \
        --output_dir outputs/dynamic_eval
"""

import os
import sys
import json
import string
import re
import argparse
import glob
import logging
import pickle
import numpy as np
import torch
import jsonlines
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sparsify import Sae
from detector import RiskDetector
from modeling_llama import LlamaForCausalLM_w_act_inhibit
from peft import PeftModel
from contextlib import nullcontext


# ===================== Metric utilities (from ParamMute) =====================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def acc_score(prediction, ground_truth):
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1.0
    return 0.0


def compute_metrics(parametric_answers, pred_answers, gold_answers):
    """ConR (Context Recall up), MemR (Memory Recall down), MR (Memorization Ratio down)."""
    pm_acc_total, acc_total = 0, 0
    for pm_answer, pred, gold in zip(parametric_answers, pred_answers, gold_answers):
        if isinstance(gold, list):
            _acc = max(acc_score(pred, g) for g in gold)
        else:
            _acc = acc_score(pred, gold)

        _pm_acc = acc_score(pred, pm_answer)
        if _pm_acc == 1:
            _acc = 0

        pm_acc_total += _pm_acc
        acc_total += _acc

    n = len(pred_answers)
    con_r = acc_total * 100 / (n + 1e-5)
    mem_r = pm_acc_total * 100 / (n + 1e-5)
    mr = (mem_r / (mem_r + con_r + 1e-5)) * 100
    return round(con_r, 2), round(mem_r, 2), round(mr, 2)


# ===================== Prompt builders =====================

def build_prompt(query, context, schema="base", tokenizer=None, use_chat_template=False):
    """Inference prompt matching ParamMute eval_CoConflictQA.py qa_to_prompt_baseline."""
    if schema == "base":
        prompt = f"{context}\nQ: {query}\nA: "
    elif schema == "opin":
        context = context.replace('"', "")
        prompt = f'Bob said "{context}"\nQ: {query[:-1]} in Bob\'s opinion?\nA: '
    elif schema == "attr":
        prompt = f"{context}\nQ:{query[:-1]} based on the given tex?\nA:"
    elif schema == "instr":
        prompt = f"Instruction: read the given information and answer the corresponding question.\n\n{context}\nQ: {query}\nA: "
    elif schema == "instr+opin":
        context = context.replace('"', "")
        prompt = f'Instruction: read the given information and answer the corresponding question.\n\nBob said "{context}"\nQ: {query[:-1]} in Bob\'s opinion?\nA:'
    else:
        prompt = f"{context}\nQ: {query}\nA: "

    if use_chat_template and tokenizer is not None:
        prompt = [{"role": "system", "content": prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return prompt


def build_risk_prompt(query, context):
    """RiskDetector prompt, byte-identical to training-data prompt_w_context."""
    return (
        f"Answer the question with as few words as possible "
        f"through the information given in the context.\n\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Answer: "
    )


# Draft prompt must be byte-identical to src/dataset.py user_format so the fine-tuned
# model sees the same template at inference as at training time.
# content = rag_input = "{context}\nQ: {question}", tokenize with add_special_tokens=False (no BOS).
_LLAMA3_USER_FMT = (
    "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

def build_gen_prompt(query, context):
    content = f"{context}\nQ: {query}"
    return _LLAMA3_USER_FMT.format(content=content)


# ===================== Inference helpers =====================

def generate_response(model, tokenizer, input_ids, max_new_tokens=32):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
        )
        generated = output_ids[0, input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def set_inhibit_strength(model, strength):
    """Dynamically update inhibit_strength on all UA-FFN modules.

    Uses class-name matching (not fixed paths) for robustness under PeftModel / DDP wrappers.
    """
    for module in model.modules():
        if module.__class__.__name__ == "LlamaMLP_w_act_inhibit":
            module.inhibit_strength = strength


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"step4_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Step4] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {log_file}")


def main(args):
    setup_logging(args.output_dir)

    logging.info("=" * 60)
    logging.info("Step 4: Two-pass inference with dynamic lambda")
    logging.info("=" * 60)
    logging.info(f"Args: {vars(args)}")

    # Load the FFN-inhibition model (reused for both passes to save GPU memory).
    # Pass 1: lambda=1 (no inhibition). Pass 2: per-sample dynamic lambda.
    logging.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_cls = LlamaForCausalLM_w_act_inhibit

    model = model_cls.from_pretrained(
        args.model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        inhibit_strength=1.0,
        inhibit_layer_list=args.inhibit_layer_list,
    )
    model.eval()

    # Load APKS LoRA adapter (optional). Without adapter -> pure base model baseline.
    use_adapter = bool(args.adapter_dir)
    if use_adapter:
        logging.info(f"Loading LoRA adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
        det_model = model.get_base_model()  # shares underlying MLP modules with model
    else:
        logging.warning("No --adapter_dir provided: running two-pass on base model only (no fine-tuning).")
        det_model = model

    logging.info(f"Loading SAE: {args.sae_name}, hookpoint: {args.hookpoint}")
    sae = Sae.load_from_disk(args.sae_name + '/' + args.hookpoint)
    sae.cfg.transcode = "transcoder" in args.sae_name
    sae = sae.to(model.device)
    sae.eval()

    _detector = RiskDetector(
        tokenizer=tokenizer,
        model=det_model,
        sae=sae,
        hookpoint=args.hookpoint,
    )

    raglens_pkl_path = os.path.join(args.detector_dir, "detector_clf.pkl")
    if os.path.exists(raglens_pkl_path):
        logging.info(f"Loading RiskDetector GAM from {raglens_pkl_path}")
        with open(raglens_pkl_path, "rb") as f:
            saved = pickle.load(f)
        detector.clf = saved["clf"]
        detector.top_k_indices = saved["top_k_indices"]
    else:
        logging.info("No saved GAM found, re-training...")
        with open(args.train_data_path, "r") as f:
            train_data = [json.loads(line) for line in f]
        train_inputs = [d["prompt_w_context"] for d in train_data]
        train_outputs = [d["pred"] for d in train_data]
        train_labels = [1 - d["is_faithful"] for d in train_data]
        split_idx = int(len(train_inputs) * 0.8)
        detector.fit(
            inputs=train_inputs[:split_idx],
            outputs=train_outputs[:split_idx],
            labels=train_labels[:split_idx],
        )
    logging.info("RiskDetector ready.")

    eval_files = sorted(glob.glob(os.path.join(args.eval_data_dir, "*.jsonl")))
    logging.info(f"Found {len(eval_files)} eval files")

    all_results_summary = {}

    for eval_file in eval_files:
        dataset_name = os.path.basename(eval_file).replace("_kc.jsonl", "").replace(".jsonl", "")
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluating: {dataset_name}")

        with jsonlines.open(eval_file, "r") as reader:
            eval_data = list(reader)
        logging.info(f"Samples loaded: {len(eval_data)}")

        gold_answers, pred_dynamic, pred_fixed, all_pm_answers = [], [], [], []
        pred_vanilla = []
        risk_scores, lambdas = [], []
        detailed_results = []

        for idx, d in tqdm(enumerate(eval_data), total=len(eval_data), desc=dataset_name):
            question = d["question"]
            context = d["context"]
            answers = d["answers"]
            parametric_answer = d["parametric_answer"]

            # Chat-template prompt byte-identical to fine-tuning (Llama3 chat, no BOS).
            prompt = build_gen_prompt(question, context)
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

            # Detection pass: adapter disabled (=base model), lambda=1, generate draft and score risk.
            det_ctx = model.disable_adapter() if use_adapter else nullcontext()
            with det_ctx:
                set_inhibit_strength(model, 1.0)
                draft = generate_response(model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens)
                raglens_prompt = build_risk_prompt(question, context)
                risk = float(detector.predict_proba(inputs=[raglens_prompt], outputs=[draft])[0])
            dynamic_lambda = max(0.0, 1.0 - args.kappa * risk)

            # Generation pass: adapter enabled (=fine-tuned model).
            # APKS: per-sample dynamic lambda.
            set_inhibit_strength(model, dynamic_lambda)
            final_pred = generate_response(model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens)
            # ParamMute baseline: fixed lambda=0.
            set_inhibit_strength(model, 0.0)
            fixed_pred = generate_response(model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens)
            # Vanilla baseline: no adapter, lambda=1 (= draft above).
            vanilla_pred = draft

            gold_answers.append(answers)
            pred_dynamic.append(final_pred)
            pred_fixed.append(fixed_pred)
            pred_vanilla.append(vanilla_pred)
            all_pm_answers.append(parametric_answer)
            risk_scores.append(risk)
            lambdas.append(dynamic_lambda)

            detailed_results.append({
                "question": question,
                "answers": answers,
                "parametric_answer": parametric_answer,
                "draft": draft,
                "risk_score": round(risk, 6),
                "dynamic_lambda": round(dynamic_lambda, 6),
                "pred_dynamic": final_pred,
                "pred_fixed_0": fixed_pred,
                "pred_vanilla": vanilla_pred,
            })

            if (idx + 1) % 100 == 0:
                mid_con_r, mid_mem_r, mid_mr = compute_metrics(
                    all_pm_answers, pred_dynamic, gold_answers
                )
                logging.info(f"  [{idx+1}/{len(eval_data)}] APKS progress: "
                             f"ConR={mid_con_r} MemR={mid_mem_r} MR={mid_mr} "
                             f"lambda_mean={np.mean(lambdas):.4f}")

        con_r_dyn, mem_r_dyn, mr_dyn = compute_metrics(all_pm_answers, pred_dynamic, gold_answers)
        con_r_fix, mem_r_fix, mr_fix = compute_metrics(all_pm_answers, pred_fixed, gold_answers)
        con_r_van, mem_r_van, mr_van = compute_metrics(all_pm_answers, pred_vanilla, gold_answers)

        logging.info(f"\n{'='*50}")
        logging.info(f"{dataset_name} final results")
        logging.info(f"  APKS dynamic lambda (kappa={args.kappa}): ConR={con_r_dyn}  MemR={mem_r_dyn}  MR={mr_dyn}")
        logging.info(f"  Fixed lambda=0 (ParamMute):              ConR={con_r_fix}  MemR={mem_r_fix}  MR={mr_fix}")
        logging.info(f"  Vanilla lambda=1 (no inhibit):            ConR={con_r_van}  MemR={mem_r_van}  MR={mr_van}")
        logging.info(f"  Risk r:    mean={np.mean(risk_scores):.4f}, std={np.std(risk_scores):.4f}")
        logging.info(f"  Lambda:    mean={np.mean(lambdas):.4f}, std={np.std(lambdas):.4f}")

        all_results_summary[dataset_name] = {
            "dynamic": {"ConR": con_r_dyn, "MemR": mem_r_dyn, "MR": mr_dyn},
            "fixed_0": {"ConR": con_r_fix, "MemR": mem_r_fix, "MR": mr_fix},
            "vanilla": {"ConR": con_r_van, "MemR": mem_r_van, "MR": mr_van},
            "risk_mean": round(float(np.mean(risk_scores)), 4),
            "risk_std": round(float(np.std(risk_scores)), 4),
            "lambda_mean": round(float(np.mean(lambdas)), 4),
            "lambda_std": round(float(np.std(lambdas)), 4),
            "num_samples": len(eval_data),
        }

        detail_path = os.path.join(args.output_dir, f"{dataset_name}_details.jsonl")
        with jsonlines.open(detail_path, mode="w") as writer:
            for r in detailed_results:
                writer.write(r)
        logging.info(f"Per-sample details saved to {detail_path}")

    logging.info(f"\n{'='*70}")
    logging.info("Summary across all datasets:")
    logging.info(f"{'Dataset':<25} {'Method':<12} {'ConR up':<8} {'MemR down':<8} {'MR down':<8}")
    logging.info("-" * 70)
    for ds, metrics in all_results_summary.items():
        logging.info(f"{ds:<25} {'APKS dynamic':<12} {metrics['dynamic']['ConR']:<8} {metrics['dynamic']['MemR']:<8} {metrics['dynamic']['MR']:<8}")
        logging.info(f"{'':<25} {'Fixed=0':<12} {metrics['fixed_0']['ConR']:<8} {metrics['fixed_0']['MemR']:<8} {metrics['fixed_0']['MR']:<8}")
        logging.info(f"{'':<25} {'Vanilla':<12} {metrics['vanilla']['ConR']:<8} {metrics['vanilla']['MemR']:<8} {metrics['vanilla']['MR']:<8}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results_summary, f, indent=2, ensure_ascii=False)
    logging.info(f"\nSummary saved to {summary_path}")
    logging.info("Step 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Two-pass inference with dynamic lambda")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--sae_name", type=str, default="EleutherAI/sae-llama-3-8b-32x")
    parser.add_argument("--hookpoint", type=str, default="layers.21.mlp")
    parser.add_argument("--detector_dir", type=str, default="outputs/detector_model")
    parser.add_argument("--train_data_path", type=str, default="data/draw_acctivations_shuf1k.jsonl")
    parser.add_argument("--eval_data_dir", type=str, default="data/CoConflictQA/test")
    parser.add_argument("--inhibit_layer_list", type=int, nargs="+", default=[21, 22, 23, 24, 25, 26])
    parser.add_argument("--adapter_dir", type=str, default="", help="APKS LoRA adapter dir; empty = base-model only (no fine-tuning baseline)")
    parser.add_argument("--kappa", type=float, default=1.0, help="Lambda mapping: lambda = max(0, 1 - kappa * r)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--schema", type=str, default="base")
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/dynamic_eval")
    args = parser.parse_args()
    main(args)
