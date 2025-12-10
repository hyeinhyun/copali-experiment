"""Lightweight evaluation helper for Vidore Benchmark v3.

The script is intentionally minimal so that it can be adapted to the
actual schema used by specific benchmark configs. It performs:

1. Dataset download from the Hugging Face Hub.
2. Text or vision-language generation for each example (optionally limited
   by `--max-samples`).
3. Exact-match accuracy tracking and JSONL logging of predictions.

Run this in an environment with network and GPU access. The container used
for authoring this repository cannot reach the Hub, so commands here are
for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vidore/colqwen2.5-v0.2 on Vidore Benchmark v3")
    parser.add_argument("--model", required=True, help="Model repo id, e.g., vidore/colqwen2.5-v0.2")
    parser.add_argument("--dataset", default="vidore/vidore-benchmark-v3", help="Dataset repo id")
    parser.add_argument("--config", default=None, help="Dataset config name (e.g., ocr, chartqa, ...)" )
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate")
    parser.add_argument("--text-column", default="question", help="Column containing the textual prompt")
    parser.add_argument("--answer-column", default="answer", help="Column containing the reference answer")
    parser.add_argument("--task-column", default=None, help="Optional column to group scores by task")
    parser.add_argument("--image-column", default=None, help="Column that holds image data or paths (enables VLM mode)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of evaluated samples")
    parser.add_argument("--output", type=Path, required=True, help="Path to the JSONL file for predictions")
    parser.add_argument("--device", default="auto", help="Torch device; defaults to auto")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length for open-ended questions")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value)
    raise TypeError(f"Unsupported image type: {type(value)}")


def setup_text_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return tokenizer, model


def setup_vlm(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return processor, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_vlm(processor, model, prompt: str, image: Image.Image, max_new_tokens: int) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(generated[0], skip_special_tokens=True)


def evaluate(args: argparse.Namespace) -> None:
    ds = load_dataset(args.dataset, name=args.config, split=args.split, trust_remote_code=True)
    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    is_vlm = args.image_column is not None
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    if is_vlm:
        processor, model = setup_vlm(args.model, device)
        tokenizer = None
    else:
        tokenizer, model = setup_text_model(args.model, device)
        processor = None

    total = 0
    correct = 0
    results: List[Dict[str, Any]] = []

    for ex in tqdm(ds, desc="evaluating"):
        prompt = ex[args.text_column]
        reference = ex[args.answer_column]

        if is_vlm:
            image = load_image(ex[args.image_column])
            prediction = generate_vlm(processor, model, prompt, image, args.max_new_tokens)
        else:
            prediction = generate_text(tokenizer, model, prompt, args.max_new_tokens)

        clean_pred = normalize_text(prediction)
        clean_ref = normalize_text(reference)
        is_correct = clean_pred == clean_ref

        total += 1
        correct += int(is_correct)

        record = {
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "correct": is_correct,
        }
        if args.task_column and args.task_column in ex:
            record["task"] = ex[args.task_column]
        results.append(record)

    accuracy = correct / max(total, 1)

    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Evaluated {total} examples | Accuracy: {accuracy:.3f}")
    print(f"Saved predictions to {args.output}")


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
