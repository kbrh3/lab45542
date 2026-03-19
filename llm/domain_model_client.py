"""
llm/domain_model_client.py

Provides a unified inference interface that can switch between:
  1. Baseline model  (vanilla flan-t5-base)
  2. Adapted model   (flan-t5-base + LoRA adapter from training/artifacts)

Usage:
    from llm.domain_model_client import generate, set_mode

    set_mode("adapted")          # or "baseline"
    answer = generate(instruction, context)
"""

import os
import pathlib
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "training" / "artifacts" / "best_adapter"
DEFAULT_BASE = "google/flan-t5-base"

_state = {
    "mode": "baseline",
    "tokenizer": None,
    "model": None,
    "device": None,
}


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_baseline():
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_BASE).to(device)
    model.eval()
    return tokenizer, model, device


def _load_adapted():
    device = _get_device()
    adapter_path = str(ADAPTER_DIR)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. Run training/train_lora.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_BASE)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()
    return tokenizer, model, device


def set_mode(mode: str = "baseline"):
    """Switch between 'baseline' and 'adapted'."""
    if mode not in ("baseline", "adapted"):
        raise ValueError(f"mode must be 'baseline' or 'adapted', got '{mode}'")
    if _state["mode"] == mode and _state["model"] is not None:
        return
    if mode == "baseline":
        tok, mdl, dev = _load_baseline()
    else:
        tok, mdl, dev = _load_adapted()
    _state.update(mode=mode, tokenizer=tok, model=mdl, device=dev)


def get_mode() -> str:
    return _state["mode"]


def generate(
    instruction: str,
    context: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    num_beams: int = 2,
) -> str:
    """Generate a response using the currently loaded model."""
    if _state["model"] is None:
        set_mode(_state["mode"])

    parts = [instruction]
    if context:
        parts.append(context)
    prompt = "\n\n".join(parts)

    inputs = _state["tokenizer"](
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(_state["device"])

    with torch.no_grad():
        out_ids = _state["model"].generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            early_stopping=True,
        )

    return _state["tokenizer"].decode(out_ids[0], skip_special_tokens=True)


def generate_both(instruction: str, context: str = "", **kwargs) -> dict:
    """Run inference with both baseline and adapted models for comparison."""
    results = {}
    for mode in ("baseline", "adapted"):
        try:
            set_mode(mode)
            results[mode] = generate(instruction, context, **kwargs)
        except Exception as e:
            results[mode] = f"[error] {e}"
    return results
