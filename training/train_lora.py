"""
training/train_lora.py

Fine-tunes a base language model using LoRA (Low-Rank Adaptation) on the
PolicyPulse legislative instruction dataset.

Base model : google/flan-t5-base  (250M params, seq2seq)
Method     : LoRA via PEFT — only ~0.5% of parameters are trainable
Dataset    : data/instruction_dataset/{train,val}.jsonl

Usage:
    python training/train_lora.py                       # defaults
    python training/train_lora.py --epochs 5 --lr 3e-4  # custom
"""

import argparse
import json
import os
import pathlib
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "instruction_dataset"
ARTIFACT_DIR = ROOT / "training" / "artifacts"

DEFAULT_BASE_MODEL = "google/flan-t5-base"
DEFAULT_EPOCHS = 4
DEFAULT_LR = 3e-4
DEFAULT_BATCH = 4
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ── Dataset ────────────────────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def _build_prompt(self, ex):
        parts = [ex["instruction"]]
        if ex.get("input"):
            parts.append(ex["input"])
        return "\n\n".join(parts)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src = self._build_prompt(ex)
        tgt = ex["output"]

        src_enc = self.tokenizer(
            src, max_length=self.max_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                tgt, max_length=self.max_tgt, truncation=True, padding="max_length", return_tensors="pt"
            )

        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── Training loop ──────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_ds = InstructionDataset(DATA_DIR / "train.jsonl", tokenizer)
    val_ds   = InstructionDataset(DATA_DIR / "val.jsonl", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
        log_rows.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_dir = ARTIFACT_DIR / "best_adapter"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  ↳ Saved best adapter (val_loss={avg_val:.4f})")

    # Save training log
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"Training log → {log_path}")

    # Save final adapter
    final_dir = ARTIFACT_DIR / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final adapter → {final_dir}")

    # Save config for reproducibility
    cfg = {
        "base_model": args.model,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ["q", "v"],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch,
        "max_src_len": MAX_SRC_LEN,
        "max_tgt_len": MAX_TGT_LEN,
        "best_val_loss": best_val_loss,
        "device": str(device),
    }
    with open(ARTIFACT_DIR / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for PolicyPulse domain")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
