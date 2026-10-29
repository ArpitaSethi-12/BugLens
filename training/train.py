#!/usr/bin/env python3
"""Fine-tune CodeBERT or CodeT5-small for defect detection (CodeXGLUE / Devign).

Example (Colab T4):
    python train.py --model codebert --epochs 5
    python train.py --model codet5   --epochs 5

Outputs (inside --out_dir, default ".."):
    models/<model>/        fine-tuned classifier, loads fully offline in the app
    results/<model>.json   metrics, curves, confusion matrix, training history
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
try:
    from app.modeling import BASE_MODELS, create_classifier, save_classifier
except ImportError:  # Colab: modeling.py sits next to this file
    from modeling import BASE_MODELS, create_classifier, save_classifier

HF_DATASET = "google/code_x_glue_cc_defect_detection"
PARQUET_URL = ("https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection/"
               "resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet")


# ----------------------------------------------------------------------------- data
def load_codexglue():
    """Return dict split -> list of (code, label). Tries three sources."""
    def to_rows(records):
        return [(r["func"], int(bool(r["target"]))) for r in records]

    # 1) HuggingFace datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET)
        print("Loaded dataset with `datasets`")
        return {s: to_rows(ds[s]) for s in ("train", "validation", "test")}
    except Exception as e:
        print("datasets.load_dataset failed:", repr(e)[:200])

    # 2) Parquet files that HuggingFace auto-converts
    try:
        import pandas as pd
        out = {}
        for s in ("train", "validation", "test"):
            df = pd.read_parquet(PARQUET_URL.format(split=s))
            out[s] = to_rows(df.to_dict("records"))
        print("Loaded dataset from parquet")
        return out
    except Exception as e:
        print("parquet download failed:", repr(e)[:200])

    # 3) Original CodeXGLUE release (function.json + split index files)
    import subprocess
    import urllib.request
    d = os.path.join(HERE, "codexglue_raw")
    os.makedirs(d, exist_ok=True)
    fj = os.path.join(d, "function.json")
    if not os.path.exists(fj):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF", "-O", fj], check=True)
    funcs = json.load(open(fj))
    base = ("https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
            "Code-Code/Defect-detection/dataset/{}.txt")
    out = {}
    for s, fname in (("train", "train"), ("validation", "valid"), ("test", "test")):
        idx = urllib.request.urlopen(base.format(fname)).read().decode().split()
        out[s] = to_rows([funcs[int(i)] for i in idx])
    print("Loaded dataset from the original CodeXGLUE release")
    return out


def load_local_jsonl(data_dir):
    out = {}
    for s in ("train", "validation", "test"):
        with open(os.path.join(data_dir, s + ".jsonl")) as f:
            out[s] = [(r["func"], int(r["target"])) for r in map(json.loads, f)]
    return out


class CodeDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        code, label = self.rows[i]
        code = " ".join(code.split())  # collapse whitespace like the CodeXGLUE baseline
        enc = self.tok(code, max_length=self.max_len, truncation=True, padding="max_length")
        return (torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                torch.tensor(label, dtype=torch.float))


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs, labels, losses = [], [], []
    bce = nn.BCEWithLogitsLoss()
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(ids, mask)
        losses.append(bce(logits.float(), y).item())
        probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
        labels.extend(y.cpu().tolist())
    return np.array(probs), np.array(labels).astype(int), float(np.mean(losses))


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds) if len(set(preds)) > 1 else 0.0,
    }
    m["auc"] = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return {k: float(v) for k, v in m.items()}


def downsample(xs, ys, n=120):
    if len(xs) <= n:
        return [[float(a), float(b)] for a, b in zip(xs, ys)]
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [[float(xs[i]), float(ys[i])] for i in idx]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BASE_MODELS), default="codebert")
    ap.add_argument("--model_name_or_path", help="override the HuggingFace id (used for offline tests)")
    ap.add_argument("--data_dir", help="folder with train/validation/test .jsonl (default: download CodeXGLUE)")
    ap.add_argument("--probe_file", default=os.path.join(HERE, "..", "data", "probe_set.json"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, ".."))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--max_train_samples", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_id, arch, pooling = BASE_MODELS[args.model]
    name_or_path = args.model_name_or_path or hf_id
    lr = args.lr or (2e-5 if arch == "roberta" else 5e-5)
    # T5 is unstable in fp16, so it trains in fp32 (it is small enough to stay fast)
    use_amp = device.type == "cuda" and (args.fp16 == "on" or (args.fp16 == "auto" and arch != "t5"))

    print(f"Model {args.model} ({name_or_path}) | device {device} | fp16 {use_amp} | lr {lr}")
    data = load_local_jsonl(args.data_dir) if args.data_dir else load_codexglue()
    if args.max_train_samples:
        data["train"] = data["train"][: args.max_train_samples]
    for s, rows in data.items():
        pos = sum(l for _, l in rows)
        print(f"  {s:<10} {len(rows):>6} functions | {pos / max(len(rows), 1):.1%} buggy")

    model, tokenizer = create_classifier(name_or_path, arch, pooling)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loaders = {
        s: DataLoader(CodeDataset(rows, tokenizer, args.max_len), batch_size=args.batch_size,
                      shuffle=(s == "train"), num_workers=2, pin_memory=device.type == "cuda")
        for s, rows in data.items()
    }

    # class weighting in case the split is imbalanced
    n_pos = sum(l for _, l in data["train"])
    pos_weight = torch.tensor((len(data["train"]) - n_pos) / max(n_pos, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups, lr=lr)
    total_steps = len(loaders["train"]) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_acc, best_state, bad_epochs = [], -1.0, None, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, steps = time.time(), 0.0, 0
        for step, (ids, mask, y) in enumerate(loaders["train"], 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(ids, mask)
            loss = loss_fn(logits.float(), y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            steps += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} loss {run_loss / steps:.4f}")

        vp, vl, vloss = predict(model, loaders["validation"], device, use_amp)
        vm = compute_metrics(vp, vl)
        history.append({"epoch": epoch, "train_loss": run_loss / max(steps, 1), "val_loss": vloss,
                        "val_accuracy": vm["accuracy"], "val_f1": vm["f1"],
                        "seconds": time.time() - t0})
        print(f"Epoch {epoch}: train_loss {run_loss / max(steps, 1):.4f} | val_loss {vloss:.4f} | "
              f"val_acc {vm['accuracy']:.4f} | val_f1 {vm['f1']:.4f} | {time.time() - t0:.0f}s")

        if vm["accuracy"] > best_acc:
            best_acc, bad_epochs = vm["accuracy"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  new best model")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break

    train_seconds = time.time() - t_start
    model.load_state_dict(best_state)

    # ---- test set
    tp, tl, _ = predict(model, loaders["test"], device, use_amp)
    tm = compute_metrics(tp, tl)
    fpr, tpr, _ = roc_curve(tl, tp)
    prec, rec, _ = precision_recall_curve(tl, tp)
    cm = confusion_matrix(tl, (tp >= 0.5).astype(int), labels=[0, 1]).tolist()
    print("TEST:", json.dumps({k: round(v, 4) for k, v in tm.items()}))

    # ---- cross-language probe (hand-written Python + C pairs)
    probe = None
    if args.probe_file and os.path.exists(args.probe_file):
        rows = json.load(open(args.probe_file))
        pl = DataLoader(CodeDataset([(r["func"], r["target"]) for r in rows], tokenizer, args.max_len),
                        batch_size=args.batch_size)
        pp, plab, _ = predict(model, pl, device, use_amp)
        probe = {"overall": compute_metrics(pp, plab), "by_language": {}, "items": []}
        for lang in sorted({r["language"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["language"] == lang]
            probe["by_language"][lang] = {**compute_metrics(pp[idx], plab[idx]), "n": len(idx)}
        for r, p in zip(rows, pp):
            probe["items"].append({"id": r["id"], "language": r["language"], "target": r["target"],
                                   "prob": float(p)})
        print("PROBE:", {k: round(v["accuracy"], 3) for k, v in probe["by_language"].items()})

    # ---- CPU inference latency (what the demo laptop will see)
    model_cpu = model.to("cpu").float().eval()
    sample = data["test"][0][0]
    enc = tokenizer(" ".join(sample.split()), max_length=args.max_len, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        model_cpu(enc["input_ids"], enc["attention_mask"])
        t0 = time.time()
        for _ in range(5):
            model_cpu(enc["input_ids"], enc["attention_mask"])
    cpu_ms = (time.time() - t0) / 5 * 1000

    result = {
        "model": args.model,
        "base_model": name_or_path,
        "arch": arch,
        "pooling": pooling,
        "parameters": n_params,
        "config": {"epochs_run": len(history), "max_epochs": args.epochs, "batch_size": args.batch_size,
                   "lr": lr, "max_len": args.max_len, "fp16": use_amp, "weight_decay": args.weight_decay,
                   "train_size": len(data["train"]), "val_size": len(data["validation"]),
                   "test_size": len(data["test"]),
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "train_seconds": train_seconds,
        "cpu_latency_ms": cpu_ms,
        "best_val_accuracy": best_acc,
        "history": history,
        "test": tm,
        "confusion_matrix": cm,
        "roc": downsample(fpr, tpr),
        "pr": downsample(rec, prec),
        "probe": probe,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    res_path = os.path.join(args.out_dir, "results", f"{args.model}.json")
    def clean(o):  # browsers cannot parse NaN in JSON
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    with open(res_path, "w") as f:
        json.dump(clean(result), f, indent=1)

    meta = {"name": args.model, "base_model": name_or_path, "arch": arch, "pooling": pooling,
            "max_len": args.max_len, "threshold": 0.5, "test_accuracy": tm["accuracy"],
            "test_f1": tm["f1"]}
    save_classifier(model_cpu, tokenizer, os.path.join(args.out_dir, "models", args.model), meta)
    print(f"Saved model to {os.path.join(args.out_dir, 'models', args.model)} and results to {res_path}")


if __name__ == "__main__":
    main()
