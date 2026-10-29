"""Stage 1: bug detection with the fine-tuned classifiers."""
import os
import threading
import time

import torch

from .modeling import load_classifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DISPLAY = {"codebert": "CodeBERT", "codet5": "CodeT5-small"}

torch.set_num_threads(max(1, (os.cpu_count() or 2)))


class Detector:
    def __init__(self):
        self.models = {}
        self.errors = {}
        self.lock = threading.Lock()
        self.load_all()

    def load_all(self):
        for name in ("codebert", "codet5"):
            path = os.path.join(MODELS_DIR, name)
            if not os.path.exists(os.path.join(path, "meta.json")):
                continue
            try:
                t0 = time.time()
                model, tok, meta = load_classifier(path)
                self.models[name] = (model, tok, meta)
                print(f"[detector] loaded {name} in {time.time() - t0:.1f}s")
            except Exception as e:  # keep the app running even if one model is broken
                self.errors[name] = str(e)
                print(f"[detector] could not load {name}: {e}")

    def status(self):
        return {
            name: {
                "loaded": name in self.models,
                "display": DISPLAY[name],
                "error": self.errors.get(name),
                "meta": self.models[name][2] if name in self.models else None,
            }
            for name in ("codebert", "codet5")
        }

    @torch.no_grad()
    def detect(self, name, code):
        model, tok, meta = self.models[name]
        max_len = meta.get("max_len", 400)
        threshold = meta.get("threshold", 0.5)

        # Same whitespace handling as training, but keep a map back to the
        # original characters so the attention heatmap lands on the real code.
        words, starts = [], []
        i, n = 0, len(code)
        while i < n:
            while i < n and code[i].isspace():
                i += 1
            if i >= n:
                break
            j = i
            while j < n and not code[j].isspace():
                j += 1
            words.append(code[i:j])
            starts.append(i)
            i = j
        flat = " ".join(words)
        char_map, pos = [], 0  # flat index -> original index
        for w, s in zip(words, starts):
            char_map.extend(range(s, s + len(w)))
            char_map.append(s + len(w))  # the joining space
            pos += len(w) + 1

        enc = tok(flat, max_length=max_len, truncation=True, return_tensors="pt",
                  return_offsets_mapping=tok.is_fast)
        if tok.is_fast:
            offsets = enc.pop("offset_mapping")[0].tolist()
        else:  # slow tokenizers cannot map tokens back to characters
            offsets = [(0, 0)] * enc["input_ids"].shape[1]
        n_tokens = int(enc["attention_mask"].sum())
        full_len = len(tok(flat)["input_ids"])

        t0 = time.time()
        with self.lock:
            logit, attentions = model(enc["input_ids"], enc["attention_mask"], output_attentions=True)
        latency = (time.time() - t0) * 1000
        prob = float(torch.sigmoid(logit)[0])

        # Attention: last layer, averaged over heads.
        last = attentions[-1][0].mean(0)  # (seq, seq)
        if meta.get("pooling") == "cls":
            scores = last[0]  # what the [CLS] summary token looks at
        else:
            scores = last.mean(0)  # attention each token receives, averaged over all tokens
        special = set(tok.all_special_ids)
        ids = enc["input_ids"][0].tolist()
        spans = []
        for tid, (a, b), sc in zip(ids, offsets, scores.tolist()):
            if tid in special or a == b:
                continue
            oa = char_map[a] if a < len(char_map) else n
            ob = char_map[b - 1] + 1 if b - 1 < len(char_map) else n
            spans.append([oa, ob, sc])
        if spans:
            vals = sorted(s[2] for s in spans)
            cap = vals[int(0.97 * (len(vals) - 1))] or max(vals) or 1.0
            for s in spans:
                s[2] = round(min(s[2] / cap, 1.0), 4)

        top = sorted(spans, key=lambda s: -s[2])[:8]
        return {
            "model": name,
            "display": DISPLAY[name],
            "probability": prob,
            "label": "buggy" if prob >= threshold else "clean",
            "threshold": threshold,
            "latency_ms": round(latency, 1),
            "tokens": n_tokens,
            "truncated": full_len > max_len,
            "attention": spans,
            "top_tokens": [{"text": code[a:b], "score": s} for a, b, s in top],
        }
