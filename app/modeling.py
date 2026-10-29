"""Shared model code used by both training and the web app.

DefectClassifier = pre-trained code encoder (CodeBERT or CodeT5 encoder)
                   + dropout + one linear layer that outputs a single logit.
sigmoid(logit) = probability that the function contains a defect.
"""
import json
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

# name -> (HuggingFace id, architecture family, pooling strategy)
BASE_MODELS = {
    "codebert": ("microsoft/codebert-base", "roberta", "cls"),
    "codet5": ("Salesforce/codet5-small", "t5", "mean"),
}


class DefectClassifier(nn.Module):
    def __init__(self, encoder, pooling="cls", dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        h = out.last_hidden_state
        if self.pooling == "cls":
            # CodeBERT: the first token (<s>) acts as the [CLS] summary token
            pooled = h[:, 0]
        else:
            # CodeT5 has no [CLS] token, so average all real (non-padding) tokens
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        return logits, (out.attentions if output_attentions else None)


def build_encoder(name_or_path, arch, eager_attention=False):
    kwargs = {}
    if eager_attention:
        kwargs["attn_implementation"] = "eager"  # needed to read attention weights
    if arch == "t5":
        enc = T5EncoderModel.from_pretrained(name_or_path, **kwargs)
    else:
        enc = AutoModel.from_pretrained(name_or_path, add_pooling_layer=False, **kwargs)
    return enc.float()  # saved weights may be fp16; always compute in fp32


def load_tokenizer(name_or_path):
    """AutoTokenizer first. CodeT5's hub files use an old format that newer
    transformers versions reject, so fall back to rebuilding the same byte-level
    BPE tokenizer straight from its vocab.json and merges.txt."""
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception as e:
        print(f"AutoTokenizer failed ({type(e).__name__}: {str(e)[:80]}), rebuilding from vocab files")
    import tempfile
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing
    from transformers import RobertaTokenizerFast
    if os.path.isdir(name_or_path):
        vocab = os.path.join(name_or_path, "vocab.json")
        merges = os.path.join(name_or_path, "merges.txt")
    else:
        from huggingface_hub import hf_hub_download
        vocab = hf_hub_download(name_or_path, "vocab.json")
        merges = hf_hub_download(name_or_path, "merges.txt")
    bpe = ByteLevelBPETokenizer(vocab, merges)
    bpe._tokenizer.post_processor = RobertaProcessing(
        ("</s>", bpe.token_to_id("</s>")), ("<s>", bpe.token_to_id("<s>")))
    path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    bpe.save(path)
    return RobertaTokenizerFast(
        tokenizer_file=path, model_max_length=512,
        bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")


def create_classifier(name_or_path, arch, pooling):
    encoder = build_encoder(name_or_path, arch)
    tokenizer = load_tokenizer(name_or_path)
    return DefectClassifier(encoder, pooling=pooling), tokenizer


def save_classifier(model, tokenizer, out_dir, meta, half=True):
    """Save in a self-contained folder so the app can load it fully offline."""
    os.makedirs(out_dir, exist_ok=True)
    model = model.to("cpu")
    if half:
        model.encoder.half()  # halves file size; app upcasts to float32 on load
    model.encoder.save_pretrained(os.path.join(out_dir, "encoder"))
    torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_classifier(model_dir, device="cpu"):
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    encoder = build_encoder(os.path.join(model_dir, "encoder"), meta["arch"], eager_attention=True)
    model = DefectClassifier(encoder, pooling=meta["pooling"])
    head_state = torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu")
    model.head.load_state_dict({k: v.float() for k, v in head_state.items()})
    model.to(device).eval()
    tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer"))
    return model, tokenizer, meta
