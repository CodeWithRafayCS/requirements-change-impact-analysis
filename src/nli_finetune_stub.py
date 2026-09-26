"""
OPTIONAL, NOT WIRED INTO run_pipeline.py.

Stub for fine-tuning the NLI cross-encoder on the training split, per the
paper's Section 6.6.4 sensitivity check ("Optional, time permitting"). Only
attempt this if steps 01-10 already run successfully end to end on real data
and you have time left before the deadline.

This trains the model to directly predict "impacted" (1) vs "not impacted" (0)
from the (change_text, requirement_text) pair, using the same reformulated
template as the zero-shot scorer, so its output stays comparable.

Usage (after you have real data and step 01 has produced cache/pairs_df.pkl):

    python -m src.nli_finetune_stub --config config.yaml --epochs 3

This is deliberately minimal (no learning-rate schedule tuning, no early
stopping beyond a fixed epoch count) -- treat it as a starting point, not a
finished experiment. If you use it, add a row to Table 12 and describe the
fine-tuning setup (epochs, batch size, optimizer) in the paper for
reproducibility, per Section 6.9.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import yaml
    import torch
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    if split["mode"] != "holdout":
        raise NotImplementedError(
            "Fine-tuning on the CV path is not implemented in this stub; "
            "either implement per-fold fine-tuning or skip this optional step."
        )

    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_df.pkl"))
    train_df = df[df["cr_id"].isin(split["train"])].reset_index(drop=True)

    templates = cfg["nli"]
    prem_t, hyp_t = templates["premise_template"], templates["hypothesis_template"]

    class PairDataset(Dataset):
        def __init__(self, frame, tokenizer, max_length):
            self.premises = [prem_t.format(change_text=t) for t in frame["change_text"]]
            self.hyps = [hyp_t.format(requirement_text=t) for t in frame["requirement_text"]]
            self.labels = frame["label"].tolist()
            self.tok = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            enc = self.tok(self.premises[idx], self.hyps[idx], truncation=True,
                            max_length=self.max_length, padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    model_name = cfg["models"]["nli_model"]
    tok = AutoTokenizer.from_pretrained(model_name)
    # Re-head the model for binary impacted/not-impacted classification rather
    # than 3-way NLI, since that is the actual target of this fine-tuning run.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                                 ignore_mismatched_sizes=True)

    ds = PairDataset(train_df, tok, cfg["nli"]["max_length"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optim = AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optim.step()
            total_loss += out.loss.item()
        print(f"[nli_finetune_stub] epoch {epoch + 1}/{args.epochs} "
              f"mean loss = {total_loss / len(dl):.4f}")

    out_dir = os.path.join(cfg["paths"]["cache_dir"], "nli_finetuned")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[nli_finetune_stub] saved fine-tuned model to {out_dir}")
    print("[nli_finetune_stub] to use it, point config.yaml's models.nli_model "
          "at this path and re-run steps 04 onward, or add a separate config "
          "for the Table 12 sensitivity row.")


if __name__ == "__main__":
    main()
