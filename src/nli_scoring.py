"""
Step 04: NLI entailment scoring for Stage-2 candidates (Section 6.4, Step 2.1).

Uses the reformulated premise/hypothesis templates from config.yaml by default.
`plain=True` switches to the plain-pair template for the Table 12 sensitivity
check.

Label order is read from the model config (id2label) rather than assumed,
because different NLI checkpoints order [contradiction, neutral, entailment]
differently.
"""
import hashlib
import os
import numpy as np


def _hash_pairs(pairs) -> str:
    h = hashlib.sha256()
    for p, hpo in pairs:
        h.update(p.encode("utf-8")); h.update(b"\x00")
        h.update(hpo.encode("utf-8")); h.update(b"\x01")
    return h.hexdigest()[:16]


def get_nli_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entail_idx = next(i for i, lab in id2label.items() if "entail" in lab)
    return tok, model, entail_idx


def nli_entailment_scores(
    change_requirement_pairs,  # list[(change_text, requirement_text)]
    tok, model, entail_idx: int,
    cfg, cache_dir: str, plain: bool = False,
) -> np.ndarray:
    """
    Returns a 1D array of entailment probabilities, one per pair, in the
    input order. Cached to disk since NLI is the slowest step.
    """
    import torch  # lazy import so modules that don't need NLI can load without torch installed
    templates = cfg["nli"]
    if plain:
        prem_t = templates["plain_premise_template"]
        hyp_t = templates["plain_hypothesis_template"]
        tag = "plain"
    else:
        prem_t = templates["premise_template"]
        hyp_t = templates["hypothesis_template"]
        tag = "reformulated"

    formatted = [
        (prem_t.format(change_text=c), hyp_t.format(requirement_text=r))
        for c, r in change_requirement_pairs
    ]
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_pairs(formatted)
    cache_path = os.path.join(cache_dir, f"nli_{tag}_{key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    batch_size = templates["batch_size"]
    max_length = templates["max_length"]
    scores = np.zeros(len(formatted), dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        for start in range(0, len(formatted), batch_size):
            batch = formatted[start:start + batch_size]
            premises = [p for p, _ in batch]
            hyps = [h for _, h in batch]
            enc = tok(premises, hyps, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores[start:start + len(batch)] = probs[:, entail_idx]

    np.save(cache_path, scores)
    return scores
