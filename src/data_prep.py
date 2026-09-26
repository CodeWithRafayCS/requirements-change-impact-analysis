"""
Step 01: load the flat pairs.csv, validate it, and build change-request-level
splits (Section 6.2.4). Splitting by CR, never by pair, prevents leakage
between candidates of the same change.

If your downloaded dataset does NOT already look like data/pairs.csv, write
the conversion in EXAMPLE_ADAPTER below and call it once to produce
data/pairs.csv, rather than editing anything downstream.
"""
import json
import os
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["cr_id", "change_text", "requirement_id", "requirement_text", "label"]


def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cr_id": str, "requirement_id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"data file is missing required columns {missing}. "
            f"See README.md section 1 and EXAMPLE_ADAPTER below."
        )
    if "req_type" not in df.columns:
        df["req_type"] = ""
    df["label"] = df["label"].astype(int)
    df["change_text"] = df["change_text"].fillna("").astype(str).str.strip()
    df["requirement_text"] = df["requirement_text"].fillna("").astype(str).str.strip()
    if not df["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0/1 values")
    if df[["cr_id", "requirement_id"]].duplicated().any():
        raise ValueError("duplicate (cr_id, requirement_id) pairs found")

    n_cr = df["cr_id"].nunique()
    n_req = df["requirement_id"].nunique()
    expected_rows = n_cr * n_req
    if len(df) != expected_rows:
        raise ValueError(
            f"pairs.csv is not a full CR x requirement cross product: "
            f"found {len(df)} rows, expected {n_cr} x {n_req} = {expected_rows}"
        )
    expected_req_ids = set(df["requirement_id"].unique())
    incomplete = [
        cr_id
        for cr_id, group in df.groupby("cr_id")
        if set(group["requirement_id"]) != expected_req_ids
    ]
    if incomplete:
        raise ValueError(
            "Some change requests do not contain the complete requirement set: "
            + ", ".join(map(str, incomplete[:10]))
        )
    pos_per_cr = df.groupby("cr_id")["label"].sum()
    print(f"[data_prep] loaded {len(df)} pairs, {n_cr} change requests, "
          f"{n_req} unique requirements")
    print(f"[data_prep] positives per CR: mean={pos_per_cr.mean():.2f}, "
          f"min={pos_per_cr.min()}, max={pos_per_cr.max()}")
    empty_impact_crs = (pos_per_cr == 0).sum()
    if empty_impact_crs:
        print(f"[data_prep] WARNING: {empty_impact_crs} CRs have zero labelled "
              f"impacts and will be excluded from recall-based metrics")
    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    """Produces the numbers for Table 3 of the paper."""
    n_req = df["requirement_id"].nunique()
    per_cr = df.groupby("cr_id")["label"].sum()
    return {
        "n_requirements": n_req,
        "n_change_requests": df["cr_id"].nunique(),
        "avg_impacted_per_cr": round(float(per_cr.mean()), 2),
        "pct_impacted_per_cr": round(float(per_cr.mean()) / n_req * 100, 2) if n_req else 0.0,
    }


def make_cr_splits(df: pd.DataFrame, cfg: dict, seed: int) -> dict:
    """
    Split change requests (not pairs) into train/val/test.
    Falls back to 5-fold CV over CRs if the dataset is small
    (Section 6.2.4 small-data rule).
    Returns {"mode": "holdout"|"cv", ...split info...}
    """
    cr_ids = sorted(df["cr_id"].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(cr_ids)
    n = len(cr_ids)

    if n < cfg["data"]["small_dataset_cr_threshold"]:
        # 5-fold CV over CRs; nested tuning happens per-fold in the tuning step.
        folds = np.array_split(cr_ids, 5)
        folds = [list(f) for f in folds]
        return {"mode": "cv", "n_folds": 5, "folds": folds, "seed": seed}

    n_train = int(round(cfg["split"]["train_frac"] * n))
    n_val = int(round(cfg["split"]["val_frac"] * n))
    train_ids = cr_ids[:n_train]
    val_ids = cr_ids[n_train:n_train + n_val]
    test_ids = cr_ids[n_train + n_val:]
    return {"mode": "holdout", "train": train_ids, "val": val_ids, "test": test_ids, "seed": seed}


def save_split(split: dict, splits_dir: str, seed: int):
    os.makedirs(splits_dir, exist_ok=True)
    path = os.path.join(splits_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[data_prep] wrote {path}")


def subset_by_cr(df: pd.DataFrame, cr_ids) -> pd.DataFrame:
    cr_ids = set(cr_ids)
    return df[df["cr_id"].isin(cr_ids)].reset_index(drop=True)


# -------------------------------------------------------------------------
# EXAMPLE_ADAPTER: edit this to match the ACTUAL structure of whatever the
# Etezadi et al. Figshare package (or SEOSS-33 fallback) actually contains.
# The pattern below assumes a common shape: one JSON file listing, per
# change request, the rationale text and a list of impacted requirement IDs,
# plus a separate file of all requirements (id -> text). Adjust the two
# `load_*` calls and the field names to match your real files, then run
# this module directly: `python -m src.data_prep`.
# -------------------------------------------------------------------------
def EXAMPLE_ADAPTER(
    change_requests_path: str,
    requirements_path: str,
    out_path: str = "data/pairs.csv",
):
    """
    Expected (EXAMPLE ONLY — replace with the real schema):
      change_requests_path -> JSON list of
          {"cr_id": ..., "change_text": ..., "impacted_ids": [...]}
      requirements_path -> JSON list of
          {"requirement_id": ..., "requirement_text": ..., "req_type": ...}
    """
    with open(change_requests_path) as f:
        crs = json.load(f)
    with open(requirements_path) as f:
        reqs = json.load(f)

    req_lookup = {r["requirement_id"]: r for r in reqs}
    rows = []
    for cr in crs:
        impacted = set(cr.get("impacted_ids", []))
        for rid, r in req_lookup.items():
            rows.append({
                "cr_id": cr["cr_id"],
                "change_text": cr["change_text"],
                "requirement_id": rid,
                "requirement_text": r["requirement_text"],
                "req_type": r.get("req_type", ""),
                "label": int(rid in impacted),
            })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[data_prep] EXAMPLE_ADAPTER wrote {len(out_df)} rows to {out_path}")
    return out_df


if __name__ == "__main__":
    print("This module is normally imported by run_pipeline.py.")
    print("Edit EXAMPLE_ADAPTER() above to match your actual dataset files, "
          "then call it once to produce data/pairs.csv.")
