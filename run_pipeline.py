#!/usr/bin/env python3
"""
Orchestrates the full experiment, matching Appendix A of the paper.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --step 03_baselines_val
    python run_pipeline.py --all --no-cache

Writes results/table8..table12 CSVs whose columns line up with the paper's
tables, plus splits/, config_used.yaml, and a few diagnostic plots.
"""
import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from src import data_prep, embed_utils, baselines, nli_scoring, features, combine, evaluate

STEPS = [
    "01_prepare_data", "02_embed", "03_baselines_val", "04_nli_scores",
    "05_features", "06_tune", "07_test_eval", "08_stats",
    "09_sensitivity_and_errors", "10_report",
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# Shared state is passed between steps via a simple dict-based "context"
# that is pickled to disk between invocations so steps can be re-run
# individually (`--step X`) as well as with `--all`.
# -------------------------------------------------------------------------
def ctx_path(cfg):
    return os.path.join(cfg["paths"]["cache_dir"], "_ctx.npz")


def save_ctx(cfg, **arrays):
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    np.savez_compressed(ctx_path(cfg), **arrays)


def load_ctx(cfg):
    p = ctx_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No saved context at {p}; run earlier steps first.")
    return dict(np.load(p, allow_pickle=True))


# -------------------------------------------------------------------------
def step_01(cfg):
    df = data_prep.load_pairs(cfg["data"]["pairs_csv"])
    stats = data_prep.dataset_stats(df)
    print("[01] dataset stats (Table 3):", stats)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["splits_dir"], exist_ok=True)
    pd.DataFrame([stats]).to_csv(os.path.join(cfg["paths"]["results_dir"], "table3_dataset_stats.csv"), index=False)

    seed = cfg["split"]["seeds"][0]  # pre-registered seed for the held-out split
    split = data_prep.make_cr_splits(df, cfg, seed)
    data_prep.save_split(split, cfg["paths"]["splits_dir"], seed)

    df.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_df.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json"), "w") as f:
        json.dump(split, f)
    print("[01] done")


def _load_df_and_split(cfg):
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_df.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    return df, split


def step_02(cfg):
    df, split = _load_df_and_split(cfg)
    model = embed_utils.get_embedder(cfg["models"]["embedding_model"])

    req_texts = df.drop_duplicates("requirement_id")[["requirement_id", "requirement_text"]]
    req_texts = req_texts.set_index("requirement_id")["requirement_text"]
    change_texts = df.drop_duplicates("cr_id")[["cr_id", "change_text"]].set_index("cr_id")["change_text"]

    req_emb = embed_utils.embed_texts_cached(req_texts.tolist(), model, cfg["paths"]["cache_dir"], "requirements")
    chg_emb = embed_utils.embed_texts_cached(change_texts.tolist(), model, cfg["paths"]["cache_dir"], "changes")

    np.save(os.path.join(cfg["paths"]["cache_dir"], "req_emb.npy"), req_emb)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "chg_emb.npy"), chg_emb)
    req_texts.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "req_index.pkl"))
    change_texts.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "chg_index.pkl"))
    print(f"[02] embedded {len(req_texts)} requirements and {len(change_texts)} change rationales")


def _cosine_lookup(df, req_index, chg_index, req_emb, chg_emb):
    """Builds a per-pair cosine similarity column aligned to df's row order."""
    req_pos = {rid: i for i, rid in enumerate(req_index.index)}
    chg_pos = {cid: i for i, cid in enumerate(chg_index.index)}
    r_idx = df["requirement_id"].map(req_pos).to_numpy()
    c_idx = df["cr_id"].map(chg_pos).to_numpy()
    sims = np.einsum("ij,ij->i", chg_emb[c_idx], req_emb[r_idx])
    return sims


def step_03(cfg):
    df, split = _load_df_and_split(cfg)
    req_emb = np.load(os.path.join(cfg["paths"]["cache_dir"], "req_emb.npy"))
    chg_emb = np.load(os.path.join(cfg["paths"]["cache_dir"], "chg_emb.npy"))
    req_index = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "req_index.pkl"))
    chg_index = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "chg_index.pkl"))

    df["cosine_sim"] = _cosine_lookup(df, req_index, chg_index, req_emb, chg_emb)

    # TF-IDF baseline (B1), computed per project would be ideal; here per full
    # corpus for simplicity since pairs.csv is expected to be single-project.
    df["tfidf_sim"] = 0.0
    for cid, group in df.groupby("cr_id"):
        pass  # tfidf computed globally below for efficiency
    all_changes = df.drop_duplicates("cr_id")[["cr_id", "change_text"]]
    all_reqs = df.drop_duplicates("requirement_id")[["requirement_id", "requirement_text"]]
    tfidf_matrix = baselines.tfidf_baseline_scores(
        all_changes["change_text"].tolist(), all_reqs["requirement_text"].tolist(), cfg
    )
    chg_pos = {cid: i for i, cid in enumerate(all_changes["cr_id"])}
    req_pos = {rid: i for i, rid in enumerate(all_reqs["requirement_id"])}
    df["tfidf_sim"] = [
        tfidf_matrix[chg_pos[row.cr_id], req_pos[row.requirement_id]] for row in df.itertuples()
    ]

    # Calibrate tau on the VALIDATION CRs only (Section 6.4 Step 1.5).
    val_df = df[df["cr_id"].isin(split.get("val", []))] if split["mode"] == "holdout" else df
    cr_ids_val = val_df["cr_id"].unique()
    n_reqs = df["requirement_id"].nunique()
    sim_matrix = np.zeros((len(cr_ids_val), n_reqs))
    label_matrix = np.zeros((len(cr_ids_val), n_reqs))
    req_order = {rid: i for i, rid in enumerate(df["requirement_id"].unique())}
    for i, cid in enumerate(cr_ids_val):
        sub = val_df[val_df["cr_id"] == cid]
        for row in sub.itertuples():
            j = req_order[row.requirement_id]
            sim_matrix[i, j] = row.cosine_sim
            label_matrix[i, j] = row.label

    tau, diagnostics = baselines.calibrate_tau(
        sim_matrix, label_matrix, cfg["stage1"]["tau_grid"], cfg["stage1"]["target_stage1_recall"]
    )
    print(f"[03] calibrated tau = {tau}")
    pd.DataFrame(diagnostics).to_csv(os.path.join(cfg["paths"]["results_dir"], "tau_calibration_diagnostics.csv"), index=False)

    df.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json"), "w") as f:
        json.dump({"tau": tau}, f)
    print("[03] done")


def step_04(cfg, plain=False):
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]

    candidates = df[df["cosine_sim"] >= tau].reset_index(drop=True)
    print(f"[04] scoring NLI for {len(candidates)} Stage-1 candidate pairs "
          f"(plain={plain})")

    tok, model, entail_idx = nli_scoring.get_nli_model(cfg["models"]["nli_model"])
    pairs = list(zip(candidates["change_text"], candidates["requirement_text"]))
    scores = nli_scoring.nli_entailment_scores(pairs, tok, model, entail_idx, cfg, cfg["paths"]["cache_dir"], plain=plain)
    colname = "nli_score_plain" if plain else "nli_score"
    candidates[colname] = scores

    out_name = "candidates_with_nli_plain.pkl" if plain else "candidates_with_nli.pkl"
    candidates.to_pickle(os.path.join(cfg["paths"]["cache_dir"], out_name))
    print("[04] done")


def step_05(cfg):
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_with_nli.pkl"))
    feats = features.extract_features_for_candidates(
        candidates["change_text"].tolist(),
        candidates["requirement_text"].tolist(),
        [""] * len(candidates),
        candidates["req_type"].tolist(),
        cfg["models"]["spacy_model"],
    )
    for k, v in feats.items():
        candidates[k] = v
    candidates["lexical_score"] = features.combine_lexical_score(feats)
    candidates.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    print(f"[05] extracted lexical features for {len(candidates)} candidates")


def step_06(cfg):
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)

    if split["mode"] != "holdout":
        raise NotImplementedError(
            "[06] dataset is small -> nested CV tuning path. See README section 4; "
            "implement per-fold tuning here by looping over split['folds'] before "
            "proceeding to step 07 with fold-specific frozen parameters."
        )

    train = candidates[candidates["cr_id"].isin(split["train"])].reset_index(drop=True)
    val = candidates[candidates["cr_id"].isin(split["val"])].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("[06] no validation candidates found -- check that Stage 1 "
                          "tau (step 03) is not so strict it drops all validation candidates.")

    for frame in (train, val):
        frame["cosine_norm"] = combine.minmax_normalize_per_group(
            frame["cosine_sim"].to_numpy(), frame["cr_id"].to_numpy()
        )
        frame["nli_norm"] = combine.minmax_normalize_per_group(
            frame["nli_score"].to_numpy(), frame["cr_id"].to_numpy()
        )
        frame["lex_norm"] = combine.minmax_normalize_per_group(
            frame["lexical_score"].to_numpy(), frame["cr_id"].to_numpy()
        )

    labels = val["label"].to_numpy()
    groups = val["cr_id"].to_numpy()

    # C1: embedding + NLI only. Constrain gamma to zero in the search itself.
    c1 = combine.tune_weights_and_theta(
        val["cosine_norm"].to_numpy(),
        val["nli_norm"].to_numpy(),
        val["lex_norm"].to_numpy(),
        labels,
        groups,
        cfg,
        allow_lexical=False,
    )
    # C2: embedding + lexical only. Constrain beta to zero in the search itself.
    c2 = combine.tune_weights_and_theta(
        val["cosine_norm"].to_numpy(),
        val["nli_norm"].to_numpy(),
        val["lex_norm"].to_numpy(),
        labels,
        groups,
        cfg,
        allow_nli=False,
    )
    # C3: full hybrid
    c3 = combine.tune_weights_and_theta(val["cosine_norm"].to_numpy(), val["nli_norm"].to_numpy(),
                                         val["lex_norm"].to_numpy(), labels, groups, cfg)

    # C4: fit on TRAIN, select C and theta on VALIDATION.
    feature_cols = ["cosine_norm", "nli_norm", "jaccard", "type_match", "entity_score"]
    logreg, best_C, c4_theta, c4_f2 = combine.fit_logistic_combiner(
        train[feature_cols].to_numpy(),
        train["label"].to_numpy(),
        val[feature_cols].to_numpy(),
        labels,
        groups,
        cfg,
        random_state=split.get("seed", 42),
    )

    frozen = {
        "C1": c1, "C2": c2, "C3": c3,
        "C4": {"logreg_C": best_C, "theta": c4_theta, "val_f2": c4_f2},
    }
    if not np.isclose(frozen["C1"]["gamma"], 0.0):
        raise AssertionError("C1 must have gamma=0 (no lexical signal).")
    if not np.isclose(frozen["C2"]["beta"], 0.0):
        raise AssertionError("C2 must have beta=0 (no NLI signal).")
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json"), "w") as f:
        json.dump(frozen, f, indent=2, default=float)
    import joblib
    joblib.dump(logreg, os.path.join(cfg["paths"]["cache_dir"], "logreg_c4.joblib"))
    print("[06] frozen parameters:", json.dumps(frozen, indent=2, default=float))
    print("[06] done")


def step_07(cfg):
    """Single held-out test evaluation of B1-B3 and C1-C4 -> Table 8 / 9."""
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    df_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json")) as f:
        frozen = json.load(f)
    import joblib
    logreg = joblib.load(os.path.join(cfg["paths"]["cache_dir"], "logreg_c4.joblib"))

    test_ids = split["test"]
    n_total_req = df_all["requirement_id"].nunique()

    test_all = df_all[df_all["cr_id"].isin(test_ids)].reset_index(drop=True)
    test_cand = candidates[candidates["cr_id"].isin(test_ids)].reset_index(drop=True)
    test_cand["cosine_norm"] = combine.minmax_normalize_per_group(test_cand["cosine_sim"].to_numpy(), test_cand["cr_id"].to_numpy())
    test_cand["nli_norm"] = combine.minmax_normalize_per_group(test_cand["nli_score"].to_numpy(), test_cand["cr_id"].to_numpy())
    test_cand["lex_norm"] = combine.minmax_normalize_per_group(test_cand["lexical_score"].to_numpy(), test_cand["cr_id"].to_numpy())

    rows = []

    # ---- B1: TF-IDF + T2 ----
    b1_mask = np.zeros(len(test_all), dtype=bool)
    for cid, g in test_all.groupby("cr_id"):
        keep_idx = baselines.t2_cutoff_indices(g["tfidf_sim"].to_numpy())
        b1_mask[g.index[keep_idx]] = True
    rows.append({"ID": "B1 TF-IDF + T2", "stage1_recall": "n/a",
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), b1_mask, test_all["cr_id"].to_numpy(), n_total_req)})

    # ---- B2: Embedding + T2 ----
    b2_mask = np.zeros(len(test_all), dtype=bool)
    for cid, g in test_all.groupby("cr_id"):
        keep_idx = baselines.t2_cutoff_indices(g["cosine_sim"].to_numpy())
        b2_mask[g.index[keep_idx]] = True
    b2_row = evaluate.main_metrics_row(test_all["label"].to_numpy(), b2_mask, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"ID": "B2 Embedding + T2", "stage1_recall": "n/a", **b2_row})

    # ---- B3: Embedding high-recall (no downstream filter) ----
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]
    b3_mask = test_all["cosine_sim"].to_numpy() >= tau
    stage1_recall = evaluate.recall_precision_f(test_all["label"].to_numpy(), b3_mask, test_all["cr_id"].to_numpy())["recall"]
    rows.append({"ID": "B3 Embedding high-recall", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), b3_mask, test_all["cr_id"].to_numpy(), n_total_req)})

    full_groups = test_all["cr_id"].to_numpy()
    full_req_ids = test_all["requirement_id"].to_numpy()
    cand_groups = test_cand["cr_id"].to_numpy()
    cand_req_ids = test_cand["requirement_id"].to_numpy()

    def full_scores(candidate_scores):
        return evaluate.candidate_scores_to_full(
            candidate_scores,
            cand_groups,
            cand_req_ids,
            full_groups,
            full_req_ids,
        )

    def full_predictions(candidate_mask):
        return evaluate.candidate_predictions_to_full(
            candidate_mask,
            cand_groups,
            cand_req_ids,
            full_groups,
            full_req_ids,
        )

    # ---- C1/C2/C3: composite scores on Stage-1 candidates ----
    def eval_composite(name, alpha, beta, gamma, theta):
        scores = combine.composite_score(test_cand["cosine_norm"].to_numpy(), test_cand["nli_norm"].to_numpy(),
                                          test_cand["lex_norm"].to_numpy(), alpha, beta, gamma)
        pred_mask = full_predictions(scores >= theta)
        metrics = evaluate.main_metrics_row(
            test_all["label"].to_numpy(), pred_mask, full_groups, n_total_req
        )
        rows.append({"ID": name, "stage1_recall": stage1_recall, **metrics})
        return scores

    c1p, c2p, c3p = frozen["C1"], frozen["C2"], frozen["C3"]
    if not np.isclose(c1p["gamma"], 0.0):
        raise AssertionError("Invalid frozen C1: gamma must be zero.")
    if not np.isclose(c2p["beta"], 0.0):
        raise AssertionError("Invalid frozen C2: beta must be zero.")
    c1_scores = eval_composite("C1 Embedding + NLI", c1p["alpha"], c1p["beta"], c1p["gamma"], c1p["theta"])
    c2_scores = eval_composite("C2 Embedding + lexical", c2p["alpha"], c2p["beta"], c2p["gamma"], c2p["theta"])
    c3_scores = eval_composite("C3 Full hybrid (tuned)", c3p["alpha"], c3p["beta"], c3p["gamma"], c3p["theta"])

    # ---- C4: learned combiner ----
    feat_matrix = test_cand[["cosine_norm", "nli_norm", "jaccard", "type_match", "entity_score"]].to_numpy()
    c4_scores = logreg.predict_proba(feat_matrix)[:, 1]
    c4_mask = c4_scores >= frozen["C4"]["theta"]
    rows.append({"ID": "C4 Full hybrid (learned)", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(
                     test_all["label"].to_numpy(),
                     full_predictions(c4_mask),
                     full_groups,
                     n_total_req,
                 )})

    # ---- Published contextual rows ----
    pub = cfg["published_reference"]
    rows.append({"ID": "Published: embedding T2 [4]", "stage1_recall": "n/a", "recall": pub["embedding_t2_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["embedding_t2_cost_pct"]})
    rows.append({"ID": "Published: ProReFiCIA no RAG [4]", "stage1_recall": "n/a", "recall": pub["prorefiicia_no_rag_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["prorefiicia_no_rag_cost_pct"]})
    rows.append({"ID": "Published: ProReFiCIA RAG [4]", "stage1_recall": "n/a", "recall": pub["prorefiicia_rag_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["prorefiicia_rag_cost_pct"]})

    table8 = pd.DataFrame(rows)
    table8.to_csv(os.path.join(cfg["paths"]["results_dir"], "table8_main_results.csv"), index=False)
    print(table8.to_string(index=False))

    # ---- Table 9: recall at matched cost ----
    budgets = cfg["matched_cost"]["budgets_pct"]
    t9_rows = []
    b2_cost = b2_row["inspection_cost_pct"]
    for name, score_col, frame in [
        ("B1 TF-IDF", "tfidf_sim", test_all), ("B2 Embedding", "cosine_sim", test_all),
        ("B3 Embedding (ranking)", "cosine_sim", test_all),
    ]:
        row = {"Method": name}
        for b in budgets:
            row[f"cost_{b}pct"] = evaluate.recall_at_top_k(frame[score_col].to_numpy(), frame["label"].to_numpy(), frame["cr_id"].to_numpy(), b)
        row["at_B2_mean_cost"] = evaluate.recall_at_top_k(frame[score_col].to_numpy(), frame["label"].to_numpy(), frame["cr_id"].to_numpy(), b2_cost)
        t9_rows.append(row)

    for name, scores in [
        ("C1 Embedding + NLI", c1_scores),
        ("C2 Embedding + lexical", c2_scores),
        ("C3 Full hybrid (tuned)", c3_scores),
        ("C4 Full hybrid (learned)", c4_scores),
    ]:
        scores_full = full_scores(scores)
        row = {"Method": name}
        for b in budgets:
            row[f"cost_{b}pct"] = evaluate.recall_at_top_k(
                scores_full,
                test_all["label"].to_numpy(),
                full_groups,
                b,
            )
        row["at_B2_mean_cost"] = evaluate.recall_at_top_k(
            scores_full,
            test_all["label"].to_numpy(),
            full_groups,
            b2_cost,
        )
        t9_rows.append(row)

    pd.DataFrame(t9_rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table9_matched_cost.csv"), index=False)

    # ---- Table 10: signal quality on Stage-1 candidates ----
    t10_rows = []
    for name, col in [("Cosine similarity", "cosine_norm"), ("NLI entailment probability", "nli_norm"),
                       ("Lexical/heuristic score", "lex_norm")]:
        aur = evaluate.signal_auroc_auprc(test_cand[col].to_numpy(), test_cand["label"].to_numpy())
        pos = test_cand.loc[test_cand["label"] == 1, col]
        neg = test_cand.loc[test_cand["label"] == 0, col]
        t10_rows.append({"signal": name, **aur, "mean_true_impacts": pos.mean(), "mean_false_positives": neg.mean()})
    for name, scores in [("Composite C3", c3_scores), ("Composite C4", c4_scores)]:
        aur = evaluate.signal_auroc_auprc(scores, test_cand["label"].to_numpy())
        pos = scores[test_cand["label"].to_numpy() == 1]
        neg = scores[test_cand["label"].to_numpy() == 0]
        t10_rows.append({"signal": name, **aur, "mean_true_impacts": pos.mean() if len(pos) else np.nan,
                          "mean_false_positives": neg.mean() if len(neg) else np.nan})
    pd.DataFrame(t10_rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table10_signal_quality.csv"), index=False)

    # cache for step 08
    test_cand.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c3_scores.npy"), c3_scores)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c4_scores.npy"), c4_scores)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c1_scores.npy"), c1_scores)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c2_scores.npy"), c2_scores)
    test_all.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))

    # Invariant: no downstream method can recover a true impact removed by
    # Stage 1, so its full-universe recall must not exceed Stage-1 recall.
    local_rows = table8[table8["ID"].str.match(r"C[1-4]")]
    if (pd.to_numeric(local_rows["recall"]) > stage1_recall + 1e-12).any():
        raise AssertionError("A hybrid recall exceeds the Stage-1 recall ceiling.")
    print("[07] done")


def _per_cr_series(scores, labels, groups):
    """Aggregate recall-at-own-mask style values into one-per-CR arrays for paired tests."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        y = labels[m]
        if y.sum() == 0:
            continue
        out[g] = (scores[m], y)
    return out


def step_08(cfg):
    """Hypothesis tests -> Table 11 (Section 6.8)."""
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    test_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))
    c3_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores.npy"))
    c4_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c4_scores.npy"))
    c1_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c1_scores.npy"))
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json")) as f:
        frozen = json.load(f)

    def per_cr_recall_at_topk(frame, score_col_or_array, budget_pct):
        scores = frame[score_col_or_array].to_numpy() if isinstance(score_col_or_array, str) else score_col_or_array
        groups = frame["cr_id"].to_numpy()
        labels = frame["label"].to_numpy()
        cr_ids, vals = [], []
        for g in np.unique(groups):
            m = groups == g
            y = labels[m]
            if y.sum() == 0:
                continue
            s = scores[m]
            n = len(s)
            k = max(1, int(np.ceil(budget_pct / 100.0 * n)))
            top_idx = np.argsort(-s)[:k]
            pred = np.zeros(n, dtype=int)
            pred[top_idx] = 1
            tp = np.sum((y == 1) & (pred == 1))
            cr_ids.append(g); vals.append(tp / y.sum())
        return np.array(cr_ids), np.array(vals)

    b2_cost = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], "table8_main_results.csv"))
    b2_cost = float(b2_cost.loc[b2_cost["ID"] == "B2 Embedding + T2", "inspection_cost_pct"].iloc[0])

    def to_full(scores):
        return evaluate.candidate_scores_to_full(
            scores,
            test_cand["cr_id"].to_numpy(),
            test_cand["requirement_id"].to_numpy(),
            test_all["cr_id"].to_numpy(),
            test_all["requirement_id"].to_numpy(),
        )

    c1_scores_full = to_full(c1_scores)
    c3_scores_full = to_full(c3_scores)
    c4_scores_full = to_full(c4_scores)

    ids_b2, rec_b2 = per_cr_recall_at_topk(test_all, "cosine_sim", b2_cost)
    ids_c3, rec_c3 = per_cr_recall_at_topk(test_all, c3_scores_full, b2_cost)
    common = np.intersect1d(ids_b2, ids_c3)
    a = rec_c3[np.isin(ids_c3, common)]
    b = rec_b2[np.isin(ids_b2, common)]

    results = []
    if len(common) >= 2:
        stat, p_h1 = evaluate.paired_wilcoxon_one_sided(a, b, alternative="greater")
        mean_diff, ci = evaluate.bootstrap_ci_diff(a, b, common, n_resamples=cfg["stats"]["bootstrap_resamples"])
        results.append({"hypothesis": "H1", "comparison": "C3 vs B2 at B2 mean cost",
                         "mean_diff_pp": mean_diff * 100, "ci_lo_pp": ci[0] * 100, "ci_hi_pp": ci[1] * 100, "p_raw": p_h1})
    else:
        results.append({"hypothesis": "H1", "comparison": "C3 vs B2 at B2 mean cost",
                         "mean_diff_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan, "p_raw": np.nan,
                         "note": "too few CRs with impacts in common for a paired test"})

    # H3a: C3 vs C1 ; H3b: C4 vs C3, all ranked and evaluated
    # against the same complete requirement universe.
    ids_c1, rec_c1 = per_cr_recall_at_topk(test_all, c1_scores_full, b2_cost)
    common13 = np.intersect1d(ids_c3, ids_c1)
    if len(common13) >= 2:
        a13 = rec_c3[np.isin(ids_c3, common13)]
        b13 = rec_c1[np.isin(ids_c1, common13)]
        _, p_h3a = evaluate.paired_wilcoxon_one_sided(a13, b13, alternative="greater")
        mean_diff, ci = evaluate.bootstrap_ci_diff(a13, b13, common13, n_resamples=cfg["stats"]["bootstrap_resamples"])
        results.append({"hypothesis": "H3a", "comparison": "C3 vs C1 at B2 mean cost",
                         "mean_diff_pp": mean_diff * 100, "ci_lo_pp": ci[0] * 100, "ci_hi_pp": ci[1] * 100, "p_raw": p_h3a})
    else:
        results.append({"hypothesis": "H3a", "comparison": "C3 vs C1 at B2 mean cost",
                         "mean_diff_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan, "p_raw": np.nan})

    ids_c4, rec_c4 = per_cr_recall_at_topk(test_all, c4_scores_full, b2_cost)
    common34 = np.intersect1d(ids_c3, ids_c4)
    if len(common34) >= 2:
        a34 = rec_c4[np.isin(ids_c4, common34)]
        b34 = rec_c3[np.isin(ids_c3, common34)]
        _, p_h3b = evaluate.paired_wilcoxon_one_sided(a34, b34, alternative="greater")
        mean_diff, ci = evaluate.bootstrap_ci_diff(a34, b34, common34, n_resamples=cfg["stats"]["bootstrap_resamples"])
        results.append({"hypothesis": "H3b", "comparison": "C4 vs C3 at B2 mean cost",
                         "mean_diff_pp": mean_diff * 100, "ci_lo_pp": ci[0] * 100, "ci_hi_pp": ci[1] * 100, "p_raw": p_h3b})
    else:
        results.append({"hypothesis": "H3b", "comparison": "C4 vs C3 at B2 mean cost",
                         "mean_diff_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan, "p_raw": np.nan})

    # Holm correction across H1, H3a, H3b
    p_map = {r["hypothesis"]: r["p_raw"] for r in results if r["hypothesis"] in cfg["stats"]["holm_correction_family"] and not np.isnan(r["p_raw"])}
    adj = evaluate.holm_correction(p_map, alpha=cfg["stats"]["alpha"]) if p_map else {}
    for r in results:
        r["p_adjusted"] = adj.get(r["hypothesis"], np.nan)
        r["supported"] = bool(r["p_adjusted"] < cfg["stats"]["alpha"]) if not np.isnan(r.get("p_adjusted", np.nan)) else "n/a"

    # H4: paired cluster bootstrap of AUROC(NLI) - AUROC(cosine) on
    # Stage-1 candidates. Resampling whole CRs preserves within-CR dependence.
    labels = test_cand["label"].to_numpy()
    h4_diff, h4_ci, p_h4 = evaluate.paired_bootstrap_auroc_diff(
        test_cand["nli_norm"].to_numpy(),
        test_cand["cosine_norm"].to_numpy(),
        labels,
        test_cand["cr_id"].to_numpy(),
        n_resamples=cfg["stats"]["bootstrap_resamples"],
        seed=0,
        alpha=cfg["stats"]["alpha"],
    )
    results.append({"hypothesis": "H4", "comparison": "AUROC(NLI) - AUROC(cosine)",
                     "mean_diff_pp": h4_diff * 100,
                     "ci_lo_pp": h4_ci[0] * 100, "ci_hi_pp": h4_ci[1] * 100,
                     "p_raw": p_h4, "p_adjusted": p_h4,
                     "supported": bool(p_h4 < cfg["stats"]["alpha"]) if not np.isnan(p_h4) else "n/a"})

    # H1a: the selected validation threshold must meet the pre-registered
    # Stage-1 recall target.
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        selected_tau = float(json.load(f)["tau"])
    tau_diag = pd.read_csv(
        os.path.join(cfg["paths"]["results_dir"], "tau_calibration_diagnostics.csv")
    )
    chosen = tau_diag.loc[np.isclose(tau_diag["tau"], selected_tau)]
    if chosen.empty:
        raise AssertionError("Selected tau is missing from calibration diagnostics.")
    val_stage1_recall = float(chosen["mean_stage1_recall"].iloc[0])
    target = float(cfg["stage1"]["target_stage1_recall"])
    results.append({
        "hypothesis": "H1a",
        "comparison": "Validation Stage-1 recall - target",
        "mean_diff_pp": (val_stage1_recall - target) * 100,
        "ci_lo_pp": np.nan,
        "ci_hi_pp": np.nan,
        "p_raw": np.nan,
        "p_adjusted": np.nan,
        "supported": bool(val_stage1_recall >= target),
        "observed_recall_pct": val_stage1_recall * 100,
        "cost_pct": float(chosen["mean_candidate_frac"].iloc[0]) * 100,
    })

    # H2: evaluate the pre-specified learned local hybrid (C4) against the
    # published recall floor and inspection-cost ceiling. The comparison is
    # descriptive because it is not a same-dataset head-to-head experiment.
    c4_threshold = float(frozen["C4"]["theta"])
    c4_pred_full = evaluate.candidate_predictions_to_full(
        c4_scores >= c4_threshold,
        test_cand["cr_id"].to_numpy(),
        test_cand["requirement_id"].to_numpy(),
        test_all["cr_id"].to_numpy(),
        test_all["requirement_id"].to_numpy(),
    )
    c4_recall_by_cr, c4_ids = [], []
    for g in np.unique(test_all["cr_id"].to_numpy()):
        mask = test_all["cr_id"].to_numpy() == g
        y = test_all.loc[mask, "label"].to_numpy()
        pred = c4_pred_full[mask]
        if y.sum() == 0:
            continue
        c4_ids.append(g)
        c4_recall_by_cr.append(float(((y == 1) & pred).sum() / y.sum()))
    c4_recall_by_cr = np.asarray(c4_recall_by_cr)
    c4_recall = float(c4_recall_by_cr.mean())
    _, c4_recall_ci = evaluate.bootstrap_ci_diff(
        c4_recall_by_cr,
        np.zeros_like(c4_recall_by_cr),
        np.asarray(c4_ids),
        n_resamples=cfg["stats"]["bootstrap_resamples"],
        seed=0,
        alpha=cfg["stats"]["alpha"],
    )
    c4_cost = float(
        evaluate.inspection_cost(
            c4_pred_full,
            test_all["cr_id"].to_numpy(),
            test_all["requirement_id"].nunique(),
        )
    )
    recall_floor = float(cfg["hypotheses"]["h2_recall_floor_pct"])
    cost_ceiling = float(cfg["hypotheses"]["h2_cost_ceiling_pct"])
    results.append({
        "hypothesis": "H2",
        "comparison": "C4 recall/cost vs published thresholds",
        "mean_diff_pp": c4_recall * 100 - recall_floor,
        "ci_lo_pp": c4_recall_ci[0] * 100 - recall_floor,
        "ci_hi_pp": c4_recall_ci[1] * 100 - recall_floor,
        "p_raw": np.nan,
        "p_adjusted": np.nan,
        "supported": bool(
            c4_recall * 100 >= recall_floor and c4_cost <= cost_ceiling
        ),
        "observed_recall_pct": c4_recall * 100,
        "cost_pct": c4_cost,
        "recall_floor_pct": recall_floor,
        "cost_ceiling_pct": cost_ceiling,
    })

    pd.DataFrame(results).to_csv(os.path.join(cfg["paths"]["results_dir"], "table11_hypothesis_tests.csv"), index=False)
    print("[08] done, see results/table11_hypothesis_tests.csv")


def step_09(cfg):
    """Sensitivity analyses (NLI template, fixed tau) + error-analysis sample -> Table 12."""
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]
    n_total_req = df["requirement_id"].nunique()
    test_ids = split["test"]
    test_all = df[df["cr_id"].isin(test_ids)].reset_index(drop=True)

    rows = []

    # Fixed tau = 0.50 sensitivity row (Table 12) -- Stage-1 recall & cost only,
    # full C1 rerun with this tau is optional and left to the user if time allows.
    fixed_tau = cfg["stage1"]["fixed_tau_sensitivity_check"]
    mask_fixed = test_all["cosine_sim"].to_numpy() >= fixed_tau
    stage1_recall_fixed = evaluate.recall_precision_f(test_all["label"].to_numpy(), mask_fixed, test_all["cr_id"].to_numpy())["recall"]
    metrics_fixed = evaluate.main_metrics_row(test_all["label"].to_numpy(), mask_fixed, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"variant": f"Fixed tau = {fixed_tau}", "stage1_recall": stage1_recall_fixed, **metrics_fixed})

    mask_calibrated = test_all["cosine_sim"].to_numpy() >= tau
    stage1_recall_cal = evaluate.recall_precision_f(test_all["label"].to_numpy(), mask_calibrated, test_all["cr_id"].to_numpy())["recall"]
    metrics_cal = evaluate.main_metrics_row(test_all["label"].to_numpy(), mask_calibrated, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"variant": f"Calibrated tau = {tau} (default)", "stage1_recall": stage1_recall_cal, **metrics_cal})

    # NLI template sensitivity requires step_04(cfg, plain=True) to have been run.
    plain_path = os.path.join(cfg["paths"]["cache_dir"], "candidates_with_nli_plain.pkl")
    if os.path.exists(plain_path):
        print("[09] plain-template NLI scores found; add your own C1-with-plain-NLI "
              "comparison here by re-running step_06/step_07 style logic on nli_score_plain.")
    else:
        print("[09] NOTE: run `python run_pipeline.py --step 04_nli_scores --plain` "
              "first to populate the NLI-template sensitivity row.")

    pd.DataFrame(rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table12_sensitivity.csv"), index=False)

    # Error-analysis sample: false negatives and false positives from C3 on test.
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    c3_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores.npy"))
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json")) as f:
        frozen = json.load(f)
    theta = frozen["C3"]["theta"]
    pred = (c3_scores >= theta).astype(int)
    test_cand = test_cand.copy()
    test_cand["c3_score"] = c3_scores
    test_cand["c3_pred"] = pred
    fn = test_cand[(test_cand["label"] == 1) & (test_cand["c3_pred"] == 0)]
    fp = test_cand[(test_cand["label"] == 0) & (test_cand["c3_pred"] == 1)]
    sample = pd.concat([
        fn.assign(error_type="false_negative").sample(min(15, len(fn)), random_state=0) if len(fn) else fn,
        fp.assign(error_type="false_positive").sample(min(15, len(fp)), random_state=0) if len(fp) else fp,
    ])
    cols = ["cr_id", "requirement_id", "change_text", "requirement_text", "cosine_sim", "nli_score", "lexical_score", "c3_score", "error_type"]
    sample[[c for c in cols if c in sample.columns]].to_csv(
        os.path.join(cfg["paths"]["results_dir"], "error_analysis_sample.csv"), index=False
    )
    print(f"[09] wrote {len(sample)} rows to error_analysis_sample.csv for manual "
          f"categorisation (Section 7.3 error analysis)")


def step_10(cfg):
    """Generates the plots referenced in Section 7.2 and freezes the config used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    c3_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores.npy"))
    c3_scores_full = evaluate.candidate_scores_to_full(
        c3_scores,
        test_cand["cr_id"].to_numpy(),
        test_cand["requirement_id"].to_numpy(),
        test_all["cr_id"].to_numpy(),
        test_all["requirement_id"].to_numpy(),
    )

    # Recall-vs-cost curve
    budgets = np.arange(1, 21)
    b2_curve = [evaluate.recall_at_top_k(test_all["cosine_sim"].to_numpy(), test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b) for b in budgets]
    c3_curve = [
        evaluate.recall_at_top_k(
            c3_scores_full,
            test_all["label"].to_numpy(),
            test_all["cr_id"].to_numpy(),
            b,
        )
        for b in budgets
    ]
    plt.figure(figsize=(6, 4))
    plt.plot(budgets, b2_curve, label="B2 Embedding-only", marker="o", ms=3)
    plt.plot(budgets, c3_curve, label="C3 Full hybrid", marker="s", ms=3)
    plt.xlabel("Inspection cost (% of requirements)"); plt.ylabel("Recall")
    plt.title("Recall vs. inspection cost (test set)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["results_dir"], "recall_vs_cost.png"), dpi=150)
    plt.close()

    # NLI score distribution for true impacts vs false positives
    labels = test_cand["label"].to_numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(test_cand.loc[labels == 1, "nli_score"], bins=20, alpha=0.6, label="True impacts", density=True)
    plt.hist(test_cand.loc[labels == 0, "nli_score"], bins=20, alpha=0.6, label="False positives", density=True)
    plt.xlabel("NLI entailment probability"); plt.ylabel("Density")
    plt.title("NLI score distribution (Stage-1 candidates)"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["results_dir"], "nli_score_distribution.png"), dpi=150)
    plt.close()

    with open(os.path.join(cfg["paths"]["results_dir"], "..", "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    print("[10] wrote plots and config_used.yaml -- pipeline complete.")
    print("[10] Copy results/table8..table12 CSV values directly into the paper's tables.")


STEP_FUNCS = {
    "01_prepare_data": step_01, "02_embed": step_02, "03_baselines_val": step_03,
    "04_nli_scores": step_04, "05_features": step_05, "06_tune": step_06,
    "07_test_eval": step_07, "08_stats": step_08,
    "09_sensitivity_and_errors": step_09, "10_report": step_10,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", choices=STEPS)
    ap.add_argument("--plain", action="store_true", help="for 04_nli_scores: also score the plain NLI template")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_cache and os.path.exists(cfg["paths"]["cache_dir"]):
        shutil.rmtree(cfg["paths"]["cache_dir"])

    if args.all:
        for s in STEPS:
            print(f"\n=== running {s} ===")
            STEP_FUNCS[s](cfg)
    elif args.step:
        if args.step == "04_nli_scores":
            step_04(cfg, plain=args.plain)
        else:
            STEP_FUNCS[args.step](cfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
