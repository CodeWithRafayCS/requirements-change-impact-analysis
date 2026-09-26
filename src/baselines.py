"""
Step 03: B1 (TF-IDF + T2), B2 (SBERT + T2), B3 (SBERT high-recall Stage 1).
Also houses the T2 dynamic-cutoff routine used by both B1 and B2, and the
tau-calibration routine used by Stage 1 for the hybrid pipeline (B3, C1-C4).

T2 definition (Section 6.5, item 4): sort scores descending, take consecutive
drops d(i) = s(i) - s(i+1), keep the top i* where i* maximises d(i).
NOTE: verify this against the exact wording in Etezadi et al. (2026) before
trusting it for the final report (flagged in the paper, Appendix B checklist).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def t2_cutoff_indices(scores: np.ndarray) -> np.ndarray:
    """
    scores: 1D array of similarity scores for all requirements w.r.t. one CR.
    Returns the indices (into `scores`) selected by the T2 dynamic cutoff.
    """
    order = np.argsort(-scores)  # descending
    sorted_scores = scores[order]
    if len(sorted_scores) <= 1:
        return order
    drops = sorted_scores[:-1] - sorted_scores[1:]
    i_star = int(np.argmax(drops))  # 0-indexed: keep the first i_star+1 items
    keep = order[: i_star + 1]
    return keep


def tfidf_baseline_scores(change_texts, requirement_texts, cfg):
    """
    Fits one TF-IDF vectorizer over (all change texts + all requirement texts)
    per project so the vocabulary is shared, then returns a (n_changes, n_reqs)
    cosine similarity matrix (Baseline B1, pre-T2).
    """
    vec = TfidfVectorizer(
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        sublinear_tf=cfg["tfidf"]["sublinear_tf"],
        stop_words=cfg["tfidf"]["stop_words"],
    )
    corpus = list(change_texts) + list(requirement_texts)
    vec.fit(corpus)
    change_vecs = vec.transform(change_texts)
    req_vecs = vec.transform(requirement_texts)
    return cosine_similarity(change_vecs, req_vecs)


def calibrate_tau(sim_matrix: np.ndarray, labels_matrix: np.ndarray, tau_grid, target_recall: float):
    """
    sim_matrix, labels_matrix: (n_changes, n_reqs), aligned.
    Chooses the LARGEST tau (smallest candidate set) whose mean Stage-1 recall
    across change requests is >= target_recall (Section 6.4, Step 1.5).
    Returns (best_tau, per_tau_diagnostics: list of dict).
    """
    diagnostics = []
    best_tau = None
    for tau in sorted(tau_grid):
        recalls = []
        cand_fracs = []
        for i in range(sim_matrix.shape[0]):
            labels = labels_matrix[i]
            if labels.sum() == 0:
                continue
            selected = sim_matrix[i] >= tau
            tp = np.sum(selected & (labels == 1))
            recalls.append(tp / labels.sum())
            cand_fracs.append(selected.mean())
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_cost = float(np.mean(cand_fracs)) if cand_fracs else 0.0
        diagnostics.append({"tau": tau, "mean_stage1_recall": mean_recall, "mean_candidate_frac": mean_cost})
        if mean_recall >= target_recall:
            best_tau = tau  # keep updating; loop is ascending so last hit = largest tau
    if best_tau is None:
        best_tau = min(tau_grid)
        print(f"[baselines] WARNING: no tau in grid reached target recall "
              f"{target_recall:.2f}; falling back to smallest tau={best_tau} "
              f"(Stage 1 flagged as non-selective, see paper Step 1.5)")
    return best_tau, diagnostics


def stage1_candidates(sim_matrix: np.ndarray, tau: float) -> np.ndarray:
    """Boolean (n_changes, n_reqs) mask of candidates passing the threshold."""
    return sim_matrix >= tau


def apply_t2_per_row(sim_matrix: np.ndarray) -> np.ndarray:
    """Boolean (n_changes, n_reqs) mask of candidates selected by T2 per CR."""
    mask = np.zeros_like(sim_matrix, dtype=bool)
    for i in range(sim_matrix.shape[0]):
        keep = t2_cutoff_indices(sim_matrix[i])
        mask[i, keep] = True
    return mask
