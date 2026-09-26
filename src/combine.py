"""
Step 06: score combination and tuning (Section 6.4 Steps 2.3-2.4, Section 6.5
items 7-8). All tuning happens on the validation split only; parameters are
frozen and reused unchanged on the test split (Section 6.6.3, step 2).
"""
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score


def minmax_normalize_per_group(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """
    Min-max normalise `values` within each group (change request) separately,
    per Section 6.4 Step 2.3. A constant-valued group is set to 0.
    """
    out = np.zeros_like(values, dtype=np.float32)
    for g in np.unique(group_ids):
        mask = group_ids == g
        v = values[mask]
        lo, hi = v.min(), v.max()
        out[mask] = 0.0 if hi == lo else (v - lo) / (hi - lo)
    return out


def weight_simplex(step: float):
    """All (alpha, beta, gamma) on a simplex with the given grid step."""
    n = int(round(1.0 / step))
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append((i * step, j * step, k * step))
    return combos


def composite_score(cos_n, nli_n, lex_n, alpha, beta, gamma):
    return alpha * cos_n + beta * nli_n + gamma * lex_n


def best_theta_for_scores(scores: np.ndarray, labels: np.ndarray, group_ids: np.ndarray,
                           theta_grid, beta_for_fscore: float = 2.0):
    """
    Sweeps theta, computes discrete precision/recall/F-beta averaged per CR,
    returns the theta maximising mean F-beta (Section 6.5 item 8 / F2 default).
    """
    best_theta, best_score = theta_grid[0], -1.0
    for theta in theta_grid:
        f_per_cr = []
        for g in np.unique(group_ids):
            mask = group_ids == g
            y = labels[mask]
            pred = (scores[mask] >= theta).astype(int)
            if y.sum() == 0 and pred.sum() == 0:
                continue
            f_per_cr.append(fbeta_score(y, pred, beta=beta_for_fscore, zero_division=0))
        mean_f = float(np.mean(f_per_cr)) if f_per_cr else 0.0
        if mean_f > best_score:
            best_score, best_theta = mean_f, theta
    return best_theta, best_score


def tune_weights_and_theta(
    cos_n,
    nli_n,
    lex_n,
    labels,
    group_ids,
    cfg,
    *,
    allow_nli=True,
    allow_lexical=True,
):
    """
    Grid search over (alpha, beta, gamma) simplex; for each, tune theta; keep
    the combination with the best validation F2. Returns
    (alpha, beta, gamma, theta, val_f2).
    """
    theta_grid = np.arange(0.05, 0.96, cfg["combiner"]["cutoff_grid_step"]).round(3)
    best = {"alpha": None, "beta": None, "gamma": None, "theta": None, "f2": -1.0}
    for (a, b, g) in weight_simplex(cfg["combiner"]["weight_grid_step"]):
        # Ablations must be constrained in weight space, not simulated by
        # multiplying an omitted signal by a zero-valued vector. Otherwise the
        # returned weights can assign mass to a missing feature and will become
        # invalid when evaluated against the real feature column.
        if not allow_nli and not np.isclose(b, 0.0):
            continue
        if not allow_lexical and not np.isclose(g, 0.0):
            continue
        scores = composite_score(cos_n, nli_n, lex_n, a, b, g)
        theta, f2 = best_theta_for_scores(scores, labels, group_ids, theta_grid)
        if f2 > best["f2"]:
            best = {"alpha": a, "beta": b, "gamma": g, "theta": theta, "f2": f2}
    if best["alpha"] is None:
        raise ValueError("No valid weight combinations for the requested ablation.")
    return best


def fit_logistic_combiner(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_group_ids: np.ndarray,
    cfg,
    *,
    random_state: int = 42,
):
    """
    Feature columns are [cos_n, nli_n, jaccard, type_match, entity_score].
    Each candidate C is fitted on TRAIN only and evaluated on VALIDATION.
    The decision threshold is also selected on validation. The returned model
    remains the train-fitted model so no post-selection probability shift is
    introduced before the held-out test evaluation.
    """
    best_model, best_C, best_f2 = None, None, -1.0
    for Cval in cfg["combiner"]["logreg_C_grid"]:
        clf = LogisticRegression(
            C=Cval,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        clf.fit(train_features, train_labels)
        probs = clf.predict_proba(val_features)[:, 1]
        theta_grid = np.arange(
            0.05, 0.96, cfg["combiner"]["cutoff_grid_step"]
        ).round(3)
        theta, f2 = best_theta_for_scores(
            probs, val_labels, val_group_ids, theta_grid
        )
        if f2 > best_f2:
            best_model, best_C, best_theta, best_f2 = clf, Cval, theta, f2
    return best_model, best_C, best_theta, best_f2
