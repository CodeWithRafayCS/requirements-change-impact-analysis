"""
Step 07/08: metrics (Section 6.7 / Table 7), matched-cost recall (Table 9),
signal quality (Table 10), and hypothesis tests (Table 11 / Section 6.8).

All metrics are computed PER CHANGE REQUEST first, then macro-averaged, per
Section 6.2.2 ("Metrics are computed per CR and averaged across CRs").
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon


def _per_cr(labels, predicted_mask, group_ids):
    """Yields (y_true, y_pred) arrays for each CR that has at least one true impact."""
    for g in np.unique(group_ids):
        m = group_ids == g
        y = labels[m]
        if y.sum() == 0:
            continue
        yield y, predicted_mask[m].astype(int)


def recall_precision_f(labels, predicted_mask, group_ids, beta=1.0):
    recalls, precisions, fbetas = [], [], []
    for y, pred in _per_cr(labels, predicted_mask, group_ids):
        tp = np.sum((y == 1) & (pred == 1))
        fn = np.sum((y == 1) & (pred == 0))
        fp = np.sum((y == 0) & (pred == 1))
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        if np.isnan(prec):
            prec = 0.0
        denom = (beta ** 2 * prec + rec) if (prec + rec) else 0
        fb = (1 + beta ** 2) * prec * rec / denom if denom else 0.0
        recalls.append(rec); precisions.append(prec); fbetas.append(fb)
    return {
        "recall": float(np.nanmean(recalls)),
        "precision": float(np.nanmean(precisions)),
        f"f{beta:g}": float(np.nanmean(fbetas)),
    }


def inspection_cost(predicted_mask, group_ids, n_total_requirements):
    costs = []
    for g in np.unique(group_ids):
        m = group_ids == g
        costs.append(predicted_mask[m].sum() / n_total_requirements * 100.0)
    return float(np.mean(costs))


def main_metrics_row(labels, predicted_mask, group_ids, n_total_requirements):
    """One row of Table 8 for a single configuration."""
    r1 = recall_precision_f(labels, predicted_mask, group_ids, beta=1.0)
    r2 = recall_precision_f(labels, predicted_mask, group_ids, beta=2.0)
    cost = inspection_cost(predicted_mask, group_ids, n_total_requirements)
    return {
        "recall": r1["recall"], "precision": r1["precision"],
        "f1": r1["f1"], "f2": r2["f2"], "inspection_cost_pct": cost,
    }


def recall_at_top_k(scores, labels, group_ids, budget_pct):
    """
    Table 9: for each CR, inspect the top ceil(budget_pct% * n_requirements_in_CR)
    scored candidates (minimum 1) and compute recall against the full impact set.
    `scores` must already reflect a global ranking usable per CR (e.g. cosine,
    or the composite score; ties broken by index order upstream).
    """
    recalls = []
    for g in np.unique(group_ids):
        m = group_ids == g
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
        recalls.append(tp / y.sum())
    return float(np.mean(recalls)) if recalls else float("nan")


def candidate_scores_to_full(
    candidate_scores,
    candidate_group_ids,
    candidate_requirement_ids,
    full_group_ids,
    full_requirement_ids,
    fill_value=-np.inf,
):
    """
    Align candidate-only scores to the complete CR x requirement universe.

    Requirements removed by Stage 1 receive ``fill_value`` so they rank after
    every candidate, while their labels remain present in downstream metrics.
    This prevents change requests with all positives removed by Stage 1 from
    disappearing from recall/F-score denominators.
    """
    lookup = {
        (str(g), str(r)): float(s)
        for s, g, r in zip(
            candidate_scores, candidate_group_ids, candidate_requirement_ids
        )
    }
    return np.asarray(
        [
            lookup.get((str(g), str(r)), fill_value)
            for g, r in zip(full_group_ids, full_requirement_ids)
        ],
        dtype=float,
    )


def candidate_predictions_to_full(
    candidate_predictions,
    candidate_group_ids,
    candidate_requirement_ids,
    full_group_ids,
    full_requirement_ids,
):
    """Align a candidate-only prediction mask to the full pair universe."""
    scores = candidate_scores_to_full(
        np.asarray(candidate_predictions, dtype=float),
        candidate_group_ids,
        candidate_requirement_ids,
        full_group_ids,
        full_requirement_ids,
        fill_value=0.0,
    )
    return scores.astype(bool)


def recall_at_matched_mean_cost(scores, labels, group_ids, target_mean_cost_pct, n_total_requirements):
    """
    Finds, per CR, the top-k needed to match a GLOBAL target mean inspection
    cost (used for the "at B2's mean cost" column of Table 9), by converting
    the percentage into an absolute k per CR sized proportionally, then
    reusing recall_at_top_k.
    """
    return recall_at_top_k(scores, labels, group_ids, target_mean_cost_pct)


def bootstrap_ci_diff(values_a, values_b, group_ids, n_resamples=10000, seed=0, alpha=0.05):
    """
    Paired bootstrap over UNIQUE change requests (resampling CRs with
    replacement) for the difference in per-CR metric values_a - values_b.
    values_a, values_b: arrays already aggregated to one value per CR
    (same order, aligned to the unique group ids passed as `group_ids`).
    """
    rng = np.random.RandomState(seed)
    n = len(values_a)
    diffs = values_a - values_b
    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(diffs.mean()), (float(lo), float(hi))


def paired_wilcoxon_one_sided(values_a, values_b, alternative="greater"):
    """values_a, values_b: one value per CR, aligned. H0: median diff = 0."""
    diffs = values_a - values_b
    if np.allclose(diffs, 0):
        return float("nan"), 1.0
    stat, p = wilcoxon(values_a, values_b, alternative=alternative, zero_method="wilcox")
    return float(stat), float(p)


def holm_correction(p_values: dict, alpha=0.05):
    """p_values: {name: p}. Returns {name: adjusted_p} using Holm's step-down method."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def signal_auroc_auprc(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def paired_bootstrap_auroc_diff(
    scores_a,
    scores_b,
    labels,
    group_ids,
    n_resamples=10000,
    seed=0,
    alpha=0.05,
):
    """
    Paired cluster bootstrap for AUROC(A) - AUROC(B), resampling whole CRs.

    Returns the observed difference, a percentile confidence interval, and a
    one-sided bootstrap p-value for H1: AUROC(A) > AUROC(B).
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    labels = np.asarray(labels)
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)
    if labels.sum() == 0 or labels.sum() == len(labels) or len(groups) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    observed = roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([np.flatnonzero(group_ids == g) for g in sampled])
        y = labels[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        diffs.append(
            roc_auc_score(y, scores_a[idx]) - roc_auc_score(y, scores_b[idx])
        )
    if not diffs:
        return float(observed), (float("nan"), float("nan")), float("nan")
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Finite-sample correction avoids a reported p-value of exactly zero.
    p_one_sided = (np.sum(diffs <= 0.0) + 1.0) / (len(diffs) + 1.0)
    return float(observed), (float(lo), float(hi)), float(p_one_sided)
