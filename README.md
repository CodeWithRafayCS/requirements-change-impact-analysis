# AI-Based Requirements Change Impact Analysis — Corrected Pipeline

This repository evaluates seven local change-impact-analysis configurations:
three baselines (B1–B3), three hand-tuned hybrid/ablation variants (C1–C3),
and one learned combiner (C4).

## Important evaluation guarantees

The corrected pipeline enforces these invariants:

1. **C1 is embedding + NLI only** (`gamma = 0`).
2. **C2 is embedding + lexical only** (`beta = 0`).
3. Metrics for every method are computed against the complete
   change-request × requirement universe. A change request is never removed
   merely because Stage 1 missed all of its true impacts.
4. Fixed inspection budgets are percentages of all requirements, not
   percentages of the Stage-1 candidate subset.
5. No C1–C4 recall may exceed the Stage-1 recall ceiling.
6. C4 is trained on the training split; its regularization and decision
   threshold are selected on validation; the held-out test split is evaluated
   afterward.
7. H4 uses a paired cluster bootstrap of
   `AUROC(NLI) - AUROC(cosine)`, resampling complete change requests.

## Dataset

`data/pairs.csv` must contain exactly these columns:

| Column | Meaning |
|---|---|
| `cr_id` | Change-request identifier |
| `change_text` | Change rationale text |
| `requirement_id` | Candidate requirement identifier |
| `requirement_text` | Candidate requirement text |
| `label` | `1` if impacted, otherwise `0` |
| `req_type` | Optional requirement type/module |

The loader now rejects duplicate pairs, non-binary labels, and incomplete
cross products. Every requirement must appear once for every change request.

The included `pairs_SYNTHETIC_DEMO.csv` is only a smoke-test dataset. Do not
report its results as research findings.

The provenance and license of `pairs.csv` must be documented separately before
publication. Its `MRM-*` identifiers appear to be Apache Archiva/JIRA-derived;
do not call it WASP or SAT-DLink unless that identity is independently verified.

## Install

Use a fresh virtual environment. Do not reuse the removed Windows `.venv`.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For strict reproducibility, record the final installed versions with:

```bash
pip freeze > environment-lock.txt
```

## Run

### Fast rerun using the included trusted caches

The supplied caches already contain embeddings, NLI scores, and lexical
features. To regenerate the corrected tuned models, metrics, tests, and plots:

```bash
python run_pipeline.py --step 06_tune
python run_pipeline.py --step 07_test_eval
python run_pipeline.py --step 08_stats
python run_pipeline.py --step 09_sensitivity_and_errors
python run_pipeline.py --step 10_report
```

Only load these pickle/joblib caches if you trust this archive. For an entirely
fresh run, remove `cache/` and use the command below.

### Full clean run

```bash
python run_pipeline.py --all --no-cache
```

This downloads/loads the configured embedding, NLI, and spaCy models and can
take substantial CPU time. `--all` runs the main reformulated NLI template.
The optional plain-template scores can be generated separately with:

```bash
python run_pipeline.py --step 04_nli_scores --plain
```

### Synthetic smoke test

```bash
python run_pipeline.py --all --config config_demo.yaml --no-cache
```

## Output files

- `results/table3_dataset_stats.csv`
- `results/table8_main_results.csv`
- `results/table9_matched_cost.csv`
- `results/table10_signal_quality.csv`
- `results/table11_hypothesis_tests.csv` (includes H1, H1a, H2, H3a, H3b, H4)
- `results/table12_sensitivity.csv`
- `results/tau_calibration_diagnostics.csv`
- `results/error_analysis_sample.csv`
- `results/recall_vs_cost.png`
- `results/nli_score_distribution.png`
- `results/frozen_parameters.json`
- `config_used.yaml`

The result filenames retain their historical numbering for compatibility; use
the CSV headers rather than assuming they match final paper table numbers.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Known scope limitations

- The main experiment uses one pre-registered holdout split (`seed=42`). It
  does not claim repeated-seed evaluation.
- The nested five-fold tuning path for datasets below the configured CR
  threshold is intentionally not implemented and raises `NotImplementedError`.
- The plain NLI-template sensitivity score can be generated, but it is not part
  of the main seven-configuration evaluation.
- Published ProReFiCIA rows are contextual external numbers, not a controlled
  same-dataset comparison.
