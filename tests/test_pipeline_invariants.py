import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import combine, data_prep, evaluate


class PipelineInvariantTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "combiner": {
                "weight_grid_step": 0.1,
                "cutoff_grid_step": 0.05,
            }
        }
        self.cos = np.array([0.0, 1.0, 0.2, 0.8])
        self.nli = np.array([1.0, 0.0, 0.3, 0.7])
        self.lex = np.array([0.4, 0.2, 0.9, 0.1])
        self.labels = np.array([0, 1, 0, 1])
        self.groups = np.array(["a", "a", "b", "b"])

    def test_c1_forces_lexical_weight_to_zero(self):
        result = combine.tune_weights_and_theta(
            self.cos,
            self.nli,
            self.lex,
            self.labels,
            self.groups,
            self.cfg,
            allow_lexical=False,
        )
        self.assertAlmostEqual(result["gamma"], 0.0)
        self.assertAlmostEqual(result["alpha"] + result["beta"], 1.0)

    def test_c2_forces_nli_weight_to_zero(self):
        result = combine.tune_weights_and_theta(
            self.cos,
            self.nli,
            self.lex,
            self.labels,
            self.groups,
            self.cfg,
            allow_nli=False,
        )
        self.assertAlmostEqual(result["beta"], 0.0)
        self.assertAlmostEqual(result["alpha"] + result["gamma"], 1.0)

    def test_candidate_alignment_preserves_missed_positive(self):
        full_scores = evaluate.candidate_scores_to_full(
            [0.9], ["cr1"], ["r1"], ["cr1", "cr1"], ["r1", "r2"]
        )
        self.assertEqual(full_scores[0], 0.9)
        self.assertTrue(np.isneginf(full_scores[1]))
        metrics = evaluate.recall_precision_f(
            np.array([0, 1]),
            full_scores >= 0.5,
            np.array(["cr1", "cr1"]),
        )
        self.assertEqual(metrics["recall"], 0.0)

    def test_loader_rejects_incomplete_cross_product(self):
        frame = pd.DataFrame(
            [
                ["c1", "change 1", "r1", "req 1", 1, "A"],
                ["c1", "change 1", "r2", "req 2", 0, "A"],
                ["c2", "change 2", "r1", "req 1", 0, "A"],
            ],
            columns=[
                "cr_id",
                "change_text",
                "requirement_id",
                "requirement_text",
                "label",
                "req_type",
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pairs.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "full CR x requirement"):
                data_prep.load_pairs(path)


if __name__ == "__main__":
    unittest.main()
