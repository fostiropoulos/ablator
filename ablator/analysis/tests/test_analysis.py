import unittest
import pandas as pd
from ablator.analysis.main import Analysis

class TestGetBestResultsByMetric(unittest.TestCase):
    def setUp(self):
        self.raw_results = pd.DataFrame({
            "path": ["path1", "path1", "path2", "path2"],
            "accuracy": [0.85, 0.90, 0.80, 0.88],
            "loss": [0.15, 0.10, 0.20, 0.12],
            "index": [0, 1, 0, 1]
        })
        self.metric_map = {"accuracy": "max", "loss": "min"}

    def test_get_best_results_by_metric(self):
        expected_results = pd.DataFrame({
            "path": ["path1", "path2", "path1", "path2"],
            "accuracy": [0.90, 0.88, 0.90, 0.88],
            "loss": [0.10, 0.12, 0.10, 0.12],
            "index": [1, 1, 1, 1],
            "best": ["accuracy", "accuracy", "loss", "loss"]
        })
        analysis_obj = Analysis(
            results=pd.DataFrame(),
            categorical_attributes=[],
            numerical_attributes=[],
            optim_metrics=self.metric_map
        )
        actual_results = analysis_obj._get_best_results_by_metric(self.raw_results, self.metric_map)
        pd.testing.assert_frame_equal(actual_results, expected_results)

if __name__ == "__main__":
    unittest.main()