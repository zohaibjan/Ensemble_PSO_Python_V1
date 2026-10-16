"""Synthetic orchestration tests; expensive learners and PSO are mocked."""
import csv
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
optimizer_module = types.ModuleType('optimizeEnsemble')
optimizer_module.optimizeEnsemble = Mock()
with patch.dict(sys.modules, {'optimizeEnsemble': optimizer_module}):
    experiment = importlib.import_module('mainProgram')
    runner = importlib.import_module('runProgram')


class FakeKMeans:
    def __init__(self, **kwargs):
        pass

    def fit(self, values):
        self.labels_ = np.zeros(len(values), dtype=int)
        return self


class RegressionTests(unittest.TestCase):
    def test_fold_isolation_and_distinct_scores(self):
        baseline, selected = object(), object()
        values = np.arange(150, dtype=float).reshape(50, 3) + 1
        labels = np.tile([0, 1], 25)

        def train(x_clusters, y_clusters):
            self.assertEqual(len(x_clusters), 1)
            self.assertEqual(len(y_clusters), 1)
            self.assertEqual(len(x_clusters[0]), 45)
            return baseline

        with patch.object(experiment.loadData, 'load_dataset', return_value=(values, labels)), \
             patch.object(experiment, 'KMeans', FakeKMeans), \
             patch.object(experiment, 'silhouette_score', return_value=1), \
             patch.object(experiment, 'plt'), \
             patch.object(experiment, 'trainClassifiers', side_effect=train) as trainer, \
             patch.object(experiment, 'optimizeEnsemble', return_value=selected), \
             patch.object(experiment, 'decisionFusion', side_effect=lambda pool, *_: 0.8 if pool is selected else 0.6):
            ordinary, optimized = experiment.mainProgram('synthetic')
        self.assertEqual(trainer.call_count, 10)
        self.assertAlmostEqual(ordinary, 0.6)
        self.assertAlmostEqual(optimized, 0.8)

    def test_csv_has_one_header_and_correct_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'scores.csv'
            with patch.object(runner, 'mainProgram', side_effect=[(0.2, 0.4), (0.6, 0.8), (0.1, 0.9)]):
                runner.run_experiments(['first'], num_runs=2, output=output)
                runner.run_experiments(['second'], output=output)
            with output.open(newline='') as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ['dataset', 'accuracy', 'optimized_accuracy'])
        self.assertAlmostEqual(float(rows[1][1]), 0.4)
        self.assertAlmostEqual(float(rows[1][2]), 0.6)
        self.assertEqual(rows[2][0], 'second')

    def test_rejects_zero_runs(self):
        with self.assertRaises(ValueError):
            runner.run_experiments([], num_runs=0)


if __name__ == '__main__':
    unittest.main()
