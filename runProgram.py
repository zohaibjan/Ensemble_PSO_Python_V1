"""Run the historical ensemble experiments and append one CSV row per dataset."""
import csv
from pathlib import Path

import numpy as np
from mainProgram import mainProgram

DATASETS = ('thyroid', 'wine', 'diabetes', 'segment', 'ecoli', 'cancer',
            'vehicle', 'iris', 'liver', 'ionosphere', 'sonar', 'glass')


def run_experiments(datasets=DATASETS, num_runs=1, output='results.csv'):
    if num_runs < 1:
        raise ValueError('num_runs must be at least one')
    output = Path(output)
    for dataset in datasets:
        scores = [mainProgram(dataset) for _ in range(num_runs)]
        accuracy, optimized_accuracy = np.mean(scores, axis=0)
        needs_header = not output.exists() or output.stat().st_size == 0
        with output.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(['dataset', 'accuracy', 'optimized_accuracy'])
            writer.writerow([dataset, accuracy, optimized_accuracy])
        print(f'{dataset}: baseline={accuracy}, optimized={optimized_accuracy}')


if __name__ == '__main__':
    run_experiments()
