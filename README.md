# Particle Swarm Optimisation for Classifier Ensembles

A Python research prototype exploring clustering, base classifiers, decision fusion and particle swarm optimisation for ensemble selection.

## Start here

- `runProgram.py` selects datasets and repeats experiments.
- `mainProgram.py` coordinates data preparation, clustering, training and evaluation.
- `trainClassifiers.py` builds base learners.
- `decisionFusion.py` combines predictions.
- `optimizeEnsemble.py` and `pso.py` contain optimisation logic.

## Dataset format

Datasets are CSV files. The final column contains the class label. Update `DATASETS` in `runProgram.py` to select dataset names.

```bash
python runProgram.py
```

Run from the repository root after installing NumPy, pandas, SciPy, scikit-learn, Matplotlib and PySwarms in your own virtual environment. The accidentally committed virtual environment and bytecode have been removed from the current tree. The current tree includes `wine.csv` and the default dataset CSVs. Check any additional names you add before running. The script opens Matplotlib plots during experiments. There is no pinned environment; inspect all imports and establish a compatible research environment first.

## Research status

This is an early experimental implementation, retained for research provenance. The cleanup resets cluster lists for every fold and returns the measured optimised score separately from the baseline. These fixes can change results; old CSV outputs are not a benchmark for the corrected code. Validation currently uses a disjoint portion of each held-out fold, so record that protocol when reporting scores. Treat the code as a starting point for reproducibility work, not as a validated benchmark or production library.

[Profile and project directory](https://github.com/zohaibjan)

## Focused regression checks

```bash
python -m unittest discover -s tests -v
```

Synthetic tests replace expensive clustering, learners and optimisation to check fold isolation, separate returned scores and CSV header/append handling. They do not reproduce model quality. Review empty PSO selections and legacy scikit-learn APIs before a full experiment.

## Automated checks

GitHub Actions runs offline regression checks on pushes and pull requests. These do not deploy the app, call paid AI services or establish model quality. `requirements-test.txt` supports the mocked orchestration tests, not a fully reproduced PSO training environment. 
