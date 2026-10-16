# Particle Swarm Optimisation for Classifier Ensembles

A Python research prototype exploring clustering, base classifiers, decision fusion and particle swarm optimisation for ensemble selection.

## Start here

- `runProgram.py` selects datasets and repeats experiments.
- `mainProgram.py` coordinates data preparation, clustering, training and evaluation.
- `trainClassifiers.py` builds base learners.
- `decisionFusion.py` combines predictions.
- `optimizeEnsemble.py` and `pso.py` contain optimisation logic.

## Dataset format

Datasets are CSV files. The final column contains the class label. Update the `data` collection in `runProgram.py` to select dataset names.

```bash
python runProgram.py
```

Run from the repository root after installing the dependencies imported by the modules. There is no pinned environment; inspect all imports and establish a compatible research environment first.

## Research status

This is an early experimental implementation, retained for research provenance. Before using its reported metrics, review fold isolation, validation splitting and the returned baseline/optimised values in `mainProgram.py`. Treat the code as a starting point for reproducibility work, not as a validated benchmark or production library.

[Profile and project directory](https://github.com/zohaibjan)
