# Toy Analysis

This repository contains two small, self-contained artifacts:

- `recruitment_attrition_analysis.py`: a recruitment attrition analysis script for a local candidate dataset
- `constrained_affine.py`: a PyTorch implementation of a vectorized constrained affine block with mixed per-feature bounds

## Constrained Affine Block

`constrained_affine.py` provides:

- `VectorizedBoundedParameter`
- `ParallelConstrainedAffineBlock`

The design supports mixed constraints in a single parameter vector:

- unbounded
- lower-only
- upper-only
- both-bounded
- fixed-value

The corresponding notebook demo is:

- `constrained_affine_demo.ipynb`

## Recruitment Attrition Analysis

The original analysis workflow is kept in:

- `recruitment_attrition_analysis.py`

It expects a local CSV with the required columns and can optionally render summary plots.

## Setup

Create a virtual environment and install dependencies:

```powershell
.\setup_venv.ps1
pip install -r requirements.txt
```

## Notes

- Local virtual environments, caches, and data files are intentionally ignored by git.
- The constrained affine notebook is written in English and is intended as the executable example for the PyTorch component.

