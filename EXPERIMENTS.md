# Experiment Playbook

This document helps future users run standardized experiments quickly.

## Model variants in this repository

| Family | Dropout setting | Notebook |
|---|---|---|
| MFuRe-CNN | Alpha dropout | `000-MFuReCNN_alpha_do.ipynb` |
| MFuRe-CNN | Standard dropout | `001-MFuReCNN_standard_do.ipynb` |
| MFuRe-CNN | No dropout | `002-MFuReCNN_no_do.ipynb` |
| MFNR-CNN | Alpha dropout | `003-MFNRCNN_alpha_do.ipynb` |
| MFNR-CNN | Standard dropout | `004-MFNRCNN_standard_do.ipynb` |
| MFNR-CNN | No dropout | `005-MFNRCNN_no_do.ipynb` |

## Recommended run order

1. `006-Evaluator.ipynb` for quick baseline validation with pretrained weights.
2. `007-Tester_with_gradcam.ipynb` for single-case testing and visual checks.
3. `008-gradcams.ipynb` for expanded saliency analysis.
4. Training notebooks (`000` to `005`) for retraining/ablation.

## Experiment log template

Copy this block into your notes for every run:

```text
Run ID:
Date:
Commit hash:
Notebook:
Architecture (MFuRe/MFNR):
Condition (alpha/standard/no):
Hardware:
Dataset build/version:
Random seed:
Epochs/batch size:
Primary metrics:
Artifacts path:
Observations:
```

## Deployment-oriented notes

- Freeze environment with `requirements.txt`.
- Keep model + preprocessing assumptions bundled together.
- Validate against representative local clinical data before operational use.
- Add model monitoring when integrating into production systems.
