# Reproducibility Guide (MFuRe-CNN / MFNR-CNN)

This guide standardizes how to rerun, compare, and report experiments from the BSPC paper.

## 1) Minimum environment report per run

Record the following in your experiment log:

- Git commit hash
- Notebook filename
- Python version
- TensorFlow/Keras version
- CUDA/cuDNN versions (if GPU)
- GPU model and VRAM
- Dataset source and extraction date
- Random seed values

## 2) Determinism controls

In notebook cells before model creation, set seeds consistently:

```python
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

> Full numerical determinism can still vary by hardware and TensorFlow ops.

## 3) Data placement contract

- Prepared dataset must be extracted under `data/`.
- Pretrained weights must be extracted under `models/`.
- Keep original archives (`.rar`, `.zip`) and checksums externally for traceability.

## 4) Evaluation protocol

Use `006-Evaluator.ipynb` with explicit model metadata:

- `architecture`: `MFuRe` or `MFNR`
- `condition`: `alpha`, `standard`, `no`

For each run, save:

- Accuracy, precision, recall, F1
- Confusion matrix figure
- Per-class report
- Timestamp + commit hash

## 5) Explainability protocol

Use:

- `007-Tester_with_gradcam.ipynb`
- `008-gradcams.ipynb`

Archive outputs for each class/case. This supports qualitative auditing and result review.

## 6) Safe experiment hygiene

- Do not overwrite working checkpoints unintentionally.
- Use versioned filenames for retrained weights.
- Separate baseline, ablation, and exploratory runs.

## 7) Suggested publication checklist

- [ ] Dataset source and preparation documented
- [ ] Exact notebook(s) and parameters listed
- [ ] Hardware/software stack reported
- [ ] Metrics + confusion matrix exported
- [ ] Explainability examples included
- [ ] Citation and DOI links included
