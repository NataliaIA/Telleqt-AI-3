# Telleqt AI — multi-view defect detection on pet food package seam

Binary classification of package samples:

- `0` = good package
- `1` = bad / seam defect

Each physical sample contains four synchronized views of the same package:

| View id | Meaning |
|---|---|
| `01` | front side, bar light |
| `02` | front side, top light |
| `03` | back side, bar light |
| `04` | back side, top light |

The defect may be visible only from one side or only under one lighting condition, therefore the main model treats the task as **multi-view sample-level classification**, not as independent single-image classification.

---

## Dataset structure

Dataset inspection:

```text
train:
  total samples: 755
  good: 401
  bad: 354
  every sample has exactly 4 images
  source groups / collection sessions: 4

test:
  total samples: 241
  every sample has exactly 4 images
  folder name = sample_id for submission
```

Expected train structure:

```text
train/
  BLTA_MCRL_2026_05_04T07_01_54_620335Z_SCP/
    good/
      sample_.../
        01_...jpg
        02_...jpg
        03_...jpg
        04_...jpg
    bad/
      sample_.../
        01_...jpg
        02_...jpg
        03_...jpg
        04_...jpg
```

Expected test structure:

```text
test/
  1/
    01.jpg
    02.jpg
    03.jpg
    04.jpg
  2/
    01.jpg
    02.jpg
    03.jpg
    04.jpg
```

---

## Main approach

The solution uses a multi-view CNN classifier:

1. Load selected views of a sample in fixed semantic order.
2. Pass every view through the same pretrained `EfficientNet-B0` encoder.
3. Aggregate view embeddings with `mean pooling + max pooling`.
4. Classify the whole physical sample, not each photo separately.
5. Train with `BCEWithLogitsLoss` and class-balanced `pos_weight`.
6. Evaluate with out-of-fold predictions.
7. Select decision threshold from OOF predictions.
8. Predict test samples with an ensemble over CV fold checkpoints.

Why this design:

- the dataset is small, so pretrained CNN features are useful;
- the defect can appear in only one view, so `max pooling` helps preserve a strong defect signal from a single image;
- `mean pooling` stabilizes prediction over all views;
- train subfolders were collected at different times, so group-aware CV is more honest than random sample splitting.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Put data into the project:

```text
project_root/
  data/
    train/
      BLTA_MCRL_.../
        good/
        bad/
    test/
      1/
      2/
      ...
```

If you have archives:

```bash
mkdir -p data
unzip train.zip -d data/train
unzip test.zip -d data/test
```

If unzipping creates an extra nested folder, pass the real folder path to commands below.

---

## Optional: inspect dataset structure

```bash
python scripts/inspect_dataset_structure.py \
  --train data/train \
  --test data/test \
  --out-dir dataset_report
```

This creates:

```text
dataset_report/
  dataset_structure_report.json
  train_samples.csv
  test_samples.csv
```

---

## 1. Main group-CV training

Recommended command for this dataset:

```bash
python -m telleqt_defects.train_cv \
  --train-root data/train \
  --out-dir runs/effnet_b0_group_cv \
  --cv group \
  --folds 4 \
  --epochs 20 \
  --batch-size 8 \
  --image-size 384 \
  --views all
```

Because there are four source collection groups, group CV uses four folds. This is similar to leave-one-acquisition-session-out validation.

For a faster smoke test:

```bash
python -m telleqt_defects.train_cv \
  --train-root data/train \
  --out-dir runs/smoke \
  --cv group \
  --folds 4 \
  --epochs 1 \
  --batch-size 4 \
  --image-size 224 \
  --views all
```

---

## 2. Threshold strategy for industrial use

Default mode selects threshold by best OOF F1:

```bash
--threshold-strategy f1
```

For industrial defect detection, false negatives are usually more expensive than false positives: missing a defective package is worse than rejecting a good one. Therefore the project also supports high-recall threshold selection:

```bash
python -m telleqt_defects.train_cv \
  --train-root data/train \
  --out-dir runs/effnet_b0_high_recall \
  --cv group \
  --folds 4 \
  --epochs 20 \
  --batch-size 8 \
  --image-size 384 \
  --views all \
  --threshold-strategy target_recall \
  --target-recall 0.95
```

The run saves `threshold_report.csv` with several operating points:

```text
fixed_0.50
best_f1
target_recall_0.90
target_recall_0.95
target_recall_0.98
```

This makes the recall/FPR trade-off explicit.

---

## 3. Outputs after training

Training writes files into `--out-dir`:

```text
runs/effnet_b0_group_cv/
  fold_0.pt
  fold_1.pt
  fold_2.pt
  fold_3.pt
  dataset_summary.json
  folds.csv
  oof_predictions.csv
  threshold.txt
  threshold_report.csv
  threshold_report.json
  metrics.json
  confusion_matrix.png
  pr_curve.png
```

`metrics.json` contains the required metrics:

```json
{
  "threshold": 0.42,
  "confusion_matrix_labels": ["good_0", "bad_1"],
  "confusion_matrix": [[...], [...]],
  "tn": 0,
  "fp": 0,
  "fn": 0,
  "tp": 0,
  "recall_bad": 0.0,
  "false_positive_rate": 0.0,
  "pr_auc": 0.0
}
```

---

## 4. Error analysis: false positives / false negatives

After training, save visual grids of mistakes:

```bash
python -m telleqt_defects.error_analysis \
  --train-root data/train \
  --run-dir runs/effnet_b0_group_cv \
  --views all \
  --max-per-type 50
```

Output:

```text
runs/effnet_b0_group_cv/errors/
  error_report.csv
  error_summary.csv
  false_positives/
    <sample>.jpg
  false_negatives/
    <sample>.jpg
```

Why this is useful:

- false negatives show which defects the model misses;
- false positives show which good packages look defect-like;
- this is a production-oriented diagnostic step, not just a leaderboard trick.

---

## 5. View ablation study

The code can train the same pipeline on different view subsets:

```bash
python -m telleqt_defects.run_ablation \
  --train-root data/train \
  --out-dir runs/ablation \
  --cv group \
  --folds 4 \
  --epochs 10 \
  --batch-size 8 \
  --image-size 384
```

Experiments included:

| Experiment | Views |
|---|---|
| `01_front_barlight` | `01` |
| `02_front_toplight` | `02` |
| `03_back_barlight` | `03` |
| `04_back_toplight` | `04` |
| `front_01_02` | `01,02` |
| `back_03_04` | `03,04` |
| `barlight_01_03` | `01,03` |
| `toplight_02_04` | `02,04` |
| `all_01_02_03_04` | `01,02,03,04` |

Summary is saved to:

```text
runs/ablation/ablation_summary.csv
```

For a shorter ablation run:

```bash
python -m telleqt_defects.run_ablation \
  --train-root data/train \
  --out-dir runs/ablation_short \
  --epochs 8 \
  --only front_01_02,back_03_04,all_01_02_03_04
```

This helps answer an important industrial question: which camera side and lighting condition actually carry the defect signal.

---

## 6. Optional Grad-CAM explanations

Generate heatmaps for confident bad samples:

```bash
python -m telleqt_defects.gradcam \
  --train-root data/train \
  --model-dir runs/effnet_b0_group_cv \
  --from-oof confident_bad \
  --top-k 8
```

Generate heatmaps for false negatives:

```bash
python -m telleqt_defects.gradcam \
  --train-root data/train \
  --model-dir runs/effnet_b0_group_cv \
  --from-oof false_negative \
  --top-k 8
```

Output:

```text
runs/effnet_b0_group_cv/gradcam/
  <sample>.jpg
```

The goal is not to claim perfect explainability, but to sanity-check that the model looks at meaningful package/seam areas rather than only background artifacts.

---

## 7. Create final submission CSV

```bash
python -m telleqt_defects.predict \
  --test-root data/test \
  --model-dir runs/effnet_b0_group_cv \
  --out-csv submission.csv \
  --batch-size 8
```

The script uses all `fold_*.pt` checkpoints and averages probabilities. The result format is exactly:

```csv
sample_id,prediction
1,0
2,0
3,1
```

For high-recall submission, use the high-recall run directory:

```bash
python -m telleqt_defects.predict \
  --test-root data/test \
  --model-dir runs/effnet_b0_high_recall \
  --out-csv submission_high_recall.csv
```

---

## Suggested report wording

The task was treated as a multi-view binary classification problem. Each physical package sample contains four synchronized views: front/back sides captured under two lighting conditions. Since the defect can be visible only on one side or under one lighting setup, the model processes all four images jointly.

The pipeline uses a shared CNN encoder for all views and aggregates view-level features using mean and max pooling. Cross-validation is performed with grouped splits by acquisition session to reduce leakage between train and validation. The decision threshold is selected on out-of-fold predictions, with an additional high-recall operating point suitable for industrial defect detection, where false negatives are more expensive than false positives.
