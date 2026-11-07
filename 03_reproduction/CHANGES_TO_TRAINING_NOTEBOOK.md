# How the 1st-place classifier was retrained

The classifier was trained with Team Kangaroo's public notebook "RSNA 2026 Knee Box-Pooling Classifier" (tascj),
linked from their [1st-place writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution), copied unchanged except for the lines below. Their notebook, labels and boxes are theirs; see the
credits in the main README.

| Run | Notebook | Change | Output |
|---|---|---|---|
| Fold-0 model | [rsna-2026-knee-box-pooling-classifier](https://www.kaggle.com/code/shalomokanume/rsna-2026-knee-box-pooling-classifier) | last cell `DEBUG=1` → `DEBUG=0` (all 8 epochs instead of the 1-epoch demo) | `fold0_ema.pth` |
| All-data model | [knee-classifier-full](https://www.kaggle.com/code/shalomokanume/knee-classifier-full) | `FOLD, N_FOLDS, SEED = 5, 5, 3` (fold 5 does not exist, so nothing is held out) and `EPOCHS = 1 if DEBUG else 5` (fits a 7 h GPU budget; the fold-0 run had stopped improving after epoch 3–5) | `fold5_ema.pth` |

Both runs: Kaggle GPU T4 ×2, internet on, inputs = the competition data and the dataset
`rsna-2026-knee-agent-labels` (CC BY 4.0). The 58 radiologist-labelled studies are never trained on and are scored
after every epoch.

## Inference

[knee-ours-submit](https://www.kaggle.com/code/shalomokanume/knee-ours-submit) runs Team Kangaroo's inference engine
(from their public notebook "knee final 20261022") with two additions, both marked in
`cell1_infer_engine_modified.py`:

1. two extra model entries (`ours_256`, `ours_full`) describing our classifiers (384 px frame, 112 mm crop at 256 px);
2. a loading fix: our checkpoints do not contain the encoder's unused classifier-head norm, which the engine
   otherwise reports as missing.

The knee-side classifiers and ROI detector come from their public dataset `knee-models-common` (not redistributed
here). `cell2_check_or_submit.py` first checks the whole chain on the 58 radiologist studies (`MODE = "check"`:
0.9360 through the engine vs 0.9356 in training), then predicts the test set (`MODE = "submit"`, internet off).
