# RSNA 2026 Knee Abnormality Detection: from my own pipeline to a 0.956 reproduction

**Task.** Predict 12 knee MRI findings (ACL, MCL, medial / lateral meniscus, medial / lateral / patellofemoral
osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion, fracture) for about 1,300 test studies, each
with several MRI series. Score: mean AUC over the 12 findings.
([competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection))

**The catch.** Only 58 of the 4,407 training studies have radiologist labels. Every other label has to be built
from free-text radiology reports (several languages) or from the images themselves, while the test set is graded by
radiologists reading the images. This turned out to be a *labelling* problem more than a modelling one.

**Result.** A late submission scoring **0.956 private / 0.965 public** (the competition's 10th place scored 0.957,
1st place 0.960). Late submissions are not ranked; the score is reported for comparison only.

I am a medical student, and I built this with an AI assistant (Claude) helping me write and debug the code. What
this repository documents is the decisions: what I tried, how I measured it, and what each step changed.

---

## Results

| Step | What changed | Radiologist-58 AUC | Private | Public |
|---|---|---|---|---|
| Public notebook (heliosli, unchanged) | starting point | n/a | 0.942 | 0.958 |
| Own pipeline in a blend | public 2.5D reader 0.8 + my 3 fold models 0.2 | n/a | 0.917 | 0.930 |
| Own pipeline, report labels | 5 folds, labels built from reports | ≈0.923 | not submitted alone | |
| Own pipeline, image-read labels | same model, Team Kangaroo's agent labels, 2 folds | 0.927 | not submitted | |
| **Reproduction of the 1st-place classifier, fold 0** | anatomy-box pooling, image-read labels, 8 epochs | 0.936 | **0.954** | 0.963 |
| **Same recipe on all the training data** | no held-out fold, 5 epochs | 0.939 | **0.956** | 0.966 |
| Average of the two | | 0.938 | 0.956 | 0.965 |

The 58 radiologist studies are never trained on, but the image-read labels were partly tuned on them, so their AUCs
are optimistic. With only 58 studies, differences under about 0.02 are within noise: the averaged model scored
*lower* than the all-data model on the 58 but tied it on the 1,300-study private test.

### Per finding (final models, radiologist-58)

| Finding | Own pipeline (image-read labels) | Reproduction (average of both) |
|---|---|---|
| ACL | 0.995 | 0.998 |
| MCL | 0.921 | 0.957 |
| Medial meniscus | 0.978 | 0.963 |
| **Lateral meniscus** | 0.867 | **0.908** |
| Medial OA | 0.992 | 0.978 |
| Lateral OA | 0.836 | 0.826 |
| PF OA | 0.857 | 0.883 |
| Effusion | 0.971 | 0.993 |
| Synovitis | 0.859 | 0.875 |
| Baker's cyst | 0.966 | 0.993 |
| Contusion | 0.928 | 0.937 |
| Fracture | 0.953 | 0.942 |

The biggest gain is where anatomy matters most: the reproduction pools each finding only inside a 3D box around the
relevant structure, so the lateral meniscus decision is made from the lateral meniscus.

---

## What I did, in order

**1. My own pipeline (`01_own_pipeline/`).**
- `A_cache_shard.py`: decodes the DICOM series into a compact cache (0.6 mm/px, 256 px, slices sorted along the
  slice normal), split into 5 CPU notebooks to stay under Kaggle's 20 GB output limit.
- `B_labels.py`: builds soft labels by scoring every available report-label source against the 58 radiologist
  studies and averaging the best sources per finding.
- `C_train.py` + `E_more_folds_driver.py`: a 2.5D model (3-slice windows from every series → CNN → a small
  transformer over the whole study → one attention pool per finding). Each fold runs in a fresh process because
  memory crept up across folds; direct disk reads and a RAM guard fixed out-of-memory crashes.
- `D_blend_into_submission.py`: rank-blends the fold models into a public notebook's submission.

**2. Reading the winners' solutions.** After the competition closed, the top-5 write-ups agreed: the biggest gains
came from **labels read from the images by vision-language models**, calibrated on the 58 radiologist studies, not
from bigger models or extra data. Report-only labels score about 0.90 against the radiologists; image-read labels
about 0.94.

**3. Testing that on my own model (`02_label_experiment/`).** `L1_score_labels.py` scores label tables against the
58 studies (AUC, log loss, confident errors). Team Kangaroo's published labels scored 0.943 against my 0.929.
Retraining my own pipeline on them moved it only from about 0.923 to 0.927, so labels alone were not enough for my
architecture.

**4. Reproducing the 1st-place classifier (`03_reproduction/`).** Team Kangaroo published their labels, ROI boxes
and training notebooks. I retrained their box-pooling ConvNeXt-T classifier (one fold, then all the data) on free
Kaggle T4 GPUs, plugged my checkpoints into their inference engine, verified the full chain on the 58 studies
(0.9360 vs 0.9356 in training) before spending a submission, and submitted. See
`03_reproduction/CHANGES_TO_TRAINING_NOTEBOOK.md` for exactly what I changed.

---

## Clinical caveats

This is a research prototype, not a clinical tool.

- **Not reliable for every finding.** Lateral OA (0.83) and PF OA (0.88) are well below what a radiologist would
  accept, and these were the weakest findings for every top team too.
- **AUC is not a clinical decision.** Clinical use needs a chosen operating point with sensitivity and specificity
  (with confidence intervals) per finding, chosen for a specific job such as ruling out normal studies.
- **Not tested outside the competition hospitals.** Different scanners, protocols and populations can change
  performance; this would need external and prospective validation, and regulatory approval, before any patient use.

## What I learned

- In medical imaging with few expert labels, **label quality sets the ceiling**; architecture and tuning decide how
  close you get.
- **Small test sets lie.** 58 studies move by about ±0.02 by chance; decisions that looked better there did not
  always transfer, and the reverse.
- **Verify before you spend.** Checking the whole inference chain on known studies caught nothing wrong this time,
  but costs 10 minutes and protects every later submission.
- **Reproducing a winning solution is a skill in itself**: reading someone else's code closely enough to adapt it,
  and measuring each change.

## Credits

- **Team Kangaroo (tascj, liushuzhi), 1st place**: the agent-made labels, ROI boxes and knee-side table
  ([rsna-2026-knee-agent-labels](https://www.kaggle.com/datasets/tascj0/rsna-2026-knee-agent-labels), CC BY 4.0), the Box-Pooling Classifier,
  ROI Detector and Side Classifier notebooks, the inference engine, and the `knee-models-common` detector / side
  weights (used, not redistributed). `03_reproduction/cell1_infer_engine_modified.py` is their code with my two
  marked changes.
- **goodpjw2008** (2.5D ConvNeXt reader, Apache-2.0) and **heliosli** (public 0.9586 notebook, Apache-2.0): the
  preprocessing and blending starting points of my own pipeline.
- **stevenleehans, Pilkwang Kim, lixin73, laymond**: public report-label tables used in `B_labels.py`.
- RSNA, the annotating radiologists and Kaggle for the competition.

No competition data, model weights or secrets are stored in this repository.
