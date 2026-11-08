# Region-wise Parkinson's Disease Detection from Resting-State EEG

Temporal graph neural networks (TGCN / TGIN / TGAT) for Parkinson's disease (PD) classification
from resting-state EEG, with a region-wise ablation that asks a specific question:

> **Does a small, anatomically-targeted electrode group carry as much PD signal as the full 60-channel montage?**

Short answer, on this dataset: **yes — and the left parietal group of 8 electrodes slightly outperforms all 60.**

| Configuration | Channels | Best model | Balanced Acc. | AUC |
|---|---:|---|---:|---:|
| All channels | 60 | TGAT | 0.717 | 0.828 |
| Left Frontal | 10 | TGAT | 0.731 | 0.799 |
| Right Frontal | 10 | TGCN | 0.738 | 0.779 |
| **Left Parietal** | **8** | **TGIN / TGAT** | **0.775** | 0.841 / 0.816 |
| Right Parietal | 8 | TGIN | 0.721 | 0.855 |

McNemar's test finds **no statistically significant difference** between the best full-montage model
and any best regional model (all *p* > 0.05) — i.e. the 8-channel parietal setup is statistically
indistinguishable from the 60-channel one, at 13% of the electrodes.

---

## Dataset

[OpenNeuro **ds004584**](https://openneuro.org/datasets/ds004584) — resting-state EEG.

| | |
|---|---|
| Subjects | 149 (100 PD, 49 Control) |
| Sampling rate | 500 Hz |
| Channels on disk | 63 → **60** retained (intersection across all subjects) |
| Windows after segmentation | **3,915** × 6 s (2,549 PD / 1,366 Control) |
| Label | `1 = PD`, `0 = Control` (from `participants.tsv` → `GROUP`) |

The data itself is **not** in this repository. Notebook 01 downloads it from OpenNeuro's public S3
bucket and writes every intermediate artifact to a Google Drive project folder
(`PD_EEG_Thesis_Features_v2/`). The notebooks are written for **Google Colab**.

---

## Pipeline

The 12 notebooks in `All_Notebooks/` run **in numeric order**. Each one reads what the previous
one wrote, so the chain is resumable — but it is not a package, and there is no orchestration script.

```
01 Acquisition --> 02 Segmentation --> 03 Features --> 04 Split
                                                        |
                                                        +--> 06 Class weights --> 07 Train --+
                                                        |                                    |
                                     05 Model defs -----+                                    v
                                                        +--> 08 Test prep ------------> 09 Test / stats
                                                                                             |
    11 Regional features --> 12 Regional train + test ----------------------------------------+
                                                                                             |
                                                                            10 Interactive inference
```

| Notebook | What it does | Key output |
|---|---|---|
| `01_Data_Acquisition_EDF` | S3 sync of ds004584; per-subject 1–40 Hz FIR bandpass, 60 Hz notch, average reference, 20-component FastICA (EOG + muscle components dropped); export to `.edf` | `manifest.csv`, `data/edf/` |
| `02_Segmentation` | Canonical 60-channel intersection; non-overlapping 6 s windows (3,000 samples); per-channel 0.5/99.5 percentile clipping of residual artifacts | `segments_all.npz` → `(3915, 60, 3000)` |
| `03_Feature_Extraction` | 35 candidate features per 1 s sub-window × 60 channels × 6 timesteps, then quota-based selection | `features_all.npz` → `(3915, 6, 60, 17)` |
| `04_Data_Preprocessing` | **Subject-wise** stratified 70/15/15 split with explicit no-overlap assertions | `subject_split.csv` |
| `05_Model_Selection` | TGCN / TGIN / TGAT definitions + dummy-input sanity check | model architecture |
| `06_Train_Data_Balance` | Inverse-frequency class weights (no synthetic resampling) | `class_weights.npy` |
| `07_Train_Model` | Trains all three models; adjacency built per window; early stopping on **balanced accuracy** with collapse detection | `results/{MODEL}_trained.pt` |
| `08_Test_Data_Preprocessing` | Builds the held-out test tensors + adjacency, unbalanced and untouched | `test_data.npz` |
| `09_Test_Model` | Full-montage evaluation: confusion matrices, ROC/PR, calibration, 13 extended metrics, bootstrap CIs, McNemar, 3D attention maps | `test_results_final.csv` |
| `10_Inference_Visualization` | `ipywidgets` panel — pick subject / 6 s segment / region / model, see the EEG trace and the prediction | interactive |
| `11_Regional_Feature_Extraction` | Re-extracts the **same 17 features** inside 4 ROIs, z-scored and graph-built within each ROI only | `features_{region}.npz` |
| `12_Regional_Model_Training_Testing` | Trains + tests 3 models × 4 regions on the **same subject split**; merges with full-montage results | `all_regions_comparison.csv` |

---

## Feature extraction

Each 6 s window is z-scored as a whole, then cut into **6 non-overlapping 1 s sub-windows**
(the temporal dimension the recurrent head consumes). Features are computed **per channel, per
sub-window**, giving a tensor of shape `(windows, 6 timesteps, 60 nodes, 17 features)`.

**Selection.** 35 candidates were ranked by |Cohen's *d*| (PD vs Control, pooled over all
windows/timesteps/channels), filtered at |*d*| ≥ 0.15, and greedily deduplicated at |*r*| ≤ 0.90 —
within per-domain quotas. The time-frequency domain exhausted its independent candidates at 3,
so 17 features were kept rather than the targeted 20.

| Domain | Selected |
|---|---|
| **Time** (6) | `hjorth_complexity`, `zero_crossing_rate`, `hjorth_mobility`, `line_length`, `kurtosis`, `peak_to_peak` |
| **Frequency** (8) | `beta_power`, `theta_beta_ratio`, `theta_power`, `spectral_entropy`, `alpha_beta_ratio`, `spectral_centroid`, `spectral_edge_95`, `gamma_power` |
| **Time-frequency** (3) | `stft_low_band_power`, `wavelet_entropy`, `teager_energy` |

The strongest single discriminator is **relative beta power** (*d* = −0.527 full-montage), which is
consistent with the beta-band account of parkinsonian cortical activity — and it sharpens when
computed regionally: *d* = −0.436 (L frontal), −0.493 (R frontal), −0.635 (L parietal),
**−0.662 (R parietal)**.

---

## Graph construction

* **Nodes** — EEG channels (60 full montage; 8–10 per ROI).
* **Edges** — Pearson correlation between raw channel time series within the 6 s window,
  thresholded at **r ≥ 0.3**, self-loops removed. One adjacency matrix per window; it is *not*
  a fixed anatomical graph, so connectivity varies window to window.
* **Node features** — the 17 selected features, per timestep.

In the regional runs, z-scoring, correlation and feature extraction all happen **inside the ROI
only** — a region never sees channels outside itself.

---

## Models

All three share the same shape: a graph layer applied independently at each of the 6 timesteps →
pooling to one vector per timestep → **2-layer GRU** → linear classifier on the final hidden state.
`hidden_dim = 48`, `dropout = 0.25`, `heads = 4`.

| Model | Graph operator | Pooling |
|---|---|---|
| **TGCN** | Degree-normalized mean aggregation (`Â X W`) | Learned per-channel attention |
| **TGIN** | GIN-style `MLP((1+ε)·x + Σ neighbors)` | Learned per-channel attention |
| **TGAT** | 4-head masked dot-product attention over the adjacency | Mean (its own attention already weights channels) |

**Training.** Adam, `lr = 1e-3`, `weight_decay = 1e-4`, batch 32, ≤100 epochs, `ReduceLROnPlateau`,
gradient clipping at 1.0, class-weighted cross-entropy. Checkpoint selection and early stopping both
track **validation balanced accuracy** (patience 15, or 25 for TGIN) rather than loss or F1 —
F1 can look healthy while a model collapses onto the majority class; balanced accuracy cannot.
A collapse check refuses to save a degenerate checkpoint silently. Seed 42 throughout; the regional
loop retries up to 3 seeds if a run collapses.

---

## Splitting and class imbalance

The split is **by subject**, stratified on label, with assertions that no subject appears in two
splits — so no window from a training subject can leak into validation or test.

| Split | Subjects | Windows | PD | Control |
|---|---:|---:|---:|---:|
| Train | 103 | 2,709 | 1,737 | 972 |
| Validation | 23 | 579 | 395 | 184 |
| Test | 23 | 627 | 417 | 210 |

The 2:1 PD/Control imbalance is handled by **inverse-frequency loss weighting**
(Control 1.3935, PD 0.7798) — no SMOTE, no oversampling, no synthetic windows. The test set is
left exactly as segmented.

---

## Results

### Full montage (60 channels), test set, n = 627 windows

| Model | Accuracy | Balanced Acc. | F1 | AUC | Sens. | Spec. | MCC | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TGCN | 0.730 | 0.691 | 0.800 | 0.820 | 0.811 | 0.571 | 0.387 | 0.387 |
| TGIN | 0.675 | 0.640 | 0.753 | 0.758 | 0.746 | 0.533 | 0.277 | 0.277 |
| **TGAT** | **0.732** | **0.717** | 0.791 | **0.828** | 0.763 | 0.671 | **0.421** | 0.419 |

Bootstrap 95% CI (2,000 resamples) for TGAT: BalAcc 0.717 [0.679, 0.755], AUC 0.828 [0.797, 0.857].
McNemar: TGCN vs TGAT *p* = 1.000 (indistinguishable); both beat TGIN (*p* = 0.009 / 0.008).

### Region-wise, same split, same 17 features

| Region | Ch. | Models (TGCN / TGIN / TGAT) | Accuracy | Balanced Acc. | F1 | AUC |
|---|---:|---|---:|---:|---:|---:|
| All | 60 | TGCN / TGIN / TGAT | 0.730 / 0.675 / 0.732 | 0.691 / 0.640 / 0.717 | 0.800 / 0.753 / 0.791 | 0.820 / 0.758 / 0.828 |
| Left Frontal | 10 | TGCN / TGIN / TGAT | 0.708 / 0.644 / 0.697 | 0.703 / 0.610 / 0.731 | 0.766 / 0.728 / 0.734 | 0.761 / 0.709 / 0.799 |
| Right Frontal | 10 | TGCN / TGIN / TGAT | 0.702 / 0.700 / 0.633 | 0.738 / 0.728 / 0.647 | 0.737 / 0.740 / 0.687 | 0.779 / 0.758 / 0.724 |
| **Left Parietal** | 8 | TGCN / TGIN / TGAT | **0.774** / 0.764 / 0.756 | 0.766 / **0.775** / **0.775** | **0.823** / 0.807 / 0.796 | **0.860** / 0.841 / 0.816 |
| Right Parietal | 8 | TGCN / TGIN / TGAT | 0.716 / 0.750 / 0.716 | 0.707 / 0.721 / 0.720 | 0.775 / 0.811 / 0.768 | 0.797 / **0.855** / 0.784 |

**Electrode groups** (10-20 montage, intersected with the available 60):

| Region | Channels |
|---|---|
| Left Frontal | Fp1, AF7, AF3, F7, F5, F3, F1, FC5, FC3, FC1 |
| Right Frontal | Fp2, AF8, AF4, F8, F6, F4, F2, FC6, FC4, FC2 |
| Left Parietal | CP5, CP3, CP1, P7, P5, P3, P1, PO7 |
| Right Parietal | CP6, CP4, CP2, P8, P6, P4, P2, PO8 |

(PO3 and PO4 were requested for the parietal groups but are absent from the canonical 60, leaving
8 channels each.)

### Takeaways

1. **Left parietal, 8 electrodes, beats the full 60-channel montage** on every headline metric —
   BalAcc 0.775 vs 0.717, AUC 0.860 vs 0.828. The CIs overlap and McNemar is non-significant, so
   the honest claim is *parity*, not superiority. Parity at 8 electrodes is still the interesting
   result.
2. **Parietal > frontal** on this dataset, in both effect size (beta-power *d* ≈ −0.64 vs ≈ −0.46)
   and downstream classification.
3. **No architecture wins everywhere.** TGAT leads the full montage, TGCN the left parietal AUC,
   TGIN the right parietal AUC. The spread across regions is larger than the spread across models.
4. **Attention concentrates centro-parietally.** The 3D attention maps in notebook 09 put
   CP1/CPz/Cz/CP2 at the top for PD windows across all three models — independently pointing at the
   same area the regional ablation identifies.

---

## Caveats

Read these before citing any number above.

* **Window-level metrics, not subject-level.** Every reported figure is computed over 627 test
  *windows*, not 23 test *subjects*. Windows from one subject are correlated, so these CIs are
  narrower than subject-level CIs would be. The split prevents leakage; it does not make windows
  independent.
* **One split, no cross-validation.** 23 test subjects is small. The bootstrap CIs quantify
  resampling noise *within* this split only — they say nothing about how the numbers would move
  under a different subject partition.
* **The regional comparison reuses the full-montage split.** Correct for a paired comparison, but it
  means the regional results inherit whatever is idiosyncratic about those 23 test subjects.
* **Some preprocessing is fit before splitting.** ICA and filtering are per-subject and therefore
  safe; the dataset-wide percentile clipping in notebooks 02 and 03 is a mild transductive
  dependency.
* **No demographic controls.** `participants.tsv` carries AGE, GENDER, MOCA and UPDRS. None are
  used as covariates or checked for confounding between groups.
* Notebook 09 cells 27 and 30 contain hardcoded reference values from an earlier run, used to verify
  that reloaded checkpoints reproduce the reported numbers (largest observed difference: 0.0005).
  Cell 12 of notebook 07 likewise replots logged training dynamics rather than retraining. Those
  cells are figure generators, not computations.

---

## Reproducing

1. Open `All_Notebooks/01_Data_Acquisition_EDF.ipynb` in Google Colab, mount Drive.
2. Change `PROJECT_ROOT` if you want a different Drive folder — it is set the same way in
   every notebook and must match across all of them.
3. Run 01 → 09 in order for the full-montage result, then 11 → 12 for the regional result.
   Notebook 10 is interactive and needs the checkpoints from 07 and 12.
4. A GPU runtime is needed for 07, 09 and 12. Notebook 01 is the long one — it downloads
   ~3 GB and runs ICA on 149 subjects.

Dependencies installed inline by the notebooks: `mne`, `mne-bids`, `awscli`, `PyWavelets`,
`torch`, `torch-geometric`, `scikit-learn`, `statsmodels`, `ipywidgets`, `pandas`, `matplotlib`.

### Directory layout produced on Drive

```
PD_EEG_Thesis_Features_v2/
├── manifest.csv
├── data/
│   ├── ds004584/              # raw download (.set/.fdt/.tsv/.json)
│   ├── edf/                   # cleaned per-subject .edf
│   ├── segments/              # segments_all.npz
│   ├── features/              # features_all.npz
│   ├── feature_csv/           # combined_features_labeled.csv
│   ├── regional_features/     # features_{region}.npz
│   └── split/                 # subject_split.csv, train/test_data.npz, class_weights.npy
├── models/
├── results/                   # full-montage checkpoints + test_results_final.csv
├── results_regional/          # regional checkpoints + all_regions_comparison.csv
├── figures/
└── figures_paper/
```

---

## Repository contents

```
All_Notebooks/
├── 01_Data_Acquisition_EDF.ipynb
├── 02_Segmentation.ipynb
├── 03_Feature_Extraction.ipynb
├── 04_Data_Preprocessing.ipynb
├── 05_Model_Selection.ipynb
├── 06_Train_Data_Balance.ipynb
├── 07_Train_Model.ipynb
├── 08_Test_Data_Preprocessing.ipynb
├── 09_Test_Model.ipynb
├── 10_Inference_Visualization.ipynb
├── 11_Regional_Feature_Extraction.ipynb
└── 12_Regional_Model_Training_Testing.ipynb
```

Notebooks are committed **with outputs**, so the tables and figures above are inspectable on
GitHub without running anything.

---

## Citation

The dataset:

> Rockhill, A. P., Jackson, N., George, J., Aron, A., & Swann, N. C. *UC San Diego Resting State
> EEG Data from Patients with Parkinson's Disease.* OpenNeuro, ds004584.

If this pipeline is useful, please also cite this repository.
