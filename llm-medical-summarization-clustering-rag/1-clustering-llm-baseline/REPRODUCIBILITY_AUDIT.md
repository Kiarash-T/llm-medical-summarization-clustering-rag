# Paper 1: clustering reproducibility audit

**Scope:** Uploaded Paper 1 source scripts and eight preserved embedding CSVs, compared against Table I in the published IEEE ACDSA 2025 paper *Embedding-Enhanced Patient Clustering for Customized Medical Report Summarization using LLMs*.

## Dataset and verification

- The source is the supplied `S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx`, worksheet `patient_dataset_with_reports_2` (149 patient rows, 9 columns). The tabular cells match the preserved `patient_dataset_with_reports_2.csv`; the workbook also contains overall text-box summaries.
- Label counts: low 35; average 62; high 52.
- To recalculate saved embeddings, use K-Means (`n_clusters=3`, `random_state=42`, `n_init=10`), Hungarian label matching for accuracy, normalized mutual information, Euclidean silhouette, and Davies–Bouldin index.
- These metrics were recomputed from the **preserved embedding artifacts**, not newly regenerated embeddings, except where explicitly stated below.

## Table I: published vs recalculated (three-decimal rounding)

| Method | Nominal dims | Published ACC / NMI / SC / DBI | Recalculated ACC / NMI / SC / DBI | Status |
|---|---:|---|---|---|
| tSNE | 2 | 0.899/0.701/0.736/0.355 | 0.899/0.701/0.731/0.366 | Partial / different |
| tSNE | 3 | 0.738/0.450/0.349/1.060 | 0.866/0.636/0.729/0.362 | Not a valid 3D reproduction: file contains 2 coordinate columns |
| UMAP | 2 | 0.906/0.717/0.795/0.274 | 0.906/0.717/0.778/0.306 | Partial / different |
| UMAP | 3 | 0.906/0.717/0.792/0.272 | 0.906/0.717/0.792/0.272 | **All four reproduced** |
| t-SNE-PSO | 2 | 0.906/0.717/0.820/0.239 | 0.906/0.717/0.854/0.206 | Partial / different |
| t-SNE-PSO | 3 | 0.906/0.717/0.837/0.222 | 0.906/0.717/0.830/0.241 | Partial / different |

## Original scripts: execution tests

Using a fresh temporary working directory and copies of the source files:

| Script | Unmodified result | Result with isolated compatibility fix |
|---|---|---|
| `Clustering_evaluation.py` | Import fails: `sklearn.utils.linear_assignment_` no longer exists in scikit-learn 1.8.0 | Use `scipy.optimize.linear_sum_assignment`; all stored-embedding metrics can be computed |
| `PCA.py` | Fails on the outdated evaluation-helper import | Ran 6D PCA + K-Means: ACC 0.89933, NMI 0.70052, SC 0.48070, DBI 0.74566 |
| `t-SNE.py` | Fails on the outdated evaluation-helper import | Ran 2D t-SNE + K-Means: ACC 0.89933, NMI 0.70052, SC 0.72814, DBI 0.37050 |
| `UMAP.py` | Cannot start: missing `umap-learn` | Not executed (package unavailable in the test environment) |
| `t-SNE-PSO.py` | Cannot start: missing `umap-learn` | Not executed; further compatibility work is also necessary for private scikit-learn t-SNE utilities |

An independent **temporary 3D** alteration of `t-SNE.py` produced ACC 0.90604, NMI 0.71709, SC 0.39190, DBI 0.98799, **not** the paper's 3D t-SNE figures (0.738, 0.450, 0.349, 1.060). These outcomes must not be described as a reproduction of the original 3D experiment.

## Reproducibility blockers and likely sources of disagreement

1. **Missing input dataset path:** All individual model scripts read both `patient_dataset_with_reports_2.csv` and `patient_dataset_with_reports.csv`, but only the first was supplied in this audit. A temporary copy of the original 149-row dataset was used as the second input so the scripts could be run in isolation; the scripts then overwrite it with cluster labels. A clean generator should not overwrite a canonical input.
2. **Obsolete API:** `sklearn.utils.linear_assignment_` was removed from current scikit-learn. Implement the equivalent Hungarian assignment via `scipy.optimize.linear_sum_assignment` for modern compatibility.
3. **Dependency gap:** `umap-learn` was not installed; an install attempt failed because this execution environment has no network access. Hence new UMAP and t-SNE-PSO embeddings could not be run. Their *preserved* embedding metrics were calculated instead.
4. **Dimensionality mismatch:** `tSNE-embedding-NC3.csv` contains **two** numeric columns. It is not a 3D embedding despite its filename.
5. **Published vs archived output discrepancy:** Some saved embeddings yield the published accuracy and NMI but different SC and DBI. That points to different saved coordinates, experimental settings, or model versions; no exact cause can be established from these files alone. Results should not be overwritten to force a match.
6. **No fixed random seed:** Source scripts invoke stochastic t-SNE/UMAP/K-Means without `random_state`, and historical library versions are not pinned. The published embedding coordinates may not be reproduced in newer environments.
7. **Only 2D model scripts are provided:** `t-SNE.py`, `UMAP.py`, and `t-SNE-PSO.py` set `n_components=2` directly; no complete separate 3D generation code was uploaded. `PCA.py` generates 6D, not the uploaded 2D/3D PCA archives.
8. **Summary-similarity reproduction is a different task:** These scripts evaluate *clustering* only. They do not regenerate the GPT-4 S1–S4 summaries or call the Text Similarity API used for Paper 1 Table II; Table II is **not reproduced** by this audit.

## Interpretation

- The **strongest exact result** is saved UMAP 3D (ACC 0.906, NMI 0.717, SC 0.792, DBI 0.272), matching Paper 1 Table I to the printed precision.
- Several other artifacts reproduce **ACC and NMI** but not **SC and DBI**.
- Paper 1 Table I is **partially reproduced from saved artifacts**. The *full original experiment* is not yet independently reproducible using the supplied scripts alone.
- No attempt was made to fabricate missing 3D outputs or tune random seeds to match the paper.
