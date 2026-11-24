# Paper 1 — Embedding-Based Patient Clustering and LLM Summarization

This directory contains the historical research code, synthetic patient data, and preserved embedding artifacts associated with:

> **K. Torabizadeh, R. Hedjam, M. Allaoui, and B. Abdulrazak.** “Embedding-Enhanced Patient Clustering for Customized Medical Report Summarization using LLMs.” *IEEE ACDSA 2025*. [Publication DOI](https://doi.org/10.1109/ACDSA65407.2025.11165844).

For the combined two-paper project, see the [repository homepage](../../README.md).

## Research overview

Paper 1 compares three dimensionality-reduction methods—**t-SNE**, **UMAP**, and **t-SNE-PSO**—followed by **K-Means (k = 3)** on numerical clinical attributes. The patient groups are then used as input for GPT-4 cluster-level medical-report summarization. Grouping by the original hypertension risk labels serves as the **S1 reference condition**; embedding-derived groups correspond to **S2 (t-SNE), S3 (UMAP), and S4 (t-SNE-PSO)**.

This public code chiefly covers **embedding, clustering, and clustering evaluation**. The supplied scripts do **not** regenerate the GPT-4 summary paragraphs or Text Similarity API results presented in the paper's Table II. Expert refinement and LLM fine-tuning were future-work stages, not implemented experiments.

## Dataset and provenance

The paper describes an approximately 150-record, **GPT-4-generated synthetic hypertension dataset** with patient clinical measurements, a generated clinical report, and a `Hypertension_Stage` label. The **preserved release has 149 rows**, not 150: 35 low-risk (label 0), 62 average-risk (label 1), and 52 high-risk (label 2). It contains 9 columns, including seven numerical clinical variables and two non-feature fields (`Hypertension_Stage`, `Clinical_Report`).

For this repository, use **`S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx` as the canonical workbook**. It preserves the patient table **and** the historical S1 cluster-summary text boxes. `patient_dataset_with_reports_2.csv` contains the same 149-row tabular patient data but cannot contain Excel text boxes. Do not treat the CSV as a second cohort or add a fabricated 150th record.

**Before running the archived scripts:** They read `patient_dataset_with_reports_2.csv` and also read/write `patient_dataset_with_reports.csv` in the current working directory. The latter is a **mutable output accumulator** containing predicted cluster labels and is not identical in purpose to the canonical input dataset. Keep an untouched copy of the canonical workbook. The historical scripts should be run only in a scratch directory because they overwrite the accumulator.

## What the files do

| File or folder | Role | Status |
| --- | --- | --- |
| `S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx` | Canonical 149-patient workbook with original S1 summaries | **Add to this directory** if not already present |
| `patient_dataset_with_reports_2.csv` | Tabular version of the same input cohort | Historical script input |
| `t-SNE.py` | 2D t-SNE, then K-Means and clustering metrics | Historical; non-deterministic and legacy helper dependency |
| `UMAP.py` | 2D UMAP, then K-Means and clustering metrics | Historical; requires `umap-learn` |
| `t-SNE-PSO.py` | 2D UMAP-initialized PSO-based embedding and clustering | Historical; relies on private scikit-learn internals |
| `PCA.py` (if retained) | Supplementary PCA experiment (6D in the supplied script) | Optional historical artifact; not part of Table I |
| `Clustering_evaluation.py` | Clustering accuracy, NMI, silhouette, and custom DBI helpers | Historical; imports removed scikit-learn `linear_assignment_` |
| `*-embedding-NC2.csv`, `*-embedding-NC3.csv` | Saved coordinate artifacts | 3D and PCA archives may need separate upload; see caution below |
| `Results on Hypertension Dataset/` | Preserved figures/results from original study | Retain and label, do not overwrite |
| `evaluate_archived_embeddings.py` | **New**, modern evaluator for saved embeddings | Re-evaluates preserved coordinates without changing original files |
| `REPRODUCIBILITY_AUDIT.md` | Audit of saved coordinates vs. Paper 1 Table I | Verification details and limitations |

## Quick start: re-evaluate the preserved embeddings

This is the **recommended first reproduction step**. It uses the original saved embeddings rather than attempting to regenerate stochastic coordinates.

From the repository root, install the Python dependencies (Python 3.10+ recommended):

```bash
python -m pip install numpy pandas scipy scikit-learn openpyxl
```

Next, enter this Paper 1 folder (the repository contains an additional project directory with the same name):

```bash
cd llm-medical-summarization-clustering-rag/1-clustering-llm-baseline
```

Ensure `S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx` and the archived embedding files are present, then run:

```bash
python evaluate_archived_embeddings.py --dataset S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx
```

This generates `results/recomputed_metrics.csv` with accuracy, NMI, silhouette coefficient (SC), and Davies–Bouldin index (DBI), plus the corresponding Table I values. **It fits a new, seeded K-Means (`n_clusters=3`, `random_state=42`, `n_init=10`) to archived embedding coordinates**, and uses Hungarian label assignment for accuracy. It does not regenerate embedding coordinates or any LLM summaries. Missing embedding CSVs are reported and skipped.

> **Important archive caveat:** `tSNE-embedding-NC3.csv` has **only two actual coordinate columns**, despite `NC3` in its name. It must **not** be represented as the published three-dimensional t-SNE experiment. The verifier flags this explicitly.

## Published clustering results (Table I)

The following are **published scores**, not claims that every provided script independently recreates them.

| Method | Dimensions | Accuracy | NMI | SC ↑ | DBI ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| t-SNE | 2 | 0.899 | 0.701 | 0.736 | 0.355 |
| t-SNE | 3 | 0.738 | 0.450 | 0.349 | 1.060 |
| UMAP | 2 | 0.906 | 0.717 | 0.795 | 0.274 |
| UMAP | 3 | 0.906 | 0.717 | 0.792 | 0.272 |
| t-SNE-PSO | 2 | 0.906 | 0.717 | 0.820 | 0.239 |
| t-SNE-PSO | 3 | 0.906 | 0.717 | 0.837 | 0.222 |

### Verification against archived CSVs

An independent evaluation using the preserved coordinates and seeded K-Means found that **3D UMAP matches all four published values to three decimals**. Several other archived files match accuracy and NMI, but not SC and DBI. The archived nominal 3D t-SNE file cannot validate 3D performance because it has only two columns. See [`REPRODUCIBILITY_AUDIT.md`](REPRODUCIBILITY_AUDIT.md) for the full comparison and what was actually executed.

## Historical source-code limitations

The original files are retained as provenance, **not described as a fully tested clean-install package**:

1. `Clustering_evaluation.py` imports `sklearn.utils.linear_assignment_`, removed from modern scikit-learn. A compatible replacement is `scipy.optimize.linear_sum_assignment`, but that adaptation is **not applied** to the historical file in this folder.
2. The embedding/K-Means scripts do not specify random seeds or a pinned historical dependency environment; rerunning can change metrics.
3. The distributed t-SNE, UMAP, and t-SNE-PSO scripts explicitly generate **2D coordinates**. They do not implement the complete published 3D experiments in their current form.
4. `UMAP.py` requires `umap-learn`. `t-SNE-PSO.py` also imports private scikit-learn t-SNE internals, which are version-sensitive.
5. `PCA.py` is a supplementary experiment, not one of the three algorithms evaluated in published Table I.
6. The published **Table II summary-similarity** experiments require GPT-4 generation, the S1/S2/S3/S4 texts, and the external Text Similarity API; these are not reproduced by the Python clustering scripts.

For any modernized pipeline, place updated scripts alongside the original files under distinct names, record package versions, and report fresh results separately from the published values. **Do not overwrite archived embeddings or tune results to match the publication.**

## Scope and use

These are synthetic, AI-generated research records. Neither the dataset nor generated summaries are a clinical decision-support tool. For provenance and interpretation, cite the [published Paper 1](https://doi.org/10.1109/ACDSA65407.2025.11165844), and consult the repository's general [data and reproducibility documentation](../../docs/REPRODUCIBILITY.md).
