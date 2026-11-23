# Paper 2 — Evidence Input Rebuild

Reconstructed, verified generator for the three evidence artifacts consumed by the Paper 2 hybrid-RAG notebook.

## Input

`patient_dataset_with_reports_2_x.xlsx`, worksheet `patient_dataset_with_reports_2`, containing 149 patient records, clinical measurements, original `Clinical_Report` text, and `Hypertension_Stage` label.

## Reproduce in a Python environment

Requires Python 3.10+ and NumPy (`python -m pip install numpy`). Place the original Excel workbook in the current directory, then run:

```bash
python rebuild_paper2_inputs.py --input patient_dataset_with_reports_2_x.xlsx --outdir Data
```

**Alternative for Google Colab:** Open `Paper2_Rebuild_Evidence_Artifacts_From_Original_Dataset.ipynb`, upload the workbook into Colab Files, and run the cells in order. The resulting files appear in `/content/` and are ready for the Paper 2 RAG notebook.

## Outputs

- `Data/cluster_profile_table.csv` — mean, min, max, and threshold-prevalence fields per risk group.
- `Data/cluster_profile_docs.md` — profile summaries rounded to one decimal place.
- `Data/snippet_bank_reports.jsonl` — one original report snippet per patient, preserving row order and full text.

Expected group sizes: low **35**, average **62**, high **52** (total **149**).

## Validation

The generated files were compared against the preserved original outputs supplied by the project researcher. All three matched **byte-for-byte**. Expected SHA-256 digests:

```
cluster_profile_table.csv: 135459b81dd938c652d1c754a9ef9eeecdfefe7680bf2e440c1b6b793d8b54d2
cluster_profile_docs.md: 2f767594da837796e833057fc7560164697f17307a149902144b2a4abd5b4abe
snippet_bank_reports.jsonl: 538ae70666144f16d112a31e53399d9ec2f2c142531f630630d9f8d73659ed6a
```

This is a **reconstructed and verified equivalent implementation**. Because the historical original generator was unavailable, no claim is made that the source code itself was recovered.

## Rules verified against the outputs

- Label mapping: `Hypertension_Stage` 0 → low, 1 → average, 2 → high risk.
- Blood-pressure threshold fields use systolic **or** diastolic (`>=130/80` and `>=140/90`).
- BMI thresholds are `>=25` and `>=30`; glucose `>=100` and `>=126`; cholesterol `>=200` and `>=240`.
- Prevalence: `(number satisfying threshold / number in cluster) * 100`.
- Patient IDs reflect the original spreadsheet row order, starting at 1.
- The snapshot has 149, **not 150**, patient rows. Do not append a fabricated row merely to match the approximate cohort size in earlier documentation.
