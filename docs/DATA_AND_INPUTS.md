# Data provenance and required inputs

## Dataset scope

Both publications report experiments using approximately 150 **synthetic hypertension patient records** with three clinical risk groups. These are **not real electronic health records**. This repository should not be presented as clinically validated or ready for diagnosis or treatment decisions.

The inspected Paper 2 artifact `snippet_bank_reports.jsonl` has **149 JSONL records**: low-risk 35, average-risk 62, and high-risk 52. With three cluster profiles, the retriever's initial document list has 152 entries. This differs from an older code comment expecting 150 snippets/153 combined documents. State the counts in each released dataset snapshot and archive a checksum. Do not silently fabricate a missing record.

## Paper 2 input contract

| File | Required content | Used for |
| --- | --- | --- |
| `cluster_profile_docs.md` | Three fenced profile blocks with risk group in first line | `[C]` structured evidence |
| `snippet_bank_reports.jsonl` | One JSON object per line: `snippet_id`, `risk_group`, `snippet_text` | Retrieval snippet bank |
| `cluster_profile_table.csv` | `risk_group` and percentage columns for BP/BMI/glucose/cholesterol | Must-mention indicator evaluation |
| `step7_eval_input_template.json` | List of three objects with `cluster` keys | Producing paired evaluation JSON files |
| `S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx` | Overall S1 summaries in Excel **drawing text boxes** | S1 reference summaries |
| `S4_TSNE-PSO-patient_dataset_with_report_clustering_tsn-pso.xlsx` | Overall S4 summaries in Excel **drawing text boxes** | S4 reference summaries |

### Cluster-label mapping from the inspected workbooks

| Risk group | S1 original label | S4 t-SNE-PSO label |
| --- | ---: | ---: |
| low risk | 0 | 1 |
| average risk | 1 | 0 |
| high risk | 2 | 2 |

**Important:** The original per-cluster summaries appear in Excel shape/drawing objects. Reading only the `Clinical_Report` table column does not extract those summaries. The correct JSON construction should extract the overall text boxes and preserve summary wording.

## Source-data integrity

- Publish data generation or preprocessing code when redistribution rights permit, and specify the random seed and parameters.
- Preserve S1 and S4 original summary texts separately from newly generated RAG summaries.
- Keep immutable snapshots of evidence and generated outputs for each run; model/API reruns may differ.
- Do not commit private/real patient records or credentials.
