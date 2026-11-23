# Reproducing the two papers

This document distinguishes **reconstructing the pipeline**, **recomputing saved metrics**, and **independently regenerating LLM text**. These are not interchangeable levels of reproducibility.

## 1. Access and prerequisites

1. Clone the repository. Paper 1 scripts are under `llm-medical-summarization-clustering-rag/1-clustering-llm-baseline/`; Paper 2 notebook and `Data/` folder are under `llm-medical-summarization-clustering-rag/2-cluster-aware-rag/`. The GitHub repository root has a separate README; preserve the nested directory layout unless you update the paths in this guide.
2. Use Python 3.10+ and install the packages described in `requirements.txt`. The package set is a starting point based on the inspected Paper 2 code; freeze and publish the actual versions used in validated runs.
3. For Paper 2, configure `OPENAI_API_KEY` in Colab Secrets or a local environment variable. The model is accessed remotely and may incur fees.
4. Keep a copy of the synthetic input data and the original Paper 1 S1 and S4 cluster-summary Excel workbooks. Do not use the patient-level `Clinical_Report` rows as substitutes for the overall cluster summaries contained in the workbook drawing text boxes.
5. Keep a per-run directory with source commit, date, Python/package versions, model ID, parameters, and output files.

### Environment setup

```bash
git clone https://github.com/Kiarash-T/llm-medical-summarization-clustering-rag.git
cd llm-medical-summarization-clustering-rag
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
python -m pip install -r requirements.txt
jupyter lab
```

Colab: open a notebook, enable the Secrets panel, add `OPENAI_API_KEY`, and make the secret available to the notebook. Do not paste API keys into Python cell source.

### Suggested environment capture

```bash
mkdir -p results/run-notes
python --version > results/run-notes/python-version.txt
python -m pip freeze > results/run-notes/pip-freeze.txt
```

## 2. Paper 1: grouping and direct summarization

**Published methodology**

| Configuration | Group membership | Summary source |
| --- | --- | --- |
| S1 | Original risk labels / assumed known groups | GPT-generated direct cluster summary; reference for summary similarity |
| S2 | t-SNE + K-Means | GPT cluster summary |
| S3 | UMAP + K-Means | GPT cluster summary |
| S4 | t-SNE-PSO + K-Means | GPT cluster summary |

To reproduce, run the corresponding Paper 1 clustering scripts (`t-SNE.py`, `UMAP.py`, `t-SNE-PSO.py` and `Clustering_evaluation.py`) on the supplied synthetic patient dataset, record preprocessing, random seeds, embedding dimensionality, K-Means parameters and LLM configuration, and save each per-cluster summary. Compare S2/S3/S4 text similarity to the S1 reference using the **same similarity method specified in Paper 1**. Do not substitute the Paper 2 TF-IDF metric for Paper 1's text-similarity API without explicitly labeling it as a new experiment.

The Paper 1 folder includes the four listed scripts, a synthetic patient CSV and a `Results on Hypertension Dataset/` folder. **Before claiming full reproduction:** inspect the scripts and their input/output file paths, and provide tested run instructions for each algorithm and its summarization step.

## 3. Paper 2: cluster-aware RAG and alpha sweep

### Required inputs (filenames from the inspected notebook)

- `cluster_profile_docs.md`: three structured per-cluster profiles
- `snippet_bank_reports.jsonl`: patient report snippets with ID and risk group
- `cluster_profile_table.csv`: risk-group statistics used by the completeness evaluator
- `step7_eval_input_template.json`: three risk-group objects for Step 7
- `S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx`: original S1 overall summaries in drawing text boxes
- `S4_TSNE-PSO-patient_dataset_with_report_clustering_tsn-pso.xlsx`: original S4 overall summaries in drawing text boxes

Put these in the notebook's working directory, or update the paths in its setup cell. If they cannot be redistributed, document download/access conditions instead of promising a fully runnable checkout.

### Execute in notebook order

1. Load profiles and snippets; check actual counts and risk labels.
2. Fit TF-IDF index and encode snippets with `all-MiniLM-L6-v2`.
3. For each α in `0.0, 0.25, 0.5, 0.75, 1.0`, retrieve evidence with top-k = 8 for each of the three risk groups.
4. Generate an evidence-constrained summary with the configured API model and save the exact summary and evidence block.
5. Score the first 12 extracted claims per summary, using maximum TF-IDF cosine similarity to evidence and threshold 0.22; compute keyword coverage and regex-based safety flags.
6. Aggregate macro results by α; choose highest supported-claim rate, then highest completeness, then lowest mean absolute-claim flags.
7. **Current notebook behavior:** after choosing the winning α, the inspected implementation makes three additional API calls using `best_alpha`. These summaries may differ from those generated during the sweep, even with the same evidence and `temperature=0`. **Recommended reproducibility enhancement:** reuse the already generated winning-α summaries and evidence from `alpha_sweep_outputs.json` for downstream evaluation. If retaining the new calls, report that the final Step 7 measurements are a separate generation.
8. Read **overall cluster summaries from Excel drawing text boxes**, not individual patient report rows; construct S1- and S4-reference input JSON files.
9. Evaluate original S1 and S4 summaries as non-RAG baselines and compare each against the RAG evaluation. Save per-cluster results and macro results.
10. Archive the generated summaries, evidence IDs, chosen α, model configuration, input hashes, and evaluator output under a unique run folder.

### Output files from the inspected pipeline

| File | Meaning |
| --- | --- |
| `alpha_sweep_results.csv` | Individual α × risk-group metrics (15 rows when complete) |
| `alpha_sweep_macro.csv` | Macro scores for each α |
| `alpha_sweep_outputs.json` | Exact generated summaries and retrieved evidence blocks used during the sweep |
| `step7_eval_input_S1vsRAG.json` | RAG summary/evidence plus S1 reference text |
| `step7_eval_input_S4vsRAG.json` | Same RAG summary/evidence plus S4 reference text in the inspected implementation |
| `step7_evaluation_results_S1baseline.csv`, `..._S4baseline.csv` | Scores of the original non-RAG summaries |
| `step7_evaluation_results_S1vsRAG.csv`, `..._S4vsRAG.csv` | RAG scores relative to S1/S4 reference texts; supported-claim/coverage/safety can be identical because RAG text is shared |

### Verification checks

- Check exactly three risk groups in the template and expected risk-group keys in all files.
- Verify each α has all three clusters; do not select a winning α from an incomplete sweep.
- Compare evidence text and IDs for the selected alpha between `alpha_sweep_outputs.json` and the Step 7 JSON files.
- Verify no API keys exist in notebooks, outputs, or Git history.
- Record a **separate evaluation for independently generated S1-RAG versus S4-RAG** only if their grouping-specific profiles, snippets, generation inputs, and outputs were actually constructed separately.

## 4. Comparing with the publication

The published Paper 2 selected α = 0.25 and reported 33.3% macro supported claims for RAG, compared with 11.1% for S1 and 8.3% for S4. Do not overwrite new experimental outputs to make them match the paper. A different selected α or different rate can arise from API generation variation, score sensitivity to phrasing, input changes, or evaluator-version differences.

The threshold score measures **lexical similarity**, not clinical entailment. A claim such as “Mean BMI: 21.5” may be numerically correct yet score below the threshold when phrased differently from the evidence profile. Safety flags also depend on exact word patterns. Preserve claim-level logs if independent reviewers need to audit reliability.

## 5. Clean release checklist

- Release the **exact notebooks, source data (when permitted), and expected output artifacts** needed for every figure/table.
- Document which inputs are synthetic; avoid claims of validation on real clinical data.
- Capture environment lockfile, commit hash, random seeds, prompts and model snapshot when available.
- Check the original diagram export, authors/affiliations, GitHub links, DOI/PMLR citation and license.
- Confirm no raw credentials are present, including in Git history or notebook outputs.
