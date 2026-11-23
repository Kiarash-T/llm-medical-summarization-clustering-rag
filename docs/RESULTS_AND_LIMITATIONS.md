# Published outcomes, independent runs, and limitations

## Published Paper 1

Paper 1 uses S1 (true-label grouping) as the **reference summary**. S2, S3 and S4 refer to t-SNE, UMAP and t-SNE-PSO + K-Means groupings, respectively. Their generated summaries were compared for textual similarity against S1; the paper also reports clustering metrics such as Accuracy, NMI, Silhouette Coefficient and Davies-Bouldin Index.

This Paper 1 similarity experiment is **not** identical to the Paper 2 claim-support evaluation. Do not reuse Paper 2 scores as Paper 1 similarity values.

## Published Paper 2 (Tables 1–3)

| Metric | S1 baseline | S4 baseline | Reported RAG |
| --- | ---: | ---: | ---: |
| Macro supported-claim rate | 11.1% | 8.3% | 33.3% |
| Macro unsupported-claim rate | 88.9% | 91.7% | 66.7% |
| Macro indicator completeness | 83.3% | 83.3% | 100.0% |

The published paper selects **α = 0.25** (25% normalized TF-IDF and 75% embedding similarity). See [the official PMLR publication](https://proceedings.mlr.press/v318/torabizadeh26a.html).

## What an independent run can and cannot reproduce

The inspected notebook performs a five-value α sweep, using three risk-group summaries per α. Different repetitions can select different optimal α, even with fixed retrieved evidence. The associated causes include:

1. Hosted generation variability (temperature zero is not a hard reproducibility guarantee).
2. TF-IDF lexical support threshold (`>= 0.22`), which is sensitive to wording, numbers and text segmentation.
3. Evaluator implementation details: first 12 claims, detection of bullet points, rule-based absolute/medication/diagnosis flags.
4. Input-snapshot discrepancies, such as 149 available snippets rather than the code comment's expected 150.

**Interpretation:** A higher automatic supported-claim rate means better performance *under that particular lexical proxy*. It is not proof that a clinician would judge more claims correct. Exact agreement with the paper requires archiving the original input, model outputs, evaluator version, and evaluation settings.

## Baseline/RAG setup to document explicitly

- **S1 baseline:** direct summary from true-label groupings, without RAG.
- **S4 baseline:** direct summary from t-SNE-PSO groupings, without RAG.
- **Inspected RAG implementation:** one set of generated summary/evidence per risk group is evaluated against two different reference summaries (S1 and S4). Because the RAG summaries are the same, supported claims, indicator completeness, and safety flags may match across both RAG comparison files. Only reference-dependent similarity may differ.
- Two genuinely independent *S1-RAG* and *S4-RAG* conditions require separately constructed, grouping-specific evidence/generation inputs and outputs; they must not be inferred from file names alone.

## Main limitations

- Small, synthetic cohort; no prospective or external clinical validation.
- Lexical claim support is not semantic entailment and does not verify numerical consistency or contradictions.
- Completeness uses keyword matching for selected indicators, not clinician-rated clinical sufficiency.
- Safety checks detect finite regex patterns; alternative wording may evade them.
- Profile documents can be retrieved again among `[S...]` items in the inspected code because profile and snippet docs share the ranking candidate pool. Filtering by `doc_type='snippet'` would change the retrieval behavior and should be reported as a new code revision.
- The method must not be used as clinical advice.
