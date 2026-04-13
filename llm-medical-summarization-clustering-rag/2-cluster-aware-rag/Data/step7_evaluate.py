#!/usr/bin/env python3
import json, re
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROFILE_TABLE = "cluster_profile_table.csv"

profiles = pd.read_csv(PROFILE_TABLE)

BP_FAMILY = ["pct_BP_elevated_or_higher", "pct_BP_stage2"]
MET_FAMILY = ["pct_BMI_overweight", "pct_BMI_obese", "pct_Glucose_prediabetes", "pct_Glucose_diabetes", "pct_Chol_borderline", "pct_Chol_high"]

INDICATOR_KEYWORDS = {
    "pct_BP_elevated_or_higher": ["blood pressure", "bp", "hypertension", "pre-hypertens", "systolic", "diastolic", "elevated bp"],
    "pct_BP_stage2": ["stage 2", "severe hypertension", "very high blood pressure", "140/90"],
    "pct_BMI_overweight": ["overweight", "bmi", "weight"],
    "pct_BMI_obese": ["obesity", "obese", "bmi"],
    "pct_Glucose_prediabetes": ["glucose", "prediabet", "blood sugar", "elevated sugar"],
    "pct_Glucose_diabetes": ["diabetes", "diabetic", "hyperglyc", "very high glucose"],
    "pct_Chol_borderline": ["cholesterol", "borderline cholesterol", "lipid"],
    "pct_Chol_high": ["high cholesterol", "hypercholesterol", "cholesterol", "lipid"]
}

ABSOLUTE_PATTERNS = [r"\ball patients\b", r"\bevery patient\b", r"\ball individuals\b", r"\bnone of the patients\b", r"\balways\b", r"\bnever\b"]
DIAGNOSIS_PATTERNS = [r"\bdiagnos(ed|is)\b", r"\bhas (?:a|an|the)\b.*\b(disease|disorder|condition)\b", r"\bchronic kidney disease\b", r"\bckd\b", r"\bmyocardial infarction\b", r"\bstroke\b"]
MEDICATION_PATTERNS = [r"\bprescribe\b", r"\bmedication\b", r"\bdrug\b", r"\bdosage\b", r"\bmetformin\b", r"\bstatin\b", r"\binsulin\b", r"\bantihypertensive\b"]

def parse_evidence_block(evidence_block: str):
    pattern = r"(\[(?:C|S\d+)\])"
    parts = re.split(pattern, evidence_block)
    items = []
    cur_id = None
    cur_text = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(pattern, p):
            if cur_id is not None:
                txt = re.sub(r"\s+", " ", "\n".join(cur_text)).strip()
                if txt:
                    items.append((cur_id, txt))
            cur_id = p
            cur_text = []
        else:
            cur_text.append(p.strip())
    if cur_id is not None:
        txt = re.sub(r"\s+", " ", "\n".join(cur_text)).strip()
        if txt:
            items.append((cur_id, txt))
    cleaned = []
    for eid, txt in items:
        txt = re.sub(r"^Cluster profile.*?:\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"^Patient snippet.*?:\s*", "", txt, flags=re.IGNORECASE)
        cleaned.append((eid, txt.strip()))
    return cleaned

def tfidf_cosine(a: str, b: str):
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = v.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0,0])

def extract_claims(summary_text: str, min_chars=20, max_claims=15):
    lines = [ln.strip() for ln in summary_text.splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.startswith(("-", "•", "*"))]
    text_no_cite = re.sub(r"\[(?:C|S\d+)\]", "", summary_text)
    text_no_cite = re.sub(r"\s+", " ", text_no_cite).strip()
    claims = []
    if bullet_lines:
        for ln in bullet_lines:
            ln = re.sub(r"^[-•*]\s*", "", ln)
            ln = re.sub(r"\[(?:C|S\d+)\]", "", ln)
            ln = re.sub(r"\s+", " ", ln).strip()
            if len(ln) >= min_chars:
                claims.append(ln)
    else:
        sents = re.split(r"(?<=[\.\!\?])\s+", text_no_cite)
        claims = [s.strip() for s in sents if len(s.strip()) >= min_chars]
    if len(claims) > max_claims:
        claims = claims[:max_claims]
    return claims

def minmax_norm(a, eps=1e-9):
    a = np.asarray(a).astype(float)
    return (a - a.min())/(a.max()-a.min()+eps)

def support_check_claims(claims, evidence_items, sim_threshold=0.22):
    e_ids = [eid for eid,_ in evidence_items]
    e_texts = [txt for _,txt in evidence_items]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = v.fit_transform(e_texts + claims)
    E = X[:len(e_texts)]
    C = X[len(e_texts):]
    sims = cosine_similarity(C, E)
    results = []
    for i, claim in enumerate(claims):
        j = int(np.argmax(sims[i]))
        best_sim = float(sims[i,j])
        results.append({
            "claim": claim,
            "supported": best_sim >= sim_threshold,
            "best_evidence_id": e_ids[j],
            "best_similarity": best_sim
        })
    return results

def pick_must_mentions(cluster_name):
    row = profiles.loc[profiles["risk_group"]==cluster_name].iloc[0]
    bp_best = max(BP_FAMILY, key=lambda c: row[c])
    met_best = max(MET_FAMILY, key=lambda c: row[c])
    return [bp_best, met_best]

def completeness_score(summary_text, must_inds):
    s = summary_text.lower()
    hits = 0
    for ind in must_inds:
        kws = INDICATOR_KEYWORDS.get(ind, [])
        if any(kw in s for kw in kws):
            hits += 1
    return hits / max(1, len(must_inds)), hits

def count_patterns(text, patterns):
    t = text.lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, t))
    return total

def evaluate_one(item, n_claims=12):
    cluster = item["cluster"]
    rag = item["rag_summary"]
    ref = item["reference_s1"]
    ev  = item["evidence_block"]
    sim = tfidf_cosine(rag, ref)

    evidence_items = parse_evidence_block(ev)
    claims = extract_claims(rag, max_claims=n_claims)
    cr = support_check_claims(claims, evidence_items)
    supported = sum(1 for r in cr if r["supported"])
    total = len(cr)
    supported_rate = (supported/total*100) if total else 0.0
    unsupported_rate = 100-supported_rate if total else 0.0

    must = pick_must_mentions(cluster)
    comp, comp_hits = completeness_score(rag, must)

    abs_ct = count_patterns(rag, ABSOLUTE_PATTERNS)
    diag_ct = count_patterns(rag, DIAGNOSIS_PATTERNS)
    med_ct = count_patterns(rag, MEDICATION_PATTERNS)

    return {
        "cluster": cluster,
        "similarity_tfidf": sim,
        "supported_claim_rate_%": supported_rate,
        "unsupported_claim_rate_%": unsupported_rate,
        "n_claims_checked": total,
        "completeness_%": comp*100,
        "must_mentions": ";".join(must),
        "must_mentions_hit": comp_hits,
        "absolute_claim_flags": abs_ct,
        "diagnosis_like_flags": diag_ct,
        "medication_flags": med_ct
    }

def main():
    with open("step7_eval_input.json","r",encoding="utf-8") as f:
        items = json.load(f)
    rows = [evaluate_one(it) for it in items]
    out = pd.DataFrame(rows)
    out.to_csv("step7_evaluation_results.csv", index=False)
    print(out)

if __name__ == "__main__":
    main()
