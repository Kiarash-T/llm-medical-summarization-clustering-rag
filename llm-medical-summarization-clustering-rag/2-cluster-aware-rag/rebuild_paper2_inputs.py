#!/usr/bin/env python3
"""Rebuild the three Paper 2 evidence files from the original patient workbook.

Tested against the 149-row input workbook and the preserved original output
files supplied by the authors. Requires only Python standard library and NumPy.
No language-model/API calls or randomness are involved.

Usage:
    python rebuild_paper2_inputs.py --input patient_dataset_with_reports_2_x.xlsx --outdir Data

The original source code is unavailable. This is a reconstructed implementation,
verified against the original file contents, not a claim to have recovered it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
INPUT_COLUMNS = [
    'Age', 'BMI', 'Glucose', 'Heart Rate', 'BP_Systolic', 'BP_Diastolic',
    'Cholesterol', 'Hypertension_Stage', 'Clinical_Report'
]
FEATURES = INPUT_COLUMNS[:7]
LABEL_TO_RISK = {0: 'low_risk', 1: 'average_risk', 2: 'high_risk'}
RISK_ORDER = ['average_risk', 'high_risk', 'low_risk']
PCT_FIELDS = [
    'pct_BP_elevated_or_higher', 'pct_BP_stage2',
    'pct_BMI_overweight', 'pct_BMI_obese',
    'pct_Glucose_prediabetes', 'pct_Glucose_diabetes',
    'pct_Chol_borderline', 'pct_Chol_high'
]


def read_original_excel(path: Path) -> list[dict]:
    """Read the workbook with the standard library, preserving stored numbers.

    This targeted OOXML reader handles shared-string and numeric cells from the
    supplied .xlsx, without requiring Excel, pandas, or openpyxl.
    """
    with ZipFile(path) as zf:
        members = set(zf.namelist())
        sheet_name = 'xl/worksheets/sheet1.xml'
        if sheet_name not in members:
            raise ValueError('Expected input data in the first Excel worksheet.')
        strings = []
        if 'xl/sharedStrings.xml' in members:
            strings_root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
            for item in strings_root.findall(f'{NS}si'):
                strings.append(''.join(t.text or '' for t in item.iter(f'{NS}t')))
        root = ET.fromstring(zf.read(sheet_name))

    matrix = []
    sheet_data = root.find(f'{NS}sheetData')
    if sheet_data is None:
        raise ValueError('No worksheet data found.')
    for row in sheet_data.findall(f'{NS}row'):
        values = {}
        for cell in row.findall(f'{NS}c'):
            addr = cell.get('r', '')
            letters = re.match(r'^[A-Z]+', addr)
            if letters is None:
                continue
            col = 0
            for char in letters.group():
                col = 26 * col + (ord(char) - 64)
            col -= 1
            kind = cell.get('t')
            node = cell.find(f'{NS}v')
            if kind == 'inlineStr':
                v = ''.join(t.text or '' for t in cell.iter(f'{NS}t'))
            elif node is None or node.text is None:
                v = None
            elif kind == 's':
                v = strings[int(node.text)]
            elif kind in ('str', 'e'):
                v = node.text
            else:
                v = float(node.text)
            values[col] = v
        if values:
            matrix.append(values)

    if not matrix:
        raise ValueError('Empty worksheet.')
    header = [matrix[0].get(i) for i in range(len(INPUT_COLUMNS))]
    if header != INPUT_COLUMNS:
        raise ValueError(f'Unexpected Excel columns: {header!r}')

    patients = []
    for source_index, row in enumerate(matrix[1:], start=1):
        values = [row.get(i) for i in range(len(INPUT_COLUMNS))]
        if all(v is None for v in values):
            continue
        if any(v is None for v in values):
            raise ValueError(f'Incomplete patient row {source_index + 1}.')
        record = dict(zip(INPUT_COLUMNS, values))
        label_value = record['Hypertension_Stage']
        if not float(label_value).is_integer() or int(label_value) not in LABEL_TO_RISK:
            raise ValueError(f'Unexpected risk label {label_value} in row {source_index + 1}.')
        record['risk_group'] = LABEL_TO_RISK[int(label_value)]
        record['patient_id'] = source_index
        patients.append(record)
    return patients


def build_profile_rows(patients: list[dict]) -> list[dict]:
    result = []
    for risk in RISK_ORDER:
        group = [r for r in patients if r['risk_group'] == risk]
        if not group:
            raise ValueError(f'Empty group: {risk}')
        n = len(group)
        profile = {'risk_group': risk, 'n': n}
        for feat in FEATURES:
            data = np.array([float(r[feat]) for r in group], dtype=np.float64)
            profile[f'{feat}_mean'] = float(np.mean(data))
            profile[f'{feat}_min'] = float(np.min(data))
            profile[f'{feat}_max'] = float(np.max(data))

        count = lambda pred: sum(bool(pred(r)) for r in group)
        # Use (count / n) * 100.0 to preserve the original floating-point values.
        thresholds = [
            lambda r: r['BP_Systolic'] >= 130 or r['BP_Diastolic'] >= 80,
            lambda r: r['BP_Systolic'] >= 140 or r['BP_Diastolic'] >= 90,
            lambda r: r['BMI'] >= 25,
            lambda r: r['BMI'] >= 30,
            lambda r: r['Glucose'] >= 100,
            lambda r: r['Glucose'] >= 126,
            lambda r: r['Cholesterol'] >= 200,
            lambda r: r['Cholesterol'] >= 240,
        ]
        for name, rule in zip(PCT_FIELDS, thresholds):
            profile[name] = (count(rule) / n) * 100.0
        result.append(profile)
    return result


def save_profile_table(profiles: list[dict], path: Path) -> None:
    columns = ['risk_group', 'n']
    for feature in FEATURES:
        for s in ['mean', 'min', 'max']:
            columns.append(f'{feature}_{s}')
    columns += PCT_FIELDS
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(profiles)


def save_profile_markdown(profiles: list[dict], path: Path) -> None:
    chunks = []
    for p in profiles:
        block = '\n'.join([
            f"CLUSTER PROFILE — {p['risk_group']} (n={p['n']})",
            f"Age: mean {p['Age_mean']:.1f}, range [{p['Age_min']:.1f}, {p['Age_max']:.1f}]",
            f"Blood pressure: mean {p['BP_Systolic_mean']:.1f}/{p['BP_Diastolic_mean']:.1f} mmHg; %>=130/80: {p['pct_BP_elevated_or_higher']:.1f}%; %>=140/90: {p['pct_BP_stage2']:.1f}%",
            f"BMI: mean {p['BMI_mean']:.1f}; %>=25: {p['pct_BMI_overweight']:.1f}%; %>=30: {p['pct_BMI_obese']:.1f}%",
            f"Glucose: mean {p['Glucose_mean']:.1f}; %>=100: {p['pct_Glucose_prediabetes']:.1f}%; %>=126: {p['pct_Glucose_diabetes']:.1f}%",
            f"Cholesterol: mean {p['Cholesterol_mean']:.1f}; %>=200: {p['pct_Chol_borderline']:.1f}%; %>=240: {p['pct_Chol_high']:.1f}%",
        ])
        chunks.append(f'```text\n{block}\n```')
    path.write_text('# Cluster profile documents (for RAG)\n\n' + '\n\n'.join(chunks) + '\n\n', encoding='utf-8')


def save_snippets(patients: list[dict], path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        for patient in patients:
            item = {
                'snippet_id': f"{patient['risk_group']}_p{patient['patient_id']}",
                'risk_group': patient['risk_group'],
                'patient_id': patient['patient_id'],
                'snippet_text': patient['Clinical_Report'],
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('patient_dataset_with_reports_2_x.xlsx'))
    parser.add_argument('--outdir', type=Path, default=Path('.'))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    patients = read_original_excel(args.input)
    profiles = build_profile_rows(patients)
    save_profile_table(profiles, args.outdir / 'cluster_profile_table.csv')
    save_profile_markdown(profiles, args.outdir / 'cluster_profile_docs.md')
    save_snippets(patients, args.outdir / 'snippet_bank_reports.jsonl')
    print(f'Regenerated three Paper 2 files from {len(patients)} patient rows:')
    for profile in profiles:
        print(f"  {profile['risk_group']}: {profile['n']} patients")
    print(f'Output directory: {args.outdir.resolve()}')


if __name__ == '__main__':
    main()
