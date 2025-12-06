"""
Disease Label Generator
=======================
Generates patient disease labels by comparing test values against
clinical thresholds parsed from the knowledge base.

This module:
1. Parses Normal_Range to extract min/max thresholds
2. Handles various formats (X < Y, A to B, gender-specific, etc.)
3. Labels patients based on abnormal test values
4. Generates multi-label disease vectors for HGT training
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThresholdRange:
    """Represents a test's normal range thresholds."""
    test_name: str
    disease: str
    organ: str
    low: Optional[float] = None      # Below this = abnormal (Under)
    high: Optional[float] = None     # Above this = abnormal (Over)
    unit: Optional[str] = None
    raw_range: str = ""              # Original string for debugging


def parse_normal_range(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse various Normal_Range formats to extract (low, high) thresholds.
    
    Supported formats:
    - "0.4 < X < 4.0"           → (0.4, 4.0)
    - "X < 100"                 → (None, 100)
    - "X > 90"                  → (90, None)
    - "39 to 46"                → (39, 46)
    - "39 - 46" or "39 – 46"    → (39, 46)
    - "< 150 mg/dL"             → (None, 150)
    - "> 90 mL/min"             → (90, None)
    - "4,000 – 11,000 /µL"      → (4000, 11000)
    - "Men: 13-17 | Women: 12-15" → averaged (12.5, 16)
    - "Males:4.32 to 5.72, Females:3.90 to 5.03" → averaged
    
    Returns:
        Tuple of (low_threshold, high_threshold), either can be None
    """
    if pd.isna(range_str) or not range_str or range_str.strip() == '':
        return None, None
    
    range_str = str(range_str).strip()
    
    # Handle "No fixed range" or similar
    if 'no fixed' in range_str.lower() or 'varies' in range_str.lower():
        return None, None
    
    # Check for gender-specific ranges first
    if _has_gender_specific(range_str):
        return _parse_gender_averaged(range_str)
    
    # Remove units and clean string
    cleaned = _clean_range_string(range_str)
    
    # Try different parsing patterns
    parsers = [
        _parse_inequality_range,      # "0.4 < X < 4.0"
        _parse_single_inequality,     # "X < 100" or "< 150"
        _parse_to_range,              # "39 to 46"
        _parse_dash_range,            # "39 - 46" or "39 – 46"
        _parse_concatenated_numbers,  # "1317" → 13, 17
        _parse_simple_number,         # Just a number
    ]
    
    for parser in parsers:
        result = parser(cleaned)
        if result != (None, None):
            return result
    
    return None, None


def _has_gender_specific(s: str) -> bool:
    """Check if string contains gender-specific ranges."""
    s_lower = s.lower()
    return any(keyword in s_lower for keyword in ['men', 'women', 'male', 'female'])


def _parse_gender_averaged(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse gender-specific ranges and return averaged thresholds.
    
    Examples:
    - "Men: 13-17 g/dL | Women: 12-15 g/dL" → (12.5, 16.0)
    - "Males:4.32 to 5.72, Females:3.90 to 5.03" → (4.11, 5.375)
    - "Men: 1317 g/dL | Women: 1215 g/dL" → parse concatenated numbers
    """
    all_lows = []
    all_highs = []
    
    # Split by common delimiters
    parts = re.split(r'[|,\n\r]', range_str)
    
    for part in parts:
        if not part.strip():
            continue
        
        # Remove gender prefix
        cleaned = re.sub(r'(men|women|male|female)[s]?\s*[:：]?\s*', '', part, flags=re.IGNORECASE)
        cleaned = _clean_range_string(cleaned)
        
        # Try to parse this part
        low, high = None, None
        for parser in [_parse_dash_range, _parse_to_range, _parse_inequality_range, _parse_concatenated_numbers]:
            low, high = parser(cleaned)
            if low is not None or high is not None:
                break
        
        if low is not None:
            all_lows.append(low)
        if high is not None:
            all_highs.append(high)
    
    # Average the values
    avg_low = sum(all_lows) / len(all_lows) if all_lows else None
    avg_high = sum(all_highs) / len(all_highs) if all_highs else None
    
    return avg_low, avg_high


def _parse_concatenated_numbers(s: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse concatenated numbers like "1317" which should be "13-17".
    This handles cases where dash was lost: "Men: 1317" → (13, 17)
    
    Heuristic: If we have a 4-digit number, try splitting in middle.
    """
    # Look for 4-digit numbers that could be two 2-digit numbers
    match = re.search(r'\b(\d{4})\b', s)
    if match:
        num = match.group(1)
        # Split in the middle: 1317 → 13, 17
        low = float(num[:2])
        high = float(num[2:])
        if low < high and low > 0:
            return low, high
    
    # Look for 6-digit numbers (3+3 split): 150000450000 unlikely, skip
    # Look for numbers like "0.61.1" (concatenated decimals)
    match = re.search(r'(\d+\.\d+)(\d+\.\d+)', s)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    return None, None


def _clean_range_string(s: str) -> str:
    """Remove units, special characters, and clean the string."""
    # Remove common units
    units = [
        r'mg/dL', r'g/dL', r'mU/L', r'mmol/L', r'mL/min/1\.73m²', r'mL/min',
        r'/µL', r'/uL', r'×10\?/L', r'×10\^9/L', r'fL', r'pg', r'%',
        r'femtoliters', r'# per uL', r'IU/L', r'U/L', r'ng/mL', r'pg/mL',
        r'µmol/L', r'umol/L', r'mEq/L', r'sec', r'seconds',
    ]
    for unit in units:
        s = re.sub(unit, '', s, flags=re.IGNORECASE)
    
    # Remove commas from numbers (4,000 → 4000)
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    
    # Normalize all types of dashes and special chars to regular dash
    # \x96 is en-dash in Windows-1252/latin-1 encoding
    s = s.replace('–', '-').replace('—', '-').replace('−', '-').replace('\x96', '-')
    
    # Handle space-separated ranges (80  100 → 80-100)
    # Also handle "80 - 100" with spaces around dash
    s = re.sub(r'(\d+\.?\d*)\s+[-]?\s*(\d+\.?\d*)', r'\1-\2', s)
    
    return s.strip()


def _parse_inequality_range(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse "0.4 < X < 4.0" format."""
    # Pattern: number < X < number
    match = re.search(r'([\d.]+)\s*<\s*[xX]\s*<\s*([\d.]+)', s)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def _parse_single_inequality(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse "X < 100", "< 150", "X > 90", "> 90" formats."""
    # X < number or < number (upper bound only)
    match = re.search(r'[xX]?\s*<\s*([\d.]+)', s)
    if match:
        return None, float(match.group(1))
    
    # X > number or > number (lower bound only)
    match = re.search(r'[xX]?\s*>\s*([\d.]+)', s)
    if match:
        return float(match.group(1)), None
    
    return None, None


def _parse_to_range(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse "39 to 46" format."""
    match = re.search(r'([\d.]+)\s+to\s+([\d.]+)', s, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def _parse_dash_range(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse "39 - 46" or "39-46" format."""
    # Be careful not to match negative numbers
    match = re.search(r'([\d.]+)\s*[-]\s*([\d.]+)', s)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        # Sanity check: low should be less than high
        if low <= high:
            return low, high
        else:
            return high, low
    return None, None


def _parse_simple_number(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse single number (treated as upper bound)."""
    match = re.match(r'^([\d.]+)$', s.strip())
    if match:
        return None, float(match.group(1))
    return None, None


# ==============================================================================
# KNOWLEDGE BASE LOADING
# ==============================================================================

def load_knowledge_base_with_thresholds(
    kb_path: str,
    encoding: str = 'latin-1'
) -> Dict[str, ThresholdRange]:
    """
    Load knowledge base and parse all thresholds.
    
    Args:
        kb_path: Path to enhanced_symptom_connectivity_analysis CSV
        encoding: File encoding (latin-1 for special characters)
        
    Returns:
        Dict mapping TestName to ThresholdRange object
    """
    df = pd.read_csv(kb_path, encoding=encoding, skiprows=1)
    df.columns = ['TestName', 'Unique_Patient_Count', 'Most_Relevant_Disease', 
                  'Target_Organ', 'Units', 'Under', 'Normal_Range', 'Over']
    
    thresholds = {}
    parse_stats = {'success': 0, 'partial': 0, 'failed': 0}
    
    for _, row in df.iterrows():
        test_name = row['TestName']
        
        # First try explicit Under/Over if available
        low, high = None, None
        
        if pd.notna(row['Under']) and row['Under'] != '-':
            under_low, under_high = parse_normal_range(str(row['Under']))
            if under_high is not None:
                low = under_high  # "X < 0.4" means 0.4 is the low threshold
        
        if pd.notna(row['Over']) and row['Over'] != '-':
            over_low, over_high = parse_normal_range(str(row['Over']))
            if over_low is not None:
                high = over_low  # "X > 4.0" means 4.0 is the high threshold
        
        # If Under/Over not available, parse Normal_Range
        if low is None and high is None:
            low, high = parse_normal_range(row['Normal_Range'])
        
        # Track stats
        if low is not None and high is not None:
            parse_stats['success'] += 1
        elif low is not None or high is not None:
            parse_stats['partial'] += 1
        else:
            parse_stats['failed'] += 1
        
        thresholds[test_name] = ThresholdRange(
            test_name=test_name,
            disease=row['Most_Relevant_Disease'],
            organ=row['Target_Organ'],
            low=low,
            high=high,
            unit=row['Units'] if pd.notna(row['Units']) else None,
            raw_range=str(row['Normal_Range'])
        )
    
    print(f"\n[Threshold Parsing Stats]")
    print(f"  Full range (low & high): {parse_stats['success']}")
    print(f"  Partial (low or high):   {parse_stats['partial']}")
    print(f"  Failed to parse:         {parse_stats['failed']}")
    print(f"  Total:                   {len(thresholds)}")
    
    return thresholds


# ==============================================================================
# PATIENT LABEL GENERATION
# ==============================================================================

@dataclass
class PatientLabels:
    """Container for patient disease labels."""
    patient_ids: List[int]
    disease_names: List[str]
    label_matrix: np.ndarray        # [num_patients, num_diseases] binary matrix
    patient_to_idx: Dict[int, int]  # PatientID -> row index
    disease_to_idx: Dict[str, int]  # Disease name -> column index
    abnormal_tests: Dict[int, List[Tuple[str, str, float]]]  # PatientID -> [(test, disease, value)]


def generate_patient_disease_labels(
    patient_reports_path: str,
    kb_path: str,
    kb_encoding: str = 'latin-1'
) -> PatientLabels:
    """
    Generate disease labels for all patients based on abnormal test values.
    
    Args:
        patient_reports_path: Path to patient_reports.csv
        kb_path: Path to knowledge base CSV
        kb_encoding: Encoding for knowledge base file
        
    Returns:
        PatientLabels object with multi-label disease matrix
    """
    # Load data
    print("[1/4] Loading patient reports...")
    patient_df = pd.read_csv(patient_reports_path)
    
    print("[2/4] Loading and parsing knowledge base thresholds...")
    thresholds = load_knowledge_base_with_thresholds(kb_path, kb_encoding)
    
    # Get unique patients and diseases
    patient_ids = sorted(patient_df['PatientID'].unique().tolist())
    disease_names = sorted(set(t.disease for t in thresholds.values()))
    
    print(f"\n[3/4] Processing {len(patient_ids)} patients, {len(disease_names)} diseases...")
    
    # Create mappings
    patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(disease_names)}
    
    # Initialize label matrix (0 = no risk, 1 = at risk)
    label_matrix = np.zeros((len(patient_ids), len(disease_names)), dtype=np.float32)
    
    # Track abnormal tests for explainability
    abnormal_tests = {pid: [] for pid in patient_ids}
    
    # Stats
    stats = {
        'tests_processed': 0,
        'tests_matched': 0,
        'abnormal_found': 0,
        'no_threshold': 0
    }
    
    # Process each patient's tests
    for _, row in patient_df.iterrows():
        pid = row['PatientID']
        test_name = row['TestName']
        test_value = row['TestValue']
        
        stats['tests_processed'] += 1
        
        # Skip if test not in knowledge base
        if test_name not in thresholds:
            continue
        
        stats['tests_matched'] += 1
        thresh = thresholds[test_name]
        
        # Skip if no thresholds parsed
        if thresh.low is None and thresh.high is None:
            stats['no_threshold'] += 1
            continue
        
        # Check if value is abnormal
        try:
            value = float(test_value)
        except (ValueError, TypeError):
            continue
        
        is_abnormal = False
        
        # Check if below normal (Under)
        if thresh.low is not None and value < thresh.low:
            is_abnormal = True
        
        # Check if above normal (Over)
        if thresh.high is not None and value > thresh.high:
            is_abnormal = True
        
        # Label patient with disease risk
        if is_abnormal:
            stats['abnormal_found'] += 1
            p_idx = patient_to_idx[pid]
            d_idx = disease_to_idx[thresh.disease]
            label_matrix[p_idx, d_idx] = 1.0
            abnormal_tests[pid].append((test_name, thresh.disease, value))
    
    print(f"\n[4/4] Label generation complete!")
    print(f"  Tests processed:  {stats['tests_processed']}")
    print(f"  Tests matched:    {stats['tests_matched']}")
    print(f"  No threshold:     {stats['no_threshold']}")
    print(f"  Abnormal values:  {stats['abnormal_found']}")
    
    # Label distribution
    patients_with_risk = (label_matrix.sum(axis=1) > 0).sum()
    avg_diseases = label_matrix.sum(axis=1).mean()
    print(f"\n[Label Distribution]")
    print(f"  Patients with >= 1 disease risk: {patients_with_risk} ({100*patients_with_risk/len(patient_ids):.1f}%)")
    print(f"  Average diseases per patient:    {avg_diseases:.2f}")
    
    # Disease distribution
    print(f"\n[Top 10 Diseases by Patient Count]")
    disease_counts = label_matrix.sum(axis=0)
    top_diseases = sorted(zip(disease_names, disease_counts), key=lambda x: -x[1])[:10]
    for disease, count in top_diseases:
        print(f"  {disease}: {int(count)} patients")
    
    return PatientLabels(
        patient_ids=patient_ids,
        disease_names=disease_names,
        label_matrix=label_matrix,
        patient_to_idx=patient_to_idx,
        disease_to_idx=disease_to_idx,
        abnormal_tests=abnormal_tests
    )


def save_labels(labels: PatientLabels, output_path: str):
    """
    Save generated labels to CSV for inspection.
    
    Saves ALL patients (both with and without disease risk) in wide format:
    PatientID, Disease1, Disease2, Disease3, ... where each column is 0/1
    """
    rows = []
    for patient_id in labels.patient_ids:
        p_idx = labels.patient_to_idx[patient_id]
        row = {'PatientID': patient_id}
        
        # Add label for each disease
        for disease, d_idx in labels.disease_to_idx.items():
            row[disease] = int(labels.label_matrix[p_idx, d_idx])
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    # Print statistics
    num_with_risk = (df.iloc[:, 1:].sum(axis=1) > 0).sum()
    num_no_risk = len(df) - num_with_risk
    print(f"\nSaved {len(df)} patient labels to {output_path}")
    print(f"  With disease risk:    {num_with_risk} patients")
    print(f"  Without disease risk: {num_no_risk} patients")
    print(f"  Format: PatientID, {len(labels.disease_names)} disease columns (0/1)")



def print_patient_summary(labels: PatientLabels, patient_id: int):
    """Print detailed summary for a specific patient."""
    if patient_id not in labels.patient_to_idx:
        print(f"Patient {patient_id} not found")
        return
    
    p_idx = labels.patient_to_idx[patient_id]
    diseases = [d for d, idx in labels.disease_to_idx.items() 
                if labels.label_matrix[p_idx, idx] > 0]
    
    print(f"\n{'='*60}")
    print(f"Patient {patient_id} Summary")
    print(f"{'='*60}")
    print(f"Disease Risks: {len(diseases)}")
    for d in diseases:
        print(f"  - {d}")
    
    print(f"\nAbnormal Tests:")
    for test, disease, value in labels.abnormal_tests.get(patient_id, []):
        print(f"  - {test}: {value} → {disease}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_generate_labels(
    emr_path: str = 'data/patient_reports.csv',
    kb_path: str = 'data/enhanced_symptom_connectivity_analysis(Sheet1).csv'
) -> PatientLabels:
    """
    Quick function to generate labels with default paths.
    
    Usage:
        from graph.labels import quick_generate_labels
        labels = quick_generate_labels()
    """
    return generate_patient_disease_labels(emr_path, kb_path)


if __name__ == '__main__':
    # Test the label generator
    labels = quick_generate_labels()
    
    # Save to file
    save_labels(labels, 'data/patient_disease_labels.csv')
    
    # Show example patient
    if labels.patient_ids:
        print_patient_summary(labels, labels.patient_ids[0])
