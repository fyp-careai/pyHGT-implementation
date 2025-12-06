"""
Graph Creation Pipeline
=======================
Main module for loading data and creating the unified heterogeneous graph
for HGT (Heterogeneous Graph Transformer) training.
"""

import torch
import pandas as pd
from typing import Dict, Tuple, Optional

from .encoders import generate_patient_embeddings
from .graph import (
    GraphData,
    construct_patient_symptom_graph,
    construct_symptom_organ_graph,
    construct_organ_disease_graph,
    construct_unified_graph,
    identify_patient_symptom_relations,
    identify_symptom_organ_relations,
    identify_organ_disease_relations
)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_emr_data(file_path: str) -> pd.DataFrame:
    """
    Load patient EMR data from CSV.
    
    Expected columns: PatientID, ReportDate, TestName, TestValue
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with EMR records
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=',',
            parse_dates=['ReportDate'],
            skipinitialspace=True
        )
        df.columns = df.columns.str.strip()
        print(f"[INFO] Loaded EMR data: {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return pd.DataFrame(columns=['PatientID', 'ReportDate', 'TestName', 'TestValue'])


def load_knowledge_base(file_path: str) -> pd.DataFrame:
    """
    Load symptom-organ-disease knowledge base from CSV.
    
    Expected columns: TestName, Target_Organ, Most_Relevant_Disease
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with knowledge base
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=',',
            encoding='latin-1',
            skipinitialspace=True
        )
        df.columns = df.columns.str.strip()
        
        if 'Unique_Patient_Count' in df.columns:
            df['Unique_Patient_Count'] = pd.to_numeric(
                df['Unique_Patient_Count'], errors='coerce'
            ).astype('Int64')
            
        print(f"[INFO] Loaded knowledge base: {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return pd.DataFrame(columns=['TestName', 'Target_Organ', 'Most_Relevant_Disease'])


# ==============================================================================
# ENTITY MAPS CREATION
# ==============================================================================

def create_entity_maps(
    emr_df: pd.DataFrame,
    kb_df: pd.DataFrame
) -> Tuple[dict, dict, dict, dict, list]:
    """
    Create mapping dictionaries for all entity types.
    
    Args:
        emr_df: EMR DataFrame
        kb_df: Knowledge base DataFrame
        
    Returns:
        Tuple of (concept_map, organ_map, disease_map, patient_idx_map, patient_ids)
    """
    # Extract unique entities
    unique_patients = emr_df['PatientID'].unique()
    unique_concepts = emr_df['TestName'].unique()
    unique_organs = kb_df['Target_Organ'].unique()
    unique_diseases = kb_df['Most_Relevant_Disease'].unique()
    
    # Create mappings
    concept_map = {name: i for i, name in enumerate(unique_concepts)}
    organ_map = {name: i for i, name in enumerate(unique_organs)}
    disease_map = {name: i for i, name in enumerate(unique_diseases)}
    patient_ids = list(unique_patients)
    patient_idx_map = {pid: i for i, pid in enumerate(patient_ids)}
    
    print(f"\n[INFO] Entity Maps Created:")
    print(f"  - Patients:  {len(patient_ids)}")
    print(f"  - Symptoms:  {len(concept_map)}")
    print(f"  - Organs:    {len(organ_map)}")
    print(f"  - Diseases:  {len(disease_map)}")
    
    return concept_map, organ_map, disease_map, patient_idx_map, patient_ids


# ==============================================================================
# MAIN GRAPH CREATION PIPELINE
# ==============================================================================

def create_hgt_graph(
    emr_file: str,
    kb_file: str,
    output_dim: int = 128,
    hidden_dim: int = 64,
    patient_labels: Optional[Dict[int, torch.Tensor]] = None
):
    """
    Main pipeline to create heterogeneous graph for HGT training.
    
    Args:
        emr_file: Path to EMR data CSV
        kb_file: Path to knowledge base CSV
        output_dim: Patient embedding dimension
        hidden_dim: TCN hidden dimension
        patient_labels: Optional dict of patient labels. If None, random labels generated.
        
    Returns:
        Tuple of (HeteroData graph, GraphData metadata, entity maps)
    """
    print("\n" + "="*60)
    print("HETEROGENEOUS GRAPH CREATION PIPELINE")
    print("="*60)
    
    # =========== STEP 1: Load Data ===========
    print("\n[STEP 1] Loading data...")
    emr_df = load_emr_data(emr_file)
    kb_df = load_knowledge_base(kb_file)
    
    if emr_df.empty or kb_df.empty:
        raise ValueError("Failed to load data files")
    
    # =========== STEP 2: Create Entity Maps ===========
    print("\n[STEP 2] Creating entity maps...")
    concept_map, organ_map, disease_map, patient_idx_map, patient_ids = create_entity_maps(emr_df, kb_df)
    
    input_dim = len(concept_map)
    num_diseases = len(disease_map)
    
    # =========== STEP 3: Generate Patient Embeddings ===========
    print("\n[STEP 3] Generating patient embeddings...")
    patient_features, final_patient_ids, final_patient_idx_map = generate_patient_embeddings(
        patient_data=emr_df,
        concept_map=concept_map,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim
    )
    
    # =========== STEP 4: Identify Relations ===========
    print("\n[STEP 4] Identifying relations...")
    ps_relations = identify_patient_symptom_relations(emr_df)
    so_relations = identify_symptom_organ_relations(kb_df)
    od_relations = identify_organ_disease_relations(kb_df)
    
    # =========== STEP 5: Build Bipartite Graphs ===========
    print("\n[STEP 5] Building bipartite graphs...")
    
    ps_graph = construct_patient_symptom_graph(
        final_patient_ids,
        patient_features,
        final_patient_idx_map,
        concept_map,
        ps_relations
    )
    
    so_graph = construct_symptom_organ_graph(
        concept_map,
        organ_map,
        so_relations
    )
    
    od_graph = construct_organ_disease_graph(
        disease_map,
        organ_map,
        od_relations
    )
    
    # =========== STEP 6: Create Labels ===========
    print("\n[STEP 6] Creating labels...")
    if patient_labels is None:
        # Generate random multi-label binary labels for testing
        patient_labels = {
            pid: torch.randint(0, 2, (num_diseases,)).float() 
            for pid in final_patient_ids
        }
        print(f"[INFO] Generated random labels for {len(patient_labels)} patients")
    
    # =========== STEP 7: Build Unified Graph ===========
    print("\n[STEP 7] Building unified graph...")
    
    graph_data = GraphData(
        patient_features=patient_features,
        patient_ids=final_patient_ids,
        patient_idx_map=final_patient_idx_map,
        concept_map=concept_map,
        disease_map=disease_map,
        organ_map=organ_map,
        patient_symptom_edges=ps_relations,
        symptom_organ_edges=so_relations,
        organ_disease_edges=od_relations,
        patient_labels=patient_labels
    )
    
    unified_graph = construct_unified_graph(ps_graph, so_graph, od_graph, graph_data)
    
    # =========== STEP 8: Extract Metadata ===========
    metadata = unified_graph.metadata()
    print(f"\n[INFO] Graph metadata:")
    print(f"  - Node types: {metadata[0]}")
    print(f"  - Edge types: {metadata[1]}")
    
    entity_maps = {
        'concept_map': concept_map,
        'organ_map': organ_map,
        'disease_map': disease_map,
        'patient_idx_map': final_patient_idx_map
    }
    
    print("\n" + "="*60)
    print("GRAPH CREATION COMPLETE")
    print("="*60)
    
    return unified_graph, graph_data, entity_maps


# ==============================================================================
# CONVENIENCE FUNCTION FOR NOTEBOOK
# ==============================================================================

def quick_create_graph(
    emr_file: str = 'data/patient_reports.csv',
    kb_file: str = 'data/enhanced_symptom_connectivity_analysis(Sheet1).csv',
    output_dim: int = 128
):
    """
    Quick graph creation with default parameters.
    Useful for notebook usage.
    
    Example:
        from graph import create
        graph, data, maps = create.quick_create_graph()
    """
    return create_hgt_graph(emr_file, kb_file, output_dim)
