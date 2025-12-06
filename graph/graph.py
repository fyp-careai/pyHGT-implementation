"""
Heterogeneous Graph Construction Functions
==========================================
This module contains functions to build bipartite graphs and merge them
into a unified HeteroData object for GNN/HGT training.
"""

import torch
import pandas as pd
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple


# ==============================================================================
# BIPARTITE GRAPH BUILDERS
# ==============================================================================

def construct_patient_symptom_graph(
    patient_ids: List,
    patient_features: Dict[int, torch.Tensor],
    patient_idx_map: dict,
    concept_map: dict,
    raw_data: pd.DataFrame
) -> HeteroData:
    """
    Constructs Patient <-> Symptom bipartite graph from EMR data.
    
    Args:
        patient_ids: List of valid patient IDs
        patient_features: Dict {PatientID: embedding tensor}
        patient_idx_map: Dict {PatientID: node index}
        concept_map: Dict {TestName: node index}
        raw_data: DataFrame with patient-symptom relations
        
    Returns:
        HeteroData with patient and symptom nodes, and bidirectional edges
    """
    data = HeteroData()

    if not patient_ids:
        print("[ERROR] No valid patients found. Cannot build graph.")
        return data

    num_symptoms = len(concept_map)
    gnn_embedding_dim = patient_features[patient_ids[0]].shape[0]

    # Node Features
    data['patient'].x = torch.stack([patient_features[pid] for pid in patient_ids])
    data['patient'].node_map = patient_idx_map
    
    data['symptom'].x = torch.rand(num_symptoms, gnn_embedding_dim)
    data['symptom'].node_map = concept_map

    # Edge Construction
    p_to_s_edges = []
    for _, row in raw_data.iterrows():
        p_id = row['PatientID']
        s_name = row['TestName']

        if p_id in patient_idx_map and s_name in concept_map:
            p_idx = patient_idx_map[p_id]
            s_idx = concept_map[s_name]
            p_to_s_edges.append((p_idx, s_idx))

    if p_to_s_edges:
        src, dst = zip(*p_to_s_edges)
        data['patient', 'has', 'symptom'].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data['symptom', 'is_related_to', 'patient'].edge_index = torch.tensor([dst, src], dtype=torch.long)
        print(f"[SUCCESS] Built {len(p_to_s_edges)} Patient->Symptom edges")
    else:
        data['patient', 'has', 'symptom'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['symptom', 'is_related_to', 'patient'].edge_index = torch.empty((2, 0), dtype=torch.long)
        print("[WARN] No Patient->Symptom edges created")

    return data


def construct_symptom_organ_graph(
    concept_map: dict,
    organ_map: dict,
    raw_data: pd.DataFrame
) -> HeteroData:
    """
    Constructs Symptom -> Organ bipartite graph from knowledge base.
    
    Args:
        concept_map: Dict {TestName: node index}
        organ_map: Dict {OrganName: node index}
        raw_data: DataFrame with TestName and Target_Organ columns
        
    Returns:
        HeteroData with symptom-organ edges
    """
    data = HeteroData()

    data['symptom'].num_nodes = len(concept_map)
    data['organ'].num_nodes = len(organ_map)

    s_to_o_edges = []
    for _, row in raw_data.iterrows():
        s_name = row['TestName']
        o_name = row['Target_Organ']
        
        if s_name in concept_map and o_name in organ_map:
            s_idx = concept_map[s_name]
            o_idx = organ_map[o_name]
            s_to_o_edges.append((s_idx, o_idx))

    if s_to_o_edges:
        src, dst = zip(*s_to_o_edges)
        data['symptom', 'measures', 'organ'].edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"[SUCCESS] Built {len(s_to_o_edges)} Symptom->Organ edges")

    return data


def construct_organ_disease_graph(
    disease_map: dict,
    organ_map: dict,
    raw_data: pd.DataFrame
) -> HeteroData:
    """
    Constructs Organ -> Disease bipartite graph from knowledge base.
    
    Args:
        disease_map: Dict {DiseaseName: node index}
        organ_map: Dict {OrganName: node index}
        raw_data: DataFrame with Target_Organ and Most_Relevant_Disease columns
        
    Returns:
        HeteroData with organ-disease edges
    """
    data = HeteroData()

    data['disease'].num_nodes = len(disease_map)
    data['organ'].num_nodes = len(organ_map)

    o_to_d_edges = []
    for _, row in raw_data.iterrows():
        d_name = row['Most_Relevant_Disease']
        o_name = row['Target_Organ']
        
        if d_name in disease_map and o_name in organ_map:
            d_idx = disease_map[d_name]
            o_idx = organ_map[o_name]
            o_to_d_edges.append((o_idx, d_idx))

    if o_to_d_edges:
        src, dst = zip(*o_to_d_edges)
        data['organ', 'affects', 'disease'].edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"[SUCCESS] Built {len(o_to_d_edges)} Organ->Disease edges")

    return data


# ==============================================================================
# UNIFIED GRAPH CONSTRUCTION
# ==============================================================================

class GraphData:
    """Container for pre-processed graph data and mappings."""
    
    def __init__(
        self,
        patient_features: Dict[int, torch.Tensor],
        patient_ids: List,
        patient_idx_map: dict,
        concept_map: dict,
        disease_map: dict,
        organ_map: dict,
        patient_symptom_edges: pd.DataFrame,
        symptom_organ_edges: pd.DataFrame,
        organ_disease_edges: pd.DataFrame,
        patient_labels: Dict[int, torch.Tensor]
    ):
        self.patient_features = patient_features
        self.patient_ids = patient_ids
        self.patient_idx_map = patient_idx_map
        self.concept_map = concept_map
        self.disease_map = disease_map
        self.organ_map = organ_map
        self.patient_symptom_edges = patient_symptom_edges
        self.symptom_organ_edges = symptom_organ_edges
        self.organ_disease_edges = organ_disease_edges
        self.patient_labels = patient_labels

        # Computed properties
        self.X_patients = torch.stack([patient_features[pid] for pid in patient_ids])
        self.embedding_dim = self.X_patients.shape[1]
        self.num_diseases = len(disease_map)
        self.num_symptoms = len(concept_map)
        self.num_organs = len(organ_map)
        self.num_patients = len(patient_ids)


def construct_unified_graph(
    ps_graph: HeteroData,
    so_graph: HeteroData,
    od_graph: HeteroData,
    graph_data: GraphData
) -> HeteroData:
    """
    Merges all bipartite graphs into a single unified HeteroData object.
    This is the final graph structure expected by HGT/GNN models.
    
    Args:
        ps_graph: Patient-Symptom graph
        so_graph: Symptom-Organ graph
        od_graph: Organ-Disease graph
        graph_data: GraphData container with all metadata
        
    Returns:
        Unified HeteroData object ready for training
    """
    final_data = HeteroData()
    emb_dim = graph_data.embedding_dim

    # ==== Node Features ====
    final_data['patient'].x = ps_graph['patient'].x
    final_data['symptom'].x = ps_graph['symptom'].x
    final_data['disease'].x = torch.rand(graph_data.num_diseases, emb_dim)
    final_data['organ'].x = torch.rand(graph_data.num_organs, emb_dim)

    # ==== Edge Indices ====
    final_data['patient', 'has', 'symptom'].edge_index = ps_graph['patient', 'has', 'symptom'].edge_index
    final_data['symptom', 'is_related_to', 'patient'].edge_index = ps_graph['symptom', 'is_related_to', 'patient'].edge_index
    final_data['symptom', 'measures', 'organ'].edge_index = so_graph['symptom', 'measures', 'organ'].edge_index
    final_data['organ', 'affects', 'disease'].edge_index = od_graph['organ', 'affects', 'disease'].edge_index

    # ==== Labels ====
    final_data['patient'].y = torch.stack([graph_data.patient_labels[pid] for pid in graph_data.patient_ids])
    final_data['disease'].y_identity = torch.eye(graph_data.num_diseases)

    # ==== Verification ====
    print("\n" + "="*50)
    print("UNIFIED GRAPH VERIFICATION")
    print("="*50)
    
    print(f"\nNode Counts:")
    print(f"  - Patients:  {final_data['patient'].x.shape[0]}")
    print(f"  - Symptoms:  {final_data['symptom'].x.shape[0]}")
    print(f"  - Organs:    {final_data['organ'].x.shape[0]}")
    print(f"  - Diseases:  {final_data['disease'].x.shape[0]}")
    
    print(f"\nEdge Counts:")
    for edge_type in final_data.edge_types:
        count = final_data[edge_type].edge_index.shape[1]
        print(f"  - {edge_type}: {count}")
    
    print(f"\nTotal Edges: {final_data.num_edges}")
    print("="*50)

    return final_data


# ==============================================================================
# RELATION IDENTIFICATION UTILITIES  
# ==============================================================================

def identify_patient_symptom_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique Patient-Symptom relations with frequency."""
    df.columns = df.columns.str.strip()
    relation_counts = df.groupby(['PatientID', 'TestName']).size().reset_index(name='Frequency')
    print(f"[INFO] Found {len(relation_counts)} unique Patient-Symptom relations")
    return relation_counts


def identify_symptom_organ_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique Symptom-Organ relations with frequency."""
    df.columns = df.columns.str.strip()
    relation_counts = df.groupby(['TestName', 'Target_Organ']).size().reset_index(name='Frequency')
    print(f"[INFO] Found {len(relation_counts)} unique Symptom-Organ relations")
    return relation_counts


def identify_organ_disease_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique Organ-Disease relations with frequency."""
    df.columns = df.columns.str.strip()
    relation_counts = df.groupby(['Target_Organ', 'Most_Relevant_Disease']).size().reset_index(name='Frequency')
    print(f"[INFO] Found {len(relation_counts)} unique Organ-Disease relations")
    return relation_counts
