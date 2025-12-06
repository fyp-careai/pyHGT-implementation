"""
HeteroData to pyHGT Format Converter
====================================
Converts PyTorch Geometric HeteroData objects to the tensor format
expected by the pyHGT model from the 2020 HGT paper.

The pyHGT model expects:
- node_feature: [N, in_dim] - all node features concatenated
- node_type: [N] - integer type ID for each node
- edge_index: [2, E] - all edges concatenated (source, target)
- edge_type: [E] - integer relation ID for each edge
- edge_time: [E] - relative time difference for temporal encoding
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PyHGTData:
    """Container for pyHGT-compatible tensors."""
    node_feature: torch.Tensor      # [N, in_dim]
    node_type: torch.Tensor         # [N]
    edge_index: torch.Tensor        # [2, E]
    edge_type: torch.Tensor         # [E]
    edge_time: torch.Tensor         # [E]
    node_dict: Dict[str, Tuple[int, int]]  # {type: (start_idx, type_id)}
    edge_dict: Dict[str, int]       # {relation: relation_id}
    num_types: int
    num_relations: int
    
    def to(self, device):
        """Move all tensors to specified device."""
        self.node_feature = self.node_feature.to(device)
        self.node_type = self.node_type.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        self.edge_time = self.edge_time.to(device)
        return self


def convert_heterodata_to_pyhgt(
    hetero_data: HeteroData,
    time_dict: Optional[Dict[str, torch.Tensor]] = None,
    default_time: int = 0,
    add_reverse_edges: bool = True,
    add_self_loops: bool = True
) -> PyHGTData:
    """
    Convert a HeteroData object to pyHGT tensor format.
    
    Args:
        hetero_data: PyG HeteroData object with node features and edges
        time_dict: Optional dict mapping edge types to time tensors.
                   If None, uses default_time for all edges.
        default_time: Default time value when no temporal info available.
                      pyHGT uses relative time (target_time - source_time + offset).
                      Set to 0 or 120 (midpoint) for static graphs.
        add_reverse_edges: Whether to add reverse edges (rev_*) for each relation
        add_self_loops: Whether to add self-loop edges for each node
        
    Returns:
        PyHGTData object with all tensors ready for GNN.forward()
        
    Example:
        >>> from graph import create_hgt_graph
        >>> from graph.converter import convert_heterodata_to_pyhgt
        >>> 
        >>> hetero_graph, _, _ = create_hgt_graph('emr.csv', 'kb.csv')
        >>> pyhgt_data = convert_heterodata_to_pyhgt(hetero_graph)
        >>> 
        >>> # Use with pyHGT model
        >>> from pyHGT.model import GNN
        >>> model = GNN(in_dim=128, n_hid=256, num_types=pyhgt_data.num_types,
        ...             num_relations=pyhgt_data.num_relations, n_heads=8, n_layers=3)
        >>> out = model(pyhgt_data.node_feature, pyhgt_data.node_type,
        ...             pyhgt_data.edge_time, pyhgt_data.edge_index, pyhgt_data.edge_type)
    """
    
    # =========================================================================
    # Step 1: Build node_dict and collect node features
    # =========================================================================
    node_types = hetero_data.node_types
    node_dict = {}  # {type_name: (start_index, type_id)}
    
    node_features_list = []
    node_type_list = []
    
    node_offset = 0
    for type_id, node_type in enumerate(node_types):
        num_nodes = hetero_data[node_type].num_nodes
        
        # Get or create node features
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            features = hetero_data[node_type].x
        else:
            # Create dummy features if not present
            in_dim = _infer_feature_dim(hetero_data)
            features = torch.zeros(num_nodes, in_dim)
            print(f"[WARN] No features for '{node_type}', using zeros")
        
        node_dict[node_type] = (node_offset, type_id)
        node_features_list.append(features)
        node_type_list.extend([type_id] * num_nodes)
        node_offset += num_nodes
    
    node_feature = torch.cat(node_features_list, dim=0)
    node_type_tensor = torch.LongTensor(node_type_list)
    
    # =========================================================================
    # Step 2: Build edge_dict and collect edges
    # =========================================================================
    edge_types = hetero_data.edge_types  # List of (src_type, rel, dst_type)
    
    # Create relation mapping (only relation names, not full triplets)
    relation_names = set()
    for src_type, rel, dst_type in edge_types:
        relation_names.add(rel)
        if add_reverse_edges:
            relation_names.add(f'rev_{rel}')
    
    if add_self_loops:
        relation_names.add('self')
    
    edge_dict = {rel: i for i, rel in enumerate(sorted(relation_names))}
    
    edge_index_list = []
    edge_type_list = []
    edge_time_list = []
    
    # Process original edges
    for src_type, rel, dst_type in edge_types:
        edge_key = (src_type, rel, dst_type)
        
        if not hasattr(hetero_data[edge_key], 'edge_index'):
            continue
            
        ei = hetero_data[edge_key].edge_index
        if ei.numel() == 0:
            continue
        
        num_edges = ei.size(1)
        src_offset = node_dict[src_type][0]
        dst_offset = node_dict[dst_type][0]
        
        # Convert local indices to global indices
        # pyHGT edge_index format: [source, target] where source -> target
        global_src = ei[0] + src_offset
        global_dst = ei[1] + dst_offset
        
        edge_index_list.append(torch.stack([global_src, global_dst], dim=0))
        edge_type_list.extend([edge_dict[rel]] * num_edges)
        
        # Handle time
        if time_dict and edge_key in time_dict:
            edge_time_list.extend(time_dict[edge_key].tolist())
        else:
            edge_time_list.extend([default_time] * num_edges)
        
        # Add reverse edges
        if add_reverse_edges:
            rev_rel = f'rev_{rel}'
            edge_index_list.append(torch.stack([global_dst, global_src], dim=0))
            edge_type_list.extend([edge_dict[rev_rel]] * num_edges)
            edge_time_list.extend([default_time] * num_edges)
    
    # Add self-loops
    if add_self_loops:
        total_nodes = node_feature.size(0)
        self_indices = torch.arange(total_nodes)
        edge_index_list.append(torch.stack([self_indices, self_indices], dim=0))
        edge_type_list.extend([edge_dict['self']] * total_nodes)
        edge_time_list.extend([default_time] * total_nodes)
    
    # Concatenate all edges
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    edge_type_tensor = torch.LongTensor(edge_type_list)
    edge_time_tensor = torch.LongTensor(edge_time_list)
    
    # =========================================================================
    # Step 3: Create output
    # =========================================================================
    return PyHGTData(
        node_feature=node_feature,
        node_type=node_type_tensor,
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        edge_time=edge_time_tensor,
        node_dict=node_dict,
        edge_dict=edge_dict,
        num_types=len(node_types),
        num_relations=len(edge_dict)
    )


def convert_with_timestamps(
    hetero_data: HeteroData,
    patient_times: Dict[int, int],
    patient_idx_map: Dict[int, int],
    base_year: int = 2020,
    time_offset: int = 120
) -> PyHGTData:
    """
    Convert HeteroData with actual timestamps for dynamic graph modeling.
    
    The pyHGT RTE (Relative Temporal Encoding) uses:
        edge_time = target_node_time - source_node_time + offset
    
    This allows the model to learn temporal patterns like:
    - Recent connections are weighted differently than old ones
    - Time-aware attention mechanisms
    
    Args:
        hetero_data: HeteroData object
        patient_times: Dict {PatientID: year/timestamp}
        patient_idx_map: Dict {PatientID: node_index in graph}
        base_year: Reference year for time calculation
        time_offset: Offset to keep values positive (default 120 for year range)
        
    Returns:
        PyHGTData with temporal edge information
        
    Note:
        For medical data, you might use:
        - Year of diagnosis/visit as timestamp
        - Days since first visit
        - Normalized time values (0-240 range for max_len in RTE)
    """
    # First convert without time
    pyhgt_data = convert_heterodata_to_pyhgt(
        hetero_data,
        default_time=time_offset,  # Use midpoint as default
        add_reverse_edges=True,
        add_self_loops=True
    )
    
    # If no temporal data provided, return as-is
    if not patient_times:
        print("[INFO] No temporal data provided, using static time values")
        return pyhgt_data
    
    # Create node time array
    total_nodes = pyhgt_data.node_feature.size(0)
    node_time = torch.zeros(total_nodes, dtype=torch.long)
    
    # Assign times to patient nodes
    patient_offset = pyhgt_data.node_dict.get('patient', (0, 0))[0]
    for pid, time_val in patient_times.items():
        if pid in patient_idx_map:
            global_idx = patient_idx_map[pid] + patient_offset
            # Normalize time to reasonable range
            normalized_time = min(max(time_val - base_year + time_offset, 0), 239)
            node_time[global_idx] = normalized_time
    
    # Recompute edge times based on source/target node times
    edge_index = pyhgt_data.edge_index
    num_edges = edge_index.size(1)
    
    new_edge_time = torch.zeros(num_edges, dtype=torch.long)
    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        
        # Relative time: target_time - source_time + offset
        rel_time = node_time[dst_idx] - node_time[src_idx] + time_offset
        rel_time = min(max(rel_time, 0), 239)  # Clamp to valid range
        new_edge_time[i] = rel_time
    
    pyhgt_data.edge_time = new_edge_time
    return pyhgt_data


def _infer_feature_dim(hetero_data: HeteroData) -> int:
    """Infer feature dimension from any node type that has features."""
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            return hetero_data[node_type].x.size(1)
    return 128  # Default


def print_conversion_summary(pyhgt_data: PyHGTData):
    """Print summary of converted data."""
    print("\n" + "="*60)
    print("pyHGT Data Conversion Summary")
    print("="*60)
    print(f"Total Nodes: {pyhgt_data.node_feature.size(0)}")
    print(f"Feature Dim: {pyhgt_data.node_feature.size(1)}")
    print(f"Total Edges: {pyhgt_data.edge_index.size(1)}")
    print(f"Num Types:   {pyhgt_data.num_types}")
    print(f"Num Relations: {pyhgt_data.num_relations}")
    print("\nNode Types:")
    for ntype, (offset, tid) in pyhgt_data.node_dict.items():
        print(f"  {ntype}: type_id={tid}, start_idx={offset}")
    print("\nEdge Types:")
    for rel, rid in sorted(pyhgt_data.edge_dict.items(), key=lambda x: x[1]):
        count = (pyhgt_data.edge_type == rid).sum().item()
        print(f"  {rel}: relation_id={rid}, count={count}")
    print("="*60 + "\n")
