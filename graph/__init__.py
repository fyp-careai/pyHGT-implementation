"""
Graph Construction Module for HGT
=================================
This package provides tools for building heterogeneous graphs
suitable for Heterogeneous Graph Transformer (HGT) training.

Main components:
- encoders: Patient feature encoders (TCN, simple aggregation)
- graph: Graph construction functions
- create: Main pipeline for graph creation

Usage:
    from graph import create_hgt_graph, quick_create_graph
    
    # Full pipeline
    graph, data, maps = create_hgt_graph('emr.csv', 'kb.csv')
    
    # Quick creation with defaults
    graph, data, maps = quick_create_graph()
"""

from .create import create_hgt_graph, quick_create_graph
from .graph import (
    GraphData,
    construct_patient_symptom_graph,
    construct_symptom_organ_graph,
    construct_organ_disease_graph,
    construct_unified_graph
)
from .encoders import (
    TCNEncoder,
    simple_aggregate_visits,
    transform_raw_to_features,
    generate_patient_embeddings
)
from .converter import (
    PyHGTData,
    convert_heterodata_to_pyhgt,
    convert_with_timestamps,
    print_conversion_summary
)
from .labels import (
    PatientLabels,
    ThresholdRange,
    generate_patient_disease_labels,
    load_knowledge_base_with_thresholds,
    quick_generate_labels,
    save_labels,
    print_patient_summary
)

__all__ = [
    # Main functions
    'create_hgt_graph',
    'quick_create_graph',
    
    # Graph builders
    'GraphData',
    'construct_patient_symptom_graph',
    'construct_symptom_organ_graph',
    'construct_organ_disease_graph',
    'construct_unified_graph',
    
    # Encoders
    'TCNEncoder',
    'simple_aggregate_visits',
    'transform_raw_to_features',
    'generate_patient_embeddings',
    
    # Converter (HeteroData -> pyHGT format)
    'PyHGTData',
    'convert_heterodata_to_pyhgt',
    'convert_with_timestamps',
    'print_conversion_summary',
    
    # Label Generation
    'PatientLabels',
    'ThresholdRange',
    'generate_patient_disease_labels',
    'load_knowledge_base_with_thresholds',
    'quick_generate_labels',
    'save_labels',
    'print_patient_summary',
]
