"""
Patient Feature Encoders for Heterogeneous Graph Construction
=============================================================
This module contains encoders that transform raw patient EMR data 
into fixed-size embeddings (h_patient^0) for GNN input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict


# ==============================================================================
# BASELINE: Simple Aggregation (Mean Pooling)
# ==============================================================================

def simple_aggregate_visits(visit_features: torch.Tensor, output_dim: int) -> torch.Tensor:
    """
    Implements Simple Aggregation (Baseline) method via Mean Pooling.
    Averages all features across the time dimension.
    
    Args:
        visit_features: Tensor of shape (sequence_length, input_dim)
        output_dim: Target embedding dimension
        
    Returns:
        Tensor of shape (output_dim,)
    """
    if visit_features.size(0) == 0:
        return torch.zeros(output_dim)

    # Take the mean across the time dimension (axis 0)
    mean_features = torch.mean(visit_features, dim=0)

    # Project to target output_dim
    # NOTE: In production, this layer should be trained with the GNN
    proj_layer = nn.Linear(mean_features.size(-1), output_dim)
    return F.relu(proj_layer(mean_features))


# ==============================================================================
# TCN (Temporal Convolutional Network) Encoder
# ==============================================================================

class Chomp1d(nn.Module):
    """Removes right-side padding artifacts from Conv1d output."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A basic residual block for TCNs with dilated convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, dilation: int, dropout: float):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder for patient visit sequences.
    Encodes a sequence of visits into a fixed-size embedding.
    
    Args:
        input_dim: Number of input features per visit
        hidden_channels: Hidden dimension in TCN layers
        output_dim: Final embedding dimension (h_patient^0 size)
        num_layers: Number of TCN layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_channels: int, output_dim: int, 
                 num_layers: int, kernel_size: int, dropout: float):
        super(TCNEncoder, self).__init__()

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation_size = 2 ** i
            out_channels = hidden_channels
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, dropout=dropout
            ))
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, output_dim)
        self.relu = nn.ReLU()

    def forward(self, visit_sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visit_sequences: Tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # TCN expects (batch_size, channels, sequence_length)
        x = visit_sequences.transpose(1, 2)
        output = self.tcn(x)
        
        # Use features from the last time step
        h_T = output[:, :, -1]
        return self.relu(self.fc(h_T))


# ==============================================================================
# Data Transformation Utility
# ==============================================================================

def transform_raw_to_features(group: pd.DataFrame, concept_map: dict, 
                               value_dim: int = 1) -> torch.Tensor:
    """
    Converts a patient's raw EMR records into a sequence of feature vectors.
    Each visit (same ReportDate) becomes one feature vector.
    
    Args:
        group: DataFrame with all records for a single patient, sorted by time
        concept_map: Dict mapping TestName -> index
        value_dim: Feature dimension per concept (default: 1 for normalized value)
        
    Returns:
        Tensor of shape (num_visits, num_concepts * value_dim)
    """
    visits = group.groupby('ReportDate').agg(list)
    num_concepts = len(concept_map)
    feature_list = []

    for _, visit in visits.iterrows():
        visit_vector = torch.zeros(num_concepts * value_dim)
        
        test_names = visit['TestName']
        test_values = visit['TestValue']

        for name, value in zip(test_names, test_values):
            if name in concept_map:
                idx = concept_map[name]
                # Simple normalization (adjust based on your data)
                normalized_value = float(value) / 100.0
                visit_vector[idx] = normalized_value

        feature_list.append(visit_vector)

    if not feature_list:
        return torch.empty(0, num_concepts * value_dim)

    return torch.stack(feature_list)


# ==============================================================================
# Patient Embedding Generator
# ==============================================================================

def generate_patient_embeddings(
    patient_data: pd.DataFrame,
    concept_map: dict,
    input_dim: int,
    output_dim: int = 128,
    hidden_dim: int = 64,
    tcn_layers: int = 2,
    tcn_kernel: int = 2,
    tcn_dropout: float = 0.2,
    value_dim: int = 1
) -> tuple:
    """
    Generates embeddings for all patients using TCN encoder.
    Falls back to simple aggregation for short sequences.
    
    Args:
        patient_data: DataFrame with columns [PatientID, ReportDate, TestName, TestValue]
        concept_map: Dict mapping TestName -> index
        input_dim: Number of unique concepts * value_dim
        output_dim: Target embedding dimension
        hidden_dim: TCN hidden dimension
        tcn_layers: Number of TCN layers
        tcn_kernel: TCN kernel size
        tcn_dropout: TCN dropout rate
        value_dim: Feature dimension per concept
        
    Returns:
        Tuple of (patient_features_dict, final_patient_ids, final_patient_idx_map)
    """
    tcn_encoder = TCNEncoder(input_dim, hidden_dim, output_dim, tcn_layers, tcn_kernel, tcn_dropout)
    patient_features_dict = {}

    for patient_id, group in patient_data.groupby('PatientID'):
        group = group.sort_values('ReportDate')
        visit_features = transform_raw_to_features(group, concept_map, value_dim)

        with torch.no_grad():
            if visit_features.size(0) >= tcn_kernel:
                ts_input = visit_features.unsqueeze(0)
                patient_features_dict[patient_id] = tcn_encoder(ts_input).squeeze(0)
            else:
                patient_features_dict[patient_id] = simple_aggregate_visits(visit_features, output_dim)

    final_patient_ids = list(patient_features_dict.keys())
    final_patient_idx_map = {pid: i for i, pid in enumerate(final_patient_ids)}

    print(f"[INFO] Generated embeddings for {len(final_patient_ids)} patients")
    print(f"[INFO] Embedding dimension: {output_dim}")

    return patient_features_dict, final_patient_ids, final_patient_idx_map
