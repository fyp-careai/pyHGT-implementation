"""
Test Recommendation Module for Medical Diagnosis Support

This module implements a link prediction-based test recommendation system that:
1. Takes patient embeddings from the GNN
2. Predicts which tests would be most informative for a patient
3. Uses the existing Matcher model for link prediction between patients and tests
4. Recommends tests when disease predictions have low confidence

The recommendation process:
- Patient enters the system with some initial tests
- System predicts diseases with confidence scores
- If confidence is low (< threshold), recommend additional tests
- Link prediction estimates the usefulness of each potential test
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple


class TestRecommender:
    """
    Recommends medical tests for patients based on link prediction.
    
    Uses the GNN embeddings and a Matcher module to predict which tests
    would be most beneficial for a patient based on:
    1. Tests the patient hasn't taken yet
    2. Disease prediction confidence levels
    3. Link strength between patient and potential tests
    """
    
    def __init__(self, matcher_model, confidence_threshold=0.7):
        """
        Initialize the test recommender.
        
        Args:
            matcher_model: The Matcher model from pyHGT for link prediction
            confidence_threshold: Threshold below which we recommend more tests
        """
        self.matcher = matcher_model
        self.confidence_threshold = confidence_threshold
        
    def get_disease_confidence(self, disease_probs):
        """
        Calculate confidence metrics for disease predictions.
        
        Args:
            disease_probs: Tensor of disease probabilities [batch_size, num_diseases]
            
        Returns:
            max_confidence: Maximum confidence across diseases
            mean_confidence: Mean confidence across diseases
            entropy: Prediction entropy (higher = more uncertain)
        """
        # Max confidence
        max_confidence = torch.max(disease_probs, dim=1)[0]
        
        # Mean confidence for positive predictions
        mean_confidence = torch.mean(disease_probs, dim=1)
        
        # Calculate entropy as uncertainty measure
        # H = -sum(p * log(p) + (1-p) * log(1-p))
        eps = 1e-10
        entropy = -(disease_probs * torch.log(disease_probs + eps) + 
                    (1 - disease_probs) * torch.log(1 - disease_probs + eps))
        entropy = torch.mean(entropy, dim=1)
        
        return max_confidence, mean_confidence, entropy
    
    def filter_available_tests(self, patient_ids, graph, all_test_embeddings):
        """
        Filter out tests that patients have already taken.
        
        Args:
            patient_ids: List of patient IDs
            graph: The heterogeneous graph object
            all_test_embeddings: Embeddings of all test nodes
            
        Returns:
            available_tests_mask: Boolean mask for each patient indicating available tests
            test_names: List of test names corresponding to embeddings
        """
        # Get all test names from graph
        test2idx = graph.node_forward["lab_test"]
        test_names = list(test2idx.keys())
        
        # For each patient, check which tests they've taken
        available_tests = []
        
        for pid in patient_ids:
            # Get edges from this patient
            patient_node = {"type": "patient", "id": str(pid)}
            taken_tests = set()
            
            # Check patient's edges to find taken tests
            if str(pid) in graph.node_forward["patient"]:
                patient_idx = graph.node_forward["patient"][str(pid)]
                
                # Look through edges to find had_test relations
                edge_key = ("patient", "lab_test", "had_test")
                if edge_key in graph.edge_list:
                    edges = graph.edge_list[edge_key]
                    for src_idx, dst_idx, time in edges:
                        if src_idx == patient_idx:
                            # Find test name from index
                            for test_name, test_idx in test2idx.items():
                                if test_idx == dst_idx:
                                    taken_tests.add(test_name)
                                    break
            
            # Create mask: True if test NOT taken
            mask = [test_name not in taken_tests for test_name in test_names]
            available_tests.append(mask)
        
        return np.array(available_tests), test_names
    
    def recommend_tests(
        self, 
        patient_embeddings, 
        test_embeddings,
        patient_ids,
        disease_probs,
        graph,
        top_k=5,
        force_recommend=False
    ):
        """
        Recommend tests for patients based on link prediction.
        
        Args:
            patient_embeddings: Embeddings from GNN [batch_size, hidden_dim]
            test_embeddings: Embeddings of all test nodes [num_tests, hidden_dim]
            patient_ids: List of patient IDs
            disease_probs: Disease prediction probabilities [batch_size, num_diseases]
            graph: The heterogeneous graph
            top_k: Number of tests to recommend
            force_recommend: If True, recommend even if confidence is high
            
        Returns:
            recommendations: List of dicts with recommendations for each patient
        """
        batch_size = patient_embeddings.shape[0]
        
        # Calculate confidence metrics
        max_conf, mean_conf, entropy = self.get_disease_confidence(disease_probs)
        
        # Filter available tests (not yet taken)
        available_mask, test_names = self.filter_available_tests(
            patient_ids, graph, test_embeddings
        )
        
        # Compute link prediction scores using Matcher
        # Shape: [batch_size, num_tests]
        with torch.no_grad():
            link_scores = self.matcher(patient_embeddings, test_embeddings, 
                                      infer=True, pair=False)
            link_scores = torch.sigmoid(link_scores)  # Convert to probabilities
        
        link_scores_np = link_scores.cpu().numpy()
        
        # Prepare recommendations
        recommendations = []
        
        for i in range(batch_size):
            patient_id = patient_ids[i]
            confidence = max_conf[i].item()
            
            # Decide if recommendation needed
            needs_recommendation = (confidence < self.confidence_threshold) or force_recommend
            
            if needs_recommendation:
                # Mask out already taken tests
                scores = link_scores_np[i].copy()
                scores[~available_mask[i]] = -np.inf  # Set taken tests to very low score
                
                # Get top-k test indices
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                # Get actual tests that are available
                recommended_tests = []
                for idx in top_indices:
                    if available_mask[i][idx]:  # Only if test is available
                        recommended_tests.append({
                            "test_name": test_names[idx],
                            "score": float(scores[idx]),
                            "rank": len(recommended_tests) + 1
                        })
                    
                    if len(recommended_tests) >= top_k:
                        break
                
                recommendations.append({
                    "patient_id": patient_id,
                    "needs_recommendation": True,
                    "confidence": float(confidence),
                    "mean_confidence": float(mean_conf[i].item()),
                    "entropy": float(entropy[i].item()),
                    "recommended_tests": recommended_tests
                })
            else:
                recommendations.append({
                    "patient_id": patient_id,
                    "needs_recommendation": False,
                    "confidence": float(confidence),
                    "mean_confidence": float(mean_conf[i].item()),
                    "entropy": float(entropy[i].item()),
                    "recommended_tests": []
                })
        
        return recommendations
    
    def explain_recommendation(self, recommendation, graph):
        """
        Provide explanation for why specific tests are recommended.
        
        Args:
            recommendation: Single recommendation dict
            graph: The heterogeneous graph
            
        Returns:
            explanation: String explaining the recommendation
        """
        if not recommendation["needs_recommendation"]:
            return f"Patient {recommendation['patient_id']}: High confidence ({recommendation['confidence']:.3f}), no additional tests needed."
        
        explanation = f"Patient {recommendation['patient_id']}:\n"
        explanation += f"  Current confidence: {recommendation['confidence']:.3f} (threshold: {self.confidence_threshold})\n"
        explanation += f"  Uncertainty (entropy): {recommendation['entropy']:.3f}\n"
        explanation += f"  Recommended tests ({len(recommendation['recommended_tests'])}):\n"
        
        for test_rec in recommendation["recommended_tests"]:
            test_name = test_rec["test_name"]
            score = test_rec["score"]
            rank = test_rec["rank"]
            
            # Get test details from graph
            test_info = self.get_test_info(test_name, graph)
            
            explanation += f"    {rank}. {test_name} (score: {score:.3f})\n"
            explanation += f"       Related organs: {', '.join(test_info['organs']) if test_info['organs'] else 'N/A'}\n"
            explanation += f"       Associated diseases: {', '.join(test_info['diseases']) if test_info['diseases'] else 'N/A'}\n"
        
        return explanation
    
    def get_test_info(self, test_name, graph):
        """
        Get organ and disease information for a test.
        
        Args:
            test_name: Name of the test
            graph: The heterogeneous graph
            
        Returns:
            info: Dict with organs and diseases
        """
        organs = []
        diseases = []
        
        if test_name not in graph.node_forward["lab_test"]:
            return {"organs": organs, "diseases": diseases}
        
        test_idx = graph.node_forward["lab_test"][test_name]
        
        # Find connected organs
        organ_edge_key = ("lab_test", "organ", "tests_organ")
        if organ_edge_key in graph.edge_list:
            for src_idx, dst_idx, _ in graph.edge_list[organ_edge_key]:
                if src_idx == test_idx:
                    # Find organ name
                    for organ_name, organ_idx in graph.node_forward["organ"].items():
                        if organ_idx == dst_idx:
                            organs.append(organ_name)
                            break
        
        # Find connected diseases
        disease_edge_key = ("lab_test", "disease", "associated_with")
        if disease_edge_key in graph.edge_list:
            for src_idx, dst_idx, _ in graph.edge_list[disease_edge_key]:
                if src_idx == test_idx:
                    # Find disease name
                    for disease_name, disease_idx in graph.node_forward["disease"].items():
                        if disease_idx == dst_idx:
                            diseases.append(disease_name)
                            break
        
        return {"organs": organs, "diseases": diseases}
