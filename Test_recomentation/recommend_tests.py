"""
Test Recommendation Inference Script

This script demonstrates how to use the trained test recommendation model
to recommend tests for patients based on their current test results and
disease prediction confidence.

Usage:
    python recommend_tests.py --patient_id <patient_id>
    python recommend_tests.py --demo  # Run demo with sample patients
"""

import os
import numpy as np
import pandas as pd
import torch
import argparse
import json

from pyHGT.data import Graph, sample_subgraph, to_torch
from pyHGT.model import GNN, Matcher
from test_recommender import TestRecommender

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================
# LOAD TRAINED MODEL
# ============================================

def load_model(model_path):
    """Load the trained test recommendation model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Model parameters
    in_dim = checkpoint['in_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_types = checkpoint['num_types']
    num_relations = checkpoint['num_relations']
    n_heads = checkpoint['n_heads']
    n_layers = checkpoint['n_layers']
    num_diseases = checkpoint['num_diseases']
    
    # Initialize models
    gnn = GNN(
        in_dim=in_dim,
        n_hid=hidden_dim,
        num_types=num_types,
        num_relations=num_relations,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.2,
        conv_name='hgt',
        prev_norm=False,
        last_norm=False,
        use_RTE=True
    ).to(device)
    
    matcher = Matcher(hidden_dim).to(device)
    
    class MultilabelClassifier(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim, out_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    clf = MultilabelClassifier(hidden_dim, num_diseases).to(device)
    
    # Load weights
    gnn.load_state_dict(checkpoint['gnn_state_dict'])
    matcher.load_state_dict(checkpoint['matcher_state_dict'])
    clf.load_state_dict(checkpoint['clf_state_dict'])
    
    # Set to evaluation mode
    gnn.eval()
    matcher.eval()
    clf.eval()
    
    return gnn, matcher, clf, checkpoint

# ============================================
# LOAD DATA AND BUILD GRAPH
# ============================================

def load_graph_and_data():
    """Load the same graph structure as training"""
    data_dir = 'data/'
    
    patient_tests = pd.read_csv(os.path.join(data_dir,"patient-test.csv"), encoding='latin1')
    test_details = pd.read_csv(os.path.join(data_dir,"test-disease-organ.csv"), encoding='latin1')
    labels_df = pd.read_csv(os.path.join(data_dir,"patient-one-hot-labeled-disease.csv"), encoding='latin1')
    
    patient_tests["patient_id"] = patient_tests["patient_id"].astype(str)
    patient_tests["test_name"] = patient_tests["test_name"].astype(str)
    test_details["test_name"] = test_details["test_name"].astype(str)
    
    if "organ" in test_details.columns:
        test_details["organ"] = test_details["organ"].astype(str)
    if "disease" in test_details.columns:
        test_details["disease"] = test_details["disease"].astype(str)
    
    test_details = test_details.drop_duplicates(subset=["test_name"]).reset_index(drop=True)
    
    # Temporal processing
    patient_tests["report_date"] = pd.to_datetime(patient_tests["report_date"])
    patient_tests["time_idx"] = patient_tests["report_date"].astype("int64") // 10**9
    min_time = patient_tests["time_idx"].min()
    patient_tests["rel_time"] = (patient_tests["time_idx"]-min_time)// 86400
    max_rel_time = int(patient_tests["rel_time"].max())
    
    def parse_multi(x):
        if pd.isna(x):
            return []
        x = str(x).strip()
        if ";" in x:
            return [item.strip() for item in x.split(";")]
        if "," in x:
            return [item.strip() for item in x.split(",")]
        return [x] if x else []
    
    # Build lab info
    lab_info = {}
    for _, row in test_details.iterrows():
        test = str(row["test_name"]).strip()
        organs = parse_multi(row.get("organ", ""))
        diseases = parse_multi(row.get("disease", ""))
        
        low_th = None
        high_th = None
        if "min" in row and not pd.isna(row["min"]):
            low_th = float(row["min"])
        if "max" in row and not pd.isna(row["max"]):
            high_th = float(row["max"])
        
        lab_info[test] = {
            "organs": organs,
            "diseases": diseases,
            "low": low_th,
            "high": high_th
        }
    
    # Compute features
    pt_merged = patient_tests.merge(
        test_details[["test_name", "min", "max"]],
        on="test_name",
        how="left"
    )
    
    def compute_abnoramability(row):
        v = float(row["test_value"])
        low = row["min"]
        high = row["max"]
        if pd.isna(low) or pd.isna(high) or low >= high:
            return 0.0
        return float((v - low) / (high - low))
    
    pt_merged["abnormality"] = pt_merged.apply(compute_abnoramability, axis=1)
    
    agg = pt_merged.groupby("patient_id").agg(
        num_tests = ("test_name", "count"),
        mean_abn = ("abnormality", "mean"),
        max_abn = ("abnormality", "max"),
        min_abn = ("abnormality", "min"),
        last_time = ("rel_time", "max")
    ).reset_index()
    
    patient_feat_df = agg.set_index("patient_id")
    
    # Extract unique nodes
    patient_ids = sorted(patient_tests["patient_id"].astype(str).unique().tolist())
    lab_tests = sorted(list(lab_info.keys()))
    all_organs = sorted({org for info in lab_info.values() for org in info["organs"] if org})
    all_diseases = sorted({dis for info in lab_info.values() for dis in info["diseases"] if dis})
    
    # Build graph
    graph = Graph()
    
    for pid in patient_ids:
        if pid in patient_feat_df.index:
            row = patient_feat_df.loc[pid]
            node = {
                "type": "patient",
                "id": pid,
                "num_tests": float(row["num_tests"]),
                "mean_abn":  float(row["mean_abn"]),
                "max_abn":   float(row["max_abn"]),
                "min_abn":   float(row["min_abn"]),
                "last_time": int(row["last_time"]),
                "time":      int(row["last_time"]),
            }
        else:
            node = {
                "type": "patient",
                "id": pid,
                "num_tests": 0.0,
                "mean_abn":  0.0,
                "max_abn":   0.0,
                "min_abn":   0.0,
                "last_time": 0,
                "time":      0,
            }
        graph.add_node(node)
    
    for test in lab_tests:
        info = lab_info[test]
        graph.add_node({
            "type": "lab_test",
            "id": test,
            "time": 0,
            "low": 0.0 if info["low"] is None else float(info["low"]),
            "high": 0.0 if info["high"] is None else float(info["high"])
        })
    
    for organ in all_organs:
        graph.add_node({
            "type": "organ",
            "id": organ,
            "time": 0
        })
    
    for disease in all_diseases:
        graph.add_node({
            "type": "disease",
            "id": disease,
            "time": 0
        })
    
    patient2idx = graph.node_forward["patient"]
    lab2idx     = graph.node_forward["lab_test"]
    organ2idx   = graph.node_forward["organ"]
    disease2idx = graph.node_forward["disease"]
    
    # Add edges
    for _, row in patient_tests.iterrows():
        pid = str(row["patient_id"])
        test = str(row["test_name"])
        t = int(row["rel_time"])
        
        if pid in patient2idx and test in lab2idx:
            graph.add_edge(
                {"type": "patient", "id": pid},
                {"type": "lab_test", "id": test},
                time=t,
                relation_type="had_test",
                directed=True,
            )
    
    for test, info in lab_info.items():
        for organ in info["organs"]:
            if organ in organ2idx:
                graph.add_edge(
                    {"type": "lab_test", "id": test},
                    {"type": "organ", "id": organ},
                    relation_type="tests_organ",
                    directed=True,
                    time=0
                )
        
        for disease in info["diseases"]:
            if disease in disease2idx:
                graph.add_edge(
                    {"type": "lab_test", "id": test},
                    {"type": "disease", "id": disease},
                    relation_type="associated_with",
                    directed=True,
                    time=0
                )
                
                for organ in info["organs"]:
                    if organ in organ2idx:
                        graph.add_edge(
                            {"type": "disease", "id": disease},
                            {"type": "organ", "id": organ},
                            relation_type="occurs_in",
                            directed=True,
                            time=0
                        )
    
    # Node features
    for t, node_list in graph.node_bacward.items():
        df = pd.DataFrame(node_list).reset_index(drop=True)
        graph.node_feature[t] = df
    
    # Disease labels
    disease_label_cols = [c for c in labels_df.columns if c != "patient_id"]
    
    return graph, patient_tests, labels_df, disease_label_cols, max_rel_time

def feature_medical(layer_data, graph):
    """Feature extraction function (same as training)"""
    feature = {}
    times = {}
    indxs = {}
    texts = {}
    
    for t in layer_data:
        if len(layer_data[t]) == 0:
            continue
        
        idxs = np.array(list(layer_data[t].keys()))
        tims = np.array(list(layer_data[t].values()))[:,1]
        
        df = graph.node_feature[t]
        feats = np.zeros((len(idxs),5), dtype=np.float32)
        
        if t == "patient":
            cols = ["num_tests", "mean_abn", "max_abn", "min_abn", "last_time"]
            vals = df.loc[idxs, cols].fillna(0).values.astype(np.float32)
            feats = vals
        elif t == "lab_test":
            for c in ["low", "high"]:
                if c not in df.columns:
                    df[c] = 0.0
            vals = df.loc[idxs, ["low", "high"]].fillna(0).values.astype(np.float32)
            feats[:,0:2] = vals
        
        feature[t] = feats
        times[t] = tims
        indxs[t] = idxs
    
    return feature, times, indxs, texts

# ============================================
# RECOMMENDATION FUNCTION
# ============================================

def recommend_tests_for_patient(patient_id, gnn, matcher, clf, graph, disease_label_cols, max_rel_time, top_k=5):
    """
    Recommend tests for a specific patient.
    
    Args:
        patient_id: Patient ID (string)
        gnn: Trained GNN model
        matcher: Trained Matcher model
        clf: Trained classifier
        graph: Heterogeneous graph
        disease_label_cols: List of disease labels
        max_rel_time: Maximum relative time
        top_k: Number of tests to recommend
    
    Returns:
        recommendations: Dict with patient info and recommendations
        disease_predictions: Disease prediction results
    """
    patient2idx = graph.node_forward["patient"]
    lab2idx = graph.node_forward["lab_test"]
    
    if str(patient_id) not in patient2idx:
        return None, None
    
    patient_idx = patient2idx[str(patient_id)]
    
    # Get patient embedding
    inp = {"patient": [(int(patient_idx), 0)]}
    
    feature, times, edge_list, indxs, texts = sample_subgraph(
        graph,
        time_range={max_rel_time: True},
        sampled_depth=2,
        sampled_number=8,
        inp=inp,
        feature_extractor=feature_medical
    )
    
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature, times, edge_list, graph
    )
    
    node_feature = node_feature.to(device)
    node_type = node_type.to(device)
    edge_time = edge_time.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    with torch.no_grad():
        all_embs = gnn(node_feature, node_type, edge_time, edge_index, edge_type)
    
    patient_offset, _ = node_dict["patient"]
    patient_emb = all_embs[patient_offset:patient_offset+1]
    
    # Disease prediction
    disease_logits = clf(patient_emb)
    disease_probs = torch.sigmoid(disease_logits)
    
    # Get all test embeddings
    all_test_indices = list(lab2idx.values())
    inp_tests = {"lab_test": [(int(tid), 0) for tid in all_test_indices]}
    
    feature_t, times_t, edge_list_t, indxs_t, texts_t = sample_subgraph(
        graph,
        time_range={max_rel_time: True},
        sampled_depth=2,
        sampled_number=8,
        inp=inp_tests,
        feature_extractor=feature_medical
    )
    
    node_feature_t, node_type_t, edge_time_t, edge_index_t, edge_type_t, node_dict_t, edge_dict_t = to_torch(
        feature_t, times_t, edge_list_t, graph
    )
    
    node_feature_t = node_feature_t.to(device)
    node_type_t = node_type_t.to(device)
    edge_time_t = edge_time_t.to(device)
    edge_index_t = edge_index_t.to(device)
    edge_type_t = edge_type_t.to(device)
    
    with torch.no_grad():
        all_embs_t = gnn(node_feature_t, node_type_t, edge_time_t, edge_index_t, edge_type_t)
    
    test_offset, _ = node_dict_t["lab_test"]
    test_embs = all_embs_t[test_offset:test_offset+len(all_test_indices)]
    
    # Get recommendations
    recommender = TestRecommender(matcher, confidence_threshold=0.7)
    recommendations = recommender.recommend_tests(
        patient_emb,
        test_embs,
        [patient_id],
        disease_probs,
        graph,
        top_k=top_k,
        force_recommend=False
    )
    
    # Format disease predictions
    disease_pred_dict = {}
    for i, disease in enumerate(disease_label_cols):
        prob = disease_probs[0, i].item()
        if prob > 0.3:  # Only show diseases with reasonable probability
            disease_pred_dict[disease] = prob
    
    # Sort by probability
    disease_pred_dict = dict(sorted(disease_pred_dict.items(), key=lambda x: x[1], reverse=True))
    
    return recommendations[0], disease_pred_dict

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Test Recommendation Inference')
    parser.add_argument('--patient_id', type=str, help='Patient ID to recommend tests for')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample patients')
    parser.add_argument('--top_k', type=int, default=5, help='Number of tests to recommend')
    parser.add_argument('--model_path', type=str, default='models_saved/test_recommendation_model.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    print("Loading model and data...")
    gnn, matcher, clf, checkpoint = load_model(args.model_path)
    graph, patient_tests, labels_df, disease_label_cols, max_rel_time = load_graph_and_data()
    
    print(f"Model loaded successfully!")
    print(f"Number of diseases: {len(disease_label_cols)}")
    print(f"Number of patients: {len(graph.node_forward['patient'])}")
    print(f"Number of tests: {len(graph.node_forward['lab_test'])}")
    
    if args.demo:
        # Demo with a few random patients
        all_patient_ids = list(graph.node_forward["patient"].keys())
        sample_patient_ids = np.random.choice(all_patient_ids, min(5, len(all_patient_ids)), replace=False)
        
        print("\n" + "="*70)
        print("DEMO: Test Recommendations for Sample Patients")
        print("="*70)
        
        for pid in sample_patient_ids:
            print(f"\n{'='*70}")
            print(f"Patient ID: {pid}")
            print(f"{'='*70}")
            
            recommendations, disease_preds = recommend_tests_for_patient(
                pid, gnn, matcher, clf, graph, disease_label_cols, max_rel_time, top_k=args.top_k
            )
            
            if recommendations is None:
                print(f"Patient {pid} not found in graph.")
                continue
            
            # Display disease predictions
            print("\n--- Disease Predictions (Top 5) ---")
            for i, (disease, prob) in enumerate(list(disease_preds.items())[:5]):
                print(f"  {i+1}. {disease}: {prob:.3f}")
            
            # Display recommendations
            print(f"\n--- Test Recommendations ---")
            print(f"Confidence: {recommendations['confidence']:.3f}")
            print(f"Needs recommendation: {recommendations['needs_recommendation']}")
            
            if recommendations['needs_recommendation']:
                print(f"\nRecommended Tests:")
                for test_rec in recommendations['recommended_tests']:
                    print(f"  {test_rec['rank']}. {test_rec['test_name']} (score: {test_rec['score']:.3f})")
            else:
                print("High confidence - no additional tests needed at this time.")
    
    elif args.patient_id:
        # Recommend for specific patient
        print(f"\nRecommending tests for patient: {args.patient_id}")
        
        recommendations, disease_preds = recommend_tests_for_patient(
            args.patient_id, gnn, matcher, clf, graph, disease_label_cols, max_rel_time, top_k=args.top_k
        )
        
        if recommendations is None:
            print(f"Patient {args.patient_id} not found in graph.")
            return
        
        print("\n" + "="*70)
        print(f"Patient ID: {args.patient_id}")
        print("="*70)
        
        # Display disease predictions
        print("\n--- Disease Predictions ---")
        for disease, prob in disease_preds.items():
            print(f"  {disease}: {prob:.3f}")
        
        # Display recommendations
        print(f"\n--- Test Recommendations ---")
        print(f"Confidence: {recommendations['confidence']:.3f}")
        print(f"Mean Confidence: {recommendations['mean_confidence']:.3f}")
        print(f"Entropy (uncertainty): {recommendations['entropy']:.3f}")
        print(f"Needs recommendation: {recommendations['needs_recommendation']}")
        
        if recommendations['needs_recommendation']:
            print(f"\nRecommended Tests:")
            for test_rec in recommendations['recommended_tests']:
                print(f"  {test_rec['rank']}. {test_rec['test_name']} (score: {test_rec['score']:.3f})")
        else:
            print("High confidence - no additional tests needed at this time.")
    
    else:
        print("Please specify --patient_id or --demo flag")
        parser.print_help()

if __name__ == "__main__":
    main()
