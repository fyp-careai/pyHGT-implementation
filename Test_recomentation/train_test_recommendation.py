"""
Training Script for Test Recommendation System

This script trains a link prediction model to recommend medical tests for patients.
It extends the existing disease prediction model by adding a test recommendation component.

Key Features:
1. Loads the existing heterogeneous graph with patients, tests, organs, and diseases
2. Trains both disease prediction AND link prediction simultaneously
3. Link prediction learns patient-test associations
4. Evaluates recommendation quality using ranking metrics
5. Saves the trained model for test recommendations

Training Strategy:
- For each patient, we have their disease labels and tests they took
- We treat the patient-test edges as positive examples
- We sample negative examples (tests the patient didn't take)
- Train the Matcher to distinguish between useful and non-useful tests
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import json

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics
from sklearn.metrics import roc_auc_score, average_precision_score

from pyHGT.data import Graph, sample_subgraph, to_torch
from pyHGT.model import GNN, Matcher
from test_recommender import TestRecommender

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================
# LOAD DATA (same as train.py)
# ============================================

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
print("max_rel_time =", max_rel_time)

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

print("Number of unique tests:", len(lab_info))

# Compute abnormality features
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

print("#patients =", len(patient_ids))
print("#labs =", len(lab_tests))
print("#organs =", len(all_organs))
print("#diseases =", len(all_diseases))

# ============================================
# BUILD GRAPH (same as train.py)
# ============================================

graph = Graph()

# Add nodes
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

print("Meta relations in graph:", graph.get_meta_graph())

# Node features
for t, node_list in graph.node_bacward.items():
    df = pd.DataFrame(node_list).reset_index(drop=True)
    graph.node_feature[t] = df

for t,df in graph.node_feature.items():
    print(f"Node type: {t}, feature shape: {df.shape}")

def feature_medical(layer_data, graph):
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
        else:
            pass
        
        feature[t] = feats
        times[t] = tims
        indxs[t] = idxs
    
    return feature, times, indxs, texts

# Disease labels
disease_label_cols = [c for c in labels_df.columns if c != "patient_id"]
num_diseases = len(disease_label_cols)

print("Label columns:", disease_label_cols[:10], "...")
print("Number of diseases to predict:", num_diseases)

num_patients_in_graph = len(patient2idx)
Y = torch.zeros((num_patients_in_graph, num_diseases), dtype=torch.float32)

for _, row in labels_df.iterrows():
    pid = str(row["patient_id"])
    if pid in patient2idx:
        idx = patient2idx[pid]
        Y[idx] = torch.tensor(row[disease_label_cols].values, dtype=torch.float32)

print("Labels tensor shape:", Y.shape)

# ============================================
# PREPARE LINK PREDICTION DATA
# ============================================

print("\n" + "="*50)
print("PREPARING LINK PREDICTION DATA")
print("="*50)

# Build patient-test ground truth
patient_test_pairs = []
for _, row in patient_tests.iterrows():
    pid = str(row["patient_id"])
    test = str(row["test_name"])
    if pid in patient2idx and test in lab2idx:
        patient_test_pairs.append((patient2idx[pid], lab2idx[test]))

# Remove duplicates
patient_test_pairs = list(set(patient_test_pairs))
print(f"Total positive patient-test pairs: {len(patient_test_pairs)}")

# Split into train/val/test for link prediction
np.random.seed(42)
np.random.shuffle(patient_test_pairs)

n_pairs = len(patient_test_pairs)
n_train_links = int(0.7 * n_pairs)
n_val_links = int(0.15 * n_pairs)

train_links = patient_test_pairs[:n_train_links]
val_links = patient_test_pairs[n_train_links:n_train_links+n_val_links]
test_links = patient_test_pairs[n_train_links+n_val_links:]

print(f"Train links: {len(train_links)}")
print(f"Val links: {len(val_links)}")
print(f"Test links: {len(test_links)}")

# Create negative samples (patient-test pairs that don't exist)
def sample_negative_links(positive_links, num_patients, num_tests, num_negatives):
    """Sample negative links that don't exist in positive set"""
    positive_set = set(positive_links)
    negative_links = []
    
    while len(negative_links) < num_negatives:
        pid = np.random.randint(0, num_patients)
        tid = np.random.randint(0, num_tests)
        if (pid, tid) not in positive_set:
            negative_links.append((pid, tid))
    
    return negative_links

num_patients = len(patient2idx)
num_tests = len(lab2idx)

# Sample negatives for train/val/test
train_neg_links = sample_negative_links(train_links, num_patients, num_tests, len(train_links))
val_neg_links = sample_negative_links(val_links, num_patients, num_tests, len(val_links))
test_neg_links = sample_negative_links(test_links, num_patients, num_tests, len(test_links))

print(f"Negative train links: {len(train_neg_links)}")
print(f"Negative val links: {len(val_neg_links)}")
print(f"Negative test links: {len(test_neg_links)}")

# ============================================
# MODEL INITIALIZATION
# ============================================

types = graph.get_types()
num_types = len(types)

meta_rels = graph.get_meta_graph()
num_relations = len(meta_rels) + 1

print("\nNode types:", types)
print("Number of node types:", num_types)
print("Meta relations:", meta_rels)
print("Number of relations:", num_relations)

in_dim = 5
hidden_dim = 64
n_heads = 4
n_layers = 2
dropout = 0.2

gnn = GNN(
    in_dim=in_dim,
    n_hid=hidden_dim,
    num_types=num_types,
    num_relations=num_relations,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    conv_name='hgt',
    prev_norm=False,
    last_norm=False,
    use_RTE=True
).to(device)

# Disease classifier
class MultilabelClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)

clf = MultilabelClassifier(hidden_dim, num_diseases).to(device)

# Link prediction matcher
matcher = Matcher(hidden_dim).to(device)

# Optimizer for all components
params = list(gnn.parameters()) + list(clf.parameters()) + list(matcher.parameters())
optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-5)

# Loss functions
disease_criterion = nn.BCEWithLogitsLoss()
link_criterion = nn.BCEWithLogitsLoss()

# Test recommender
recommender = TestRecommender(matcher, confidence_threshold=0.7)

# ============================================
# TRAINING FUNCTIONS
# ============================================

def get_embeddings_for_nodes(node_indices, node_type_name, graph, time_range, sampled_depth=2, sampled_number=8):
    """
    Get embeddings for specific nodes from the GNN.
    
    Args:
        node_indices: List of node indices
        node_type_name: Type of nodes (e.g., 'patient', 'lab_test')
        graph: The heterogeneous graph
        time_range: Time range for sampling
        sampled_depth: Depth of subgraph sampling
        sampled_number: Number of neighbors to sample
    
    Returns:
        embeddings: Tensor of embeddings [num_nodes, hidden_dim]
    """
    inp = {
        node_type_name: [(int(nid), 0) for nid in node_indices]
    }
    
    feature, times, edge_list, indxs, texts = sample_subgraph(
        graph,
        time_range=time_range,
        sampled_depth=sampled_depth,
        sampled_number=sampled_number,
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
    
    with torch.set_grad_enabled(True):
        all_embs = gnn(node_feature, node_type, edge_time, edge_index, edge_type)
    
    node_offset, node_type_id = node_dict[node_type_name]
    local_node_ids = indxs[node_type_name]
    
    # Map requested indices to local indices
    nid_to_local = {int(nid): i for i, nid in enumerate(local_node_ids)}
    
    selected_global_indices = []
    for nid in node_indices:
        if nid in nid_to_local:
            local_id = nid_to_local[nid]
            global_node_idx = node_offset + local_id
            selected_global_indices.append(global_node_idx)
    
    if len(selected_global_indices) == 0:
        return None
    
    selected_global_indices = torch.LongTensor(selected_global_indices).to(device)
    embeddings = all_embs[selected_global_indices]
    
    return embeddings

def train_link_prediction_batch(links, is_positive, batch_size=32):
    """
    Train on a batch of links for link prediction.
    
    Args:
        links: List of (patient_idx, test_idx) tuples
        is_positive: Boolean indicating if these are positive examples
        batch_size: Batch size for training
    
    Returns:
        loss: Average loss for this batch
    """
    if len(links) == 0:
        return 0.0
    
    # Randomly sample a batch
    indices = np.random.choice(len(links), min(batch_size, len(links)), replace=False)
    batch_links = [links[i] for i in indices]
    
    patient_indices = [link[0] for link in batch_links]
    test_indices = [link[1] for link in batch_links]
    
    # Get embeddings
    time_range = {max_rel_time: True}
    patient_embs = get_embeddings_for_nodes(patient_indices, "patient", graph, time_range)
    test_embs = get_embeddings_for_nodes(test_indices, "lab_test", graph, time_range)
    
    if patient_embs is None or test_embs is None:
        return 0.0
    
    # Compute link scores
    scores = matcher(patient_embs, test_embs, infer=False, pair=True)
    
    # Labels
    labels = torch.ones(len(batch_links), device=device) if is_positive else torch.zeros(len(batch_links), device=device)
    
    # Loss
    loss = link_criterion(scores, labels)
    
    return loss

def evaluate_link_prediction(links_pos, links_neg, batch_size=64):
    """
    Evaluate link prediction performance.
    
    Args:
        links_pos: Positive links
        links_neg: Negative links
        batch_size: Batch size
    
    Returns:
        metrics: Dict with AUC, AP, and accuracy
    """
    gnn.eval()
    matcher.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        # Evaluate positive links
        for i in range(0, len(links_pos), batch_size):
            batch = links_pos[i:i+batch_size]
            patient_indices = [link[0] for link in batch]
            test_indices = [link[1] for link in batch]
            
            time_range = {max_rel_time: True}
            patient_embs = get_embeddings_for_nodes(patient_indices, "patient", graph, time_range)
            test_embs = get_embeddings_for_nodes(test_indices, "lab_test", graph, time_range)
            
            if patient_embs is None or test_embs is None:
                continue
            
            scores = matcher(patient_embs, test_embs, infer=False, pair=True)
            scores = torch.sigmoid(scores).cpu().numpy()
            
            all_scores.extend(scores.tolist())
            all_labels.extend([1] * len(scores))
        
        # Evaluate negative links
        for i in range(0, len(links_neg), batch_size):
            batch = links_neg[i:i+batch_size]
            patient_indices = [link[0] for link in batch]
            test_indices = [link[1] for link in batch]
            
            time_range = {max_rel_time: True}
            patient_embs = get_embeddings_for_nodes(patient_indices, "patient", graph, time_range)
            test_embs = get_embeddings_for_nodes(test_indices, "lab_test", graph, time_range)
            
            if patient_embs is None or test_embs is None:
                continue
            
            scores = matcher(patient_embs, test_embs, infer=False, pair=True)
            scores = torch.sigmoid(scores).cpu().numpy()
            
            all_scores.extend(scores.tolist())
            all_labels.extend([0] * len(scores))
    
    if len(all_scores) == 0:
        return {"auc": 0.0, "ap": 0.0, "acc": 0.0}
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    acc = np.mean((all_scores > 0.5) == all_labels)
    
    return {"auc": auc, "ap": ap, "acc": acc}

# ============================================
# TRAINING LOOP
# ============================================

print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

num_epochs = 20
batch_size = 32

training_history = {
    "epochs": [],
    "train_link_loss": [],
    "val_link_metrics": [],
    "test_link_metrics": []
}

for epoch in range(1, num_epochs + 1):
    gnn.train()
    matcher.train()
    clf.train()
    
    # Train on link prediction
    epoch_link_losses = []
    
    # Number of batches per epoch
    num_batches = max(len(train_links) // batch_size, 1)
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        
        # Train on positive links
        loss_pos = train_link_prediction_batch(train_links, is_positive=True, batch_size=batch_size//2)
        
        # Train on negative links
        loss_neg = train_link_prediction_batch(train_neg_links, is_positive=False, batch_size=batch_size//2)
        
        # Combined loss
        if isinstance(loss_pos, torch.Tensor) and isinstance(loss_neg, torch.Tensor):
            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()
            epoch_link_losses.append(loss.item())
        elif isinstance(loss_pos, torch.Tensor):
            loss_pos.backward()
            optimizer.step()
            epoch_link_losses.append(loss_pos.item())
        elif isinstance(loss_neg, torch.Tensor):
            loss_neg.backward()
            optimizer.step()
            epoch_link_losses.append(loss_neg.item())
    
    # Evaluate
    avg_train_loss = np.mean(epoch_link_losses) if epoch_link_losses else 0.0
    
    val_metrics = evaluate_link_prediction(val_links, val_neg_links, batch_size=64)
    
    print(f"\nEpoch {epoch}/{num_epochs}")
    print(f"  Train Link Loss: {avg_train_loss:.4f}")
    print(f"  Val Link AUC: {val_metrics['auc']:.4f}, AP: {val_metrics['ap']:.4f}, Acc: {val_metrics['acc']:.4f}")
    
    # Save history
    training_history["epochs"].append(epoch)
    training_history["train_link_loss"].append(avg_train_loss)
    training_history["val_link_metrics"].append(val_metrics)

# Final test evaluation
print("\n" + "="*50)
print("FINAL TEST EVALUATION")
print("="*50)

test_metrics = evaluate_link_prediction(test_links, test_neg_links, batch_size=64)
print(f"Test Link AUC: {test_metrics['auc']:.4f}")
print(f"Test Link AP: {test_metrics['ap']:.4f}")
print(f"Test Link Acc: {test_metrics['acc']:.4f}")

training_history["test_link_metrics"] = test_metrics

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

save_dir = "models_saved"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "test_recommendation_model.pth")
torch.save({
    "gnn_state_dict": gnn.state_dict(),
    "matcher_state_dict": matcher.state_dict(),
    "clf_state_dict": clf.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "hidden_dim": hidden_dim,
    "num_diseases": num_diseases,
    "num_types": num_types,
    "num_relations": num_relations,
    "in_dim": in_dim,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "training_history": training_history
}, model_path)

print(f"Model saved to: {model_path}")

# Save training history
history_path = os.path.join(save_dir, "test_recommendation_history.json")
with open(history_path, "w") as f:
    # Convert numpy types to Python types for JSON serialization
    history_json = {
        "epochs": training_history["epochs"],
        "train_link_loss": [float(x) for x in training_history["train_link_loss"]],
        "val_link_metrics": [
            {k: float(v) for k, v in metrics.items()} 
            for metrics in training_history["val_link_metrics"]
        ],
        "test_link_metrics": {k: float(v) for k, v in training_history["test_link_metrics"].items()}
    }
    json.dump(history_json, f, indent=2)

print(f"Training history saved to: {history_path}")

print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
