import os
import numpy as np
import pandas as pd
from collections import defaultdict

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics

from pyHGT.data import Graph, sample_subgraph, to_torch
from pyHGT.model import GNN, Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

#temporal processing
patient_tests["report_date"] = pd.to_datetime(patient_tests["report_date"])
patient_tests["time_idx"] = patient_tests["report_date"].astype("int64") // 10**9  # convert to unix timestamp in seconds

min_time = patient_tests["time_idx"].min()
patient_tests["rel_time"] = (patient_tests["time_idx"]-min_time)// 86400  # convert to days

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

print("example lab info:", list(lab_info.items())[:3]) # only show 3 examples


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
        return 0.0  # cannot determine abnormality
    return float((v - low) / (high - low))


pt_merged["abnormality"] = pt_merged.apply(compute_abnoramability, axis=1)

# aggregate per person
agg = pt_merged.groupby("patient_id").agg(
    num_tests = ("test_name", "count"),
    mean_abn = ("abnormality", "mean"),
    max_abn = ("abnormality", "max"),
    min_abn = ("abnormality", "min"),
    last_time = ("rel_time", "max")
).reset_index()

patient_feat_df = agg.set_index("patient_id")
patient_feat_df.head()

# extract unique node ids
patient_ids = sorted(patient_tests["patient_id"].astype(str).unique().tolist())
lab_tests = sorted(list(lab_info.keys()))

all_organs = sorted({org for info in lab_info.values() for org in info["organs"] if org})
all_diseases = sorted({dis for info in lab_info.values() for dis in info["diseases"] if dis})

print("#patients =", len(patient_ids))
print("#labs =", len(lab_tests))
print("#organs =", len(all_organs))
print("#diseases =", len(all_diseases))

print("#nodes = ", len(patient_ids) + len(lab_tests) + len(all_organs) + len(all_diseases))

#build pyHGT graph - add nodes

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

print("Example patient2idx:", list(lab2idx.items())[:5])

#build pyHGT graph - add edges with temporal info

for _, row in patient_tests.iterrows():
    pid = str(row["patient_id"])
    test = str(row["test_name"])
    t = int(row["rel_time"])

    if pid in patient2idx and test in lab2idx: # src and dst nodes exist
        # patient - lab_test edge
        graph.add_edge(
            {"type": "patient", "id": pid},
            {"type": "lab_test", "id": test},
            time=t,
            relation_type="had_test",
            directed=True,
        )

for test, info in lab_info.items():

    # lab_test - organ edges
    for organ in info["organs"]:
        if organ in organ2idx:
            graph.add_edge(
                {"type": "lab_test", "id": test},
                {"type": "organ", "id": organ},
                relation_type="tests_organ",
                directed=True,
                time=0
            )

    # lab_test - disease edges
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

#node features

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

    # for t in layer_data:
    #     if len(layer_data[t]) == 0:
    #         continue

    all_types = graph.node_feature.keys()

    for t in layer_data:
        if len(layer_data[t]) == 0:
            # feature[t] = np.zeros((0,5), dtype=np.float32)  # empty features
            # times[t] = np.array((0,), dtype=np.int32)
            # indxs[t] = np.array((0,), dtype=np.int32)
            continue

        idxs = np.array(list(layer_data[t].keys()))
        tims = np.array(list(layer_data[t].values()))[:,1]

        df = graph.node_feature[t]
        feats = np.zeros((len(idxs),5), dtype=np.float32)  # 5 features per node

        if t == "patient":

            cols = ["num_tests", "mean_abn", "max_abn", "min_abn", "last_time"]
            vals = df.loc[idxs, cols].fillna(0).values.astype(np.float32)
            feats = vals # all 5 features

        elif t == "lab_test":
            for c in ["low", "high"]:
                if c not in df.columns:
                    df[c] = 0.0
            vals = df.loc[idxs, ["low", "high"]].fillna(0).values.astype(np.float32)
            feats[:,0:2] = vals  # first 2 features and other 3 are zeros

        else:
            pass  # no features for organ and disease nodes

        feature[t] = feats
        times[t] = tims
        indxs[t] = idxs

    return feature, times, indxs, texts 


disease_label_cols = [c for c in labels_df.columns if c != "patient_id"]
num_diseases = len(disease_label_cols)

print("Label columns:", disease_label_cols)
print("Number of diseases to predict:", num_diseases)

num_patients_in_graph = len(patient2idx)
Y = torch.zeros((num_patients_in_graph, num_diseases), dtype=torch.float32)

for _, row in labels_df.iterrows():
    pid = str(row["patient_id"])
    if pid in patient2idx:
        idx = patient2idx[pid]
        Y[idx] = torch.tensor(row[disease_label_cols].values, dtype=torch.float32)

print("Labels tensor shape:", Y.shape)

types = graph.get_types()
num_types = len(types)

meta_rels = graph.get_meta_graph()
num_relations = len(meta_rels) + 1

print("Node types:", types)
print("Number of node types:", num_types)
print("Meta relations:", meta_rels)
print("Number of relations:", num_relations)

in_dim = 5
hidden_dim = 64
n_heads = 4
n_layers  =2
dropout   = 0.2

num_diseases = Y.shape[1]  # number of disease classes

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

## Multi-label classifier
class MultilabelClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)
    
clf = MultilabelClassifier(hidden_dim, num_diseases).to(device)

params = list(gnn.parameters()) + list(clf.parameters())

optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-5)

criterion = nn.BCEWithLogitsLoss() # use cross-entropy if accuracy not enough

# Train/test/val split

all_patient_indices = np.array(list(graph.node_forward["patient"].values())) #defined above also. incase of errors remove this line if needed
np.random.shuffle(all_patient_indices)

N = len(all_patient_indices)
n_train = int(0.7 * N)
n_val   = int(0.15 * N)

train_idx = all_patient_indices[:n_train]
val_idx   = all_patient_indices[n_train:n_train+n_val]
test_idx  = all_patient_indices[n_train+n_val:]

print("#train =", len(train_idx))
print("#val   =", len(val_idx))
print("#test  =", len(test_idx))


def get_patient_batch_embeddings(batch_pids, graph, time_range, sampled_depth=2, sampled_number=8):
    
    inp = {
        "patient": [(int(pid),0) for pid in batch_pids]
    }

    feature, times, edge_list, indxs, texts = sample_subgraph(
        graph,
        time_range={max_rel_time: True},
        sampled_depth=2,
        sampled_number=8,
        inp = inp,
        feature_extractor=feature_medical
    )

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature,times,edge_list, graph
    )

    node_feature = node_feature.to(device)
    node_type = node_feature.to(device)
    edge_time = node_feature.to(device)
    edge_index = node_feature.to(device)
    edge_type = node_feature.to(device)

    with torch.set_grad_enabled(True):
        all_embs = gnn(node_feature, node_type, edge_time, edge_index, edge_type)

    patient_offset, patient_type_id =node_dict["patient"]

    local_patient_ids = indxs["patient"]
    pid_to_local = {int(pid): i for i, pid in enumerate(local_patient_ids)}


    selected_global_indices = []
    for pid in batch_pids:
        if pid in pid_to_local:
            local_id = pid_to_local[pid]
            global_node_idx = patient_offset + local_id
            selected_global_indices.append(global_node_idx)

        else:
            pass

    if len(selected_global_indices) == 0:

        return None, None
    
    selected_global_indices = torch.LongTensor(selected_global_indices).to(device)
    batch_embs = all_embs[selected_global_indices]

    batch_labels = Y[batch_pids]
    batch_labels = batch_labels.to(device)

    return batch_embs, batch_labels


#time range define
from sklearn.metrics import f1_score


time_range={max_rel_time: True}

#Train/val loop

def evaluate(split_idx, graph, time_range, batch_size=64):
    gnn.eval()
    clf.eval()
    losses = []
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i in range(0, len(split_idx), batch_size):
            batch_pids = split_idx[i:i+batch_size]
            batch_embs, batch_labels = get_patient_batch_embeddings(
                batch_pids, graph, time_range
            )

            if batch_embs is None:
                continue

            logits = clf(batch_embs)
            loss = criterion(logits, batch_labels)

            losses.append(loss.item())
            trues = batch_labels.cpu().numpy()

            all_preds.append(torch.sigmoid(logits).cpu())
            all_trues.append(batch_labels.cpu())
    
    if len(losses) == 0:
        return None, None
    

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    f1 = f1_score(all_trues.flatten(), all_preds.flatten(), zero_division=0)

    return float(np.mean(losses)), float(f1)

num_epochs = 20
batch_size = 64
sampled_depth=2
sampled_number=8

train_idx_arr = np.array(train_idx)
val_idx_arr = np.array(val_idx)
test_idx_arr = np.array(test_idx)

for epoch in range(1, num_epochs+1):
    gnn.train()
    clf.train()

    perm = np.random.permutation(len(train_idx_arr))
    train_idx_arr_shuffled = train_idx_arr[perm]

    epoch_losses = []

    for i in range(0, len(train_idx_arr), batch_size):
        batch_pids = train_idx_arr_shuffled[i:i+batch_size]

        optimizer.zero_grad()

        batch_embs, batch_labels = get_patient_batch_embeddings(
            batch_pids, graph, time_range,
            sampled_depth=sampled_depth,
            sampled_number=sampled_number
        )

        if batch_embs is None:
            continue
        
        logits = clf(batch_embs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    train_loss = np.mean(epoch_losses) if epoch_losses else None
    val_loss, val_f1 = evaluate(val_idx_arr, graph, time_range, batch_size)

    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}")

test_loss, test_f1 = evaluate(test_idx_arr, graph, time_range, batch_size=64)
print(f"Test Loss = {test_loss:.4f}, Test F1 = {test_f1:.4f}")