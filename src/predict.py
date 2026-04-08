import os
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# ---------- User settings ----------
DATA_DIR = "sampled_data"
PLAIN_PATH = os.path.join(DATA_DIR, "graph_data_plain.pt")
MODEL_PATH = os.path.join(DATA_DIR, "graphsage_model.pth")
VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_encoder_mapping.joblib")  # optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DBPEDIA_14_NAMES = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
    "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
    "Plant", "Album", "Film", "WrittenWork"
]

# GraphSAGE hyperparameters used at training: adapt if different
HIDDEN = 256
NUM_LAYERS = 2
DROPOUT = 0.5

# When connecting new node to existing graph, how many TF-IDF neighbors to attach
K_TFIDF = 10

# When building subgraph for inference, neighbors per layer (match training neighbor sizes if possible)
NEIGHBORS_PER_LAYER = [15, 10]  # example for a 2-layer model

# ----- Helper: GraphSAGE model class (must match training) -----
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            # in -> hidden
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            # hidden -> hidden (num_layers - 2 times)
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # hidden -> out
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            is_last = (i == len(self.convs) - 1)
            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ----- Helper: build adjacency list from edge_index (numpy) -----
def build_adj_list(edge_index_np, num_nodes=None):
    adj = defaultdict(list)
    src = edge_index_np[0]
    dst = edge_index_np[1]
    for u, v in zip(src, dst):
        adj[int(u)].append(int(v))
    if num_nodes is not None:
        # ensure every node has an entry (even if empty)
        for i in range(num_nodes):
            _ = adj[i]
    return adj

# ----- Helper: k-hop neighbor sampling to build a small induced subgraph -----
def build_subgraph_nodes(target_nodes, adj, neighbors_per_layer, seed=None):
    """
    Expand neighborhood around target_nodes layer by layer.
    For each node at each layer, sample up to neighbors_per_layer[layer] neighbors.
    Returns list of nodes in the induced subgraph (unique), and mapping dict old->new indices.
    """
    if seed is not None:
        np.random.seed(seed)
    layers = []
    current = set(int(x) for x in target_nodes)
    layers.append(current)
    all_nodes = set(current)
    for layer_k in neighbors_per_layer:
        next_nodes = set()
        for node in current:
            neighs = adj.get(node, [])
            if len(neighs) == 0:
                continue
            # sample up to layer_k neighbors (no replacement if possible)
            if len(neighs) <= layer_k:
                chosen = neighs
            else:
                chosen = list(np.random.choice(neighs, size=layer_k, replace=False))
            for n in chosen:
                next_nodes.add(int(n))
        if not next_nodes:
            break
        layers.append(next_nodes)
        all_nodes.update(next_nodes)
        current = next_nodes
    # return deterministic order (sorted) for stable indexing
    nodes_sorted = sorted(all_nodes)
    old_to_new = {old: new for new, old in enumerate(nodes_sorted)}
    return nodes_sorted, old_to_new

# ----- Helper: create subgraph edge_index and x tensor based on node list -----
def induced_subgraph(node_list, old_to_new, edge_index_np, x_all_np):
    """
    node_list: list of original node indices included
    old_to_new: mapping original->new index (0..n_sub-1)
    Returns: x_sub (torch.FloatTensor NxD), edge_index_sub (torch.LongTensor 2xE)
    """
    node_set = set(node_list)
    src = edge_index_np[0]
    dst = edge_index_np[1]
    u_sub = []
    v_sub = []
    for u, v in zip(src, dst):
        if (int(u) in node_set) and (int(v) in node_set):
            u_sub.append(old_to_new[int(u)])
            v_sub.append(old_to_new[int(v)])
    if len(u_sub) == 0:
        edge_index_sub = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_sub = torch.tensor([u_sub, v_sub], dtype=torch.long)
    x_sub = torch.from_numpy(x_all_np[np.array(node_list, dtype=int)])
    return x_sub, edge_index_sub

# ----- Main neighbor_batch_predict implementation (uses induced subgraph) -----
def neighbor_batch_predict(model, full_data, target_node_idx, neighbors_per_layer, device="cpu", seed=None):
    """
    model: GraphSAGE model
    full_data: torch_geometric.data.Data with attributes x (NxD) and edge_index (2xE)
    target_node_idx: array-like of node indices in the full graph to predict
    neighbors_per_layer: list of ints for neighbor counts per layer (outer->inner)
    Returns:
      preds: np.array of predicted class indices (len = len(target_node_idx))
      probs: np.array of softmax probability vectors (len x num_classes)
    """
    model.eval()
    x_all_np = full_data.x.cpu().numpy()
    edge_index_np = full_data.edge_index.cpu().numpy().astype(int)
    n_all = x_all_np.shape[0]
    adj = build_adj_list(edge_index_np, num_nodes=n_all)

    preds = []
    probs_list = []
    for t in np.array(target_node_idx).flatten():
        # expand nodes for this target
        nodes, old_to_new = build_subgraph_nodes([int(t)], adj, neighbors_per_layer, seed=seed)
        x_sub, edge_index_sub = induced_subgraph(nodes, old_to_new, edge_index_np, x_all_np)
        # map target index to new index
        target_new_idx = old_to_new[int(t)]
        # move to device
        x_sub = x_sub.to(device)
        edge_index_sub = edge_index_sub.to(device)
        with torch.no_grad():
            out = model(x_sub, edge_index_sub)  # shape: (n_sub, out_dim)
            logits = out[target_new_idx].cpu()
            probs = F.softmax(logits, dim=0).numpy()
            pred_id = int(np.argmax(probs))
        preds.append(pred_id)
        probs_list.append(probs)
    return np.array(preds, dtype=int), np.vstack(probs_list)

# ------------------- Script begins here -------------------
def predict_single(text):
    # basic file checks
    for path in [PLAIN_PATH, MODEL_PATH, VECTORIZER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # load vectorizer and plain graph data
    vec = joblib.load(VECTORIZER_PATH)
    plain = torch.load(PLAIN_PATH, map_location="cpu", weights_only=False)
    if "x" not in plain or "edge_index" not in plain:
        raise ValueError("graph_data_plain.pt must be a dict with keys 'x' and 'edge_index'")

    existing_x = plain["x"].cpu().numpy().astype(np.float32)  # NxD
    edge_index_old = plain["edge_index"].cpu().numpy().astype(int)  # 2xE

    in_channels = existing_x.shape[1]
    num_classes = len(DBPEDIA_14_NAMES)

    # construct model and load state
    model = GraphSAGE(in_channels, HIDDEN, num_classes, num_layers=NUM_LAYERS, dropout=DROPOUT)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    # if state is a dict with keys 'model_state_dict' etc, try to find the right key
    if isinstance(state, dict) and ('state_dict' in state and isinstance(state['state_dict'], dict)):
        sd = state['state_dict']
    elif isinstance(state, dict) and ('model_state_dict' in state and isinstance(state['model_state_dict'], dict)):
        sd = state['model_state_dict']
    else:
        sd = state
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()

    # optional label mapping
    label_map = None
    if os.path.exists(LABEL_MAP_PATH):
        try:
            label_map = joblib.load(LABEL_MAP_PATH)
        except Exception:
            label_map = None

    # convert input text to TF-IDF vector
    x_new = vec.transform([text]).toarray().astype(np.float32)

    # find K nearest TF-IDF neighbors to attach the new node
    knn = NearestNeighbors(n_neighbors=min(K_TFIDF, existing_x.shape[0]), metric='cosine', n_jobs=-1)
    knn.fit(existing_x)
    distances, neigh = knn.kneighbors(x_new, return_distance=True)  # neigh is shape (1,K)
    neigh = neigh[0].astype(int)

    # create new edges connecting new node to these neighbors (bidirectional)
    n_old = existing_x.shape[0]
    new_idx = n_old
    u = []
    v = []
    for nb in neigh:
        u.append(new_idx); v.append(int(nb))
        u.append(int(nb)); v.append(new_idx)

    # Build the augmented feature matrix and edge_index
    all_x = np.vstack([existing_x, x_new]).astype(np.float32)
    u_all = np.concatenate([edge_index_old[0], np.array(u, dtype=int)])
    v_all = np.concatenate([edge_index_old[1], np.array(v, dtype=int)])
    edge_index_new = np.vstack([u_all, v_all]).astype(np.int64)

    temp_data = Data(x=torch.from_numpy(all_x), edge_index=torch.from_numpy(edge_index_new))

    # run neighbor-batch (subgraph) predict on the single new node
    preds, probs = neighbor_batch_predict(
        model,
        temp_data,
        target_node_idx=np.array([new_idx], dtype=int),
        neighbors_per_layer=NEIGHBORS_PER_LAYER,
        device=DEVICE,
        seed=42
    )

    pred_id = int(preds[0])
    prob_vector = probs[0] if probs.size else None

    # final label name (prefer label_map if provided)
    if label_map is not None:
        try:
            pred_name = label_map[pred_id]
        except Exception:
            pred_name = DBPEDIA_14_NAMES[pred_id] if 0 <= pred_id < len(DBPEDIA_14_NAMES) else "UNKNOWN"
    else:
        pred_name = DBPEDIA_14_NAMES[pred_id] if 0 <= pred_id < len(DBPEDIA_14_NAMES) else "UNKNOWN"

    return {
        "class_id": pred_id,
        "class_name": pred_name,
        "probabilities": prob_vector.tolist() if prob_vector is not None else None
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'your text here'")
        sys.exit(1)
    text = sys.argv[1]
    result = predict_single(text)
    print("Prediction (DBpedia-14):")
    print("  class_id   :", result["class_id"])
    print("  class_name :", result["class_name"])
    if result["probabilities"]:
        for i, p in enumerate(result["probabilities"]):
            label = DBPEDIA_14_NAMES[i] if i < len(DBPEDIA_14_NAMES) else str(i)
            print(f"    {i:02d} {label:25s} : {p:.4f}")