import importlib.util
import os
import pickle
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

if importlib.util.find_spec('torch_sparse') is None and importlib.util.find_spec('pyg_lib') is None:
    raise ImportError(
        "NeighborLoader requires either 'torch-sparse' or 'pyg-lib'. "
        "Install the appropriate backend for your PyTorch version. "
        "Example: pip install pyg-lib -f https://data.pyg.org/whl/torch-2.11.0+cpu.html"
    )

try:
    from src.model import GraphSAGE
except ModuleNotFoundError:
    # Allow running with `python src/train.py` from the project root.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import GraphSAGE

import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = "sampled_data"
PLAIN_PATH = os.path.join(DATA_DIR, "graph_data_plain.pt")
MODEL_PATH = os.path.join(DATA_DIR, "graphsage_model.pth")
MODEL_METADATA_PATH = os.path.join(DATA_DIR, "model_metadata.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(PLAIN_PATH):
    raise FileNotFoundError(f"{PLAIN_PATH} not found. Run preprocessing first.")

# Load the plain dict
d = torch.load(PLAIN_PATH, weights_only=False)
x = d["x"]
edge_index = d["edge_index"]
y = d["y"]
train_mask = d["train_mask"]
val_mask = d["val_mask"]
test_mask = d["test_mask"]

# Build PyG Data
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

in_channels = data.num_node_features
num_classes = int(data.y.max().item() + 1)

# Hyperparameters (tweak as needed)
HIDDEN = 256
NUM_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 30
BATCH_SIZE = 512
NEIGHBORS = [15, 10]  # neighbors per layer for NeighborLoader

model = GraphSAGE(in_channels, HIDDEN, num_classes, num_layers=NUM_LAYERS, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).tolist()
train_loader = NeighborLoader(
    data,
    num_neighbors=NEIGHBORS,
    input_nodes=train_idx,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_idx = data.val_mask.nonzero(as_tuple=False).view(-1).tolist()
val_loader = NeighborLoader(
    data,
    num_neighbors=NEIGHBORS,
    input_nodes=val_idx,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()
test_loader = NeighborLoader(
    data,
    num_neighbors=NEIGHBORS,
    input_nodes=test_idx,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            target_n = batch.batch_size
            preds = out[:target_n].argmax(dim=1)
            total_correct += (preds == batch.y[:target_n]).sum().item()
            total_examples += target_n
    return total_correct / total_examples if total_examples > 0 else 0

best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        target_n = batch.batch_size  # nodes with labels in this mini-batch
        loss = criterion(out[:target_n], batch.y[:target_n])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target_n
        total_examples += target_n
    avg_loss = total_loss / (total_examples + 1e-12)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved best model (val {best_val:.4f}) to {MODEL_PATH}")

print("Training complete. Best val:", best_val)

# Save metadata
metadata = {
    'in_channels': in_channels,
    'hidden_channels': HIDDEN,
    'out_channels': num_classes,
    'num_layers': NUM_LAYERS,
    'dropout': 0.5,
    'best_val_acc': best_val,
    'epochs_trained': EPOCHS,
    'lr': LR,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'neighbors': NEIGHBORS
}
with open(MODEL_METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Saved model metadata to {MODEL_METADATA_PATH}")