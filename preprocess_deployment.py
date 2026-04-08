import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------- CONFIG ----------------------
DATA_INPUT_DIR = "data"
TRAIN_CSV = os.path.join(DATA_INPUT_DIR, "train.csv")

OUTPUT_DIR = "sampled_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling sizes per class (adjust)
N_PER_LABEL_TRAIN = 2000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

TFIDF_MAX_FEATURES = 5000
K_NEIGHBORS = 10

# Filenames
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_encoder_mapping.joblib")
GRAPH_PLAIN_PATH = os.path.join(OUTPUT_DIR, "graph_data_plain.pt")

# ---------- Utilities ----------
def find_label_column(df):
    possible = ['label', 'labels', 'class', 'classes', 'target', 'targets']
    for col in df.columns:
        if col.lower() in possible:
            return col
    # Assume last column
    return df.columns[-1]

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenize (simple split)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Remove non-alpha
    tokens = [t for t in tokens if t.isalpha()]
    return " ".join(tokens)

# ---------- Main ----------
print("Loading data...")
df = pd.read_csv(TRAIN_CSV)
label_col = find_label_column(df)
print(f"Detected label column: {label_col}")

# Sample
sampled = df.groupby(label_col).apply(lambda x: x.sample(min(len(x), N_PER_LABEL_TRAIN), random_state=RANDOM_STATE)).reset_index(drop=True)
print(f"Sampled {len(sampled)} training samples")

# Clean text
text_cols = [col for col in df.columns if col != label_col]
text_col = text_cols[0] if text_cols else df.columns[0]
sampled['text'] = sampled[text_col].apply(clean_text)
print("Text cleaned")

# Label mapping
unique_raw_labels = list(sampled[label_col].unique())
unique_sorted = sorted(unique_raw_labels, key=lambda x: int(x) if x.isdigit() else x)
label2int = {lab: idx for idx, lab in enumerate(unique_sorted)}
int2label = {v: k for k, v in label2int.items()}
sampled['label_encoded'] = sampled[label_col].map(label2int)

# Save mapping
joblib.dump({"label2int": label2int, "int2label": int2label}, LABEL_MAP_PATH)
print("Saved label mapping")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train = vectorizer.fit_transform(sampled['text'].values)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("Saved TF-IDF vectorizer")

X_train_arr = X_train.toarray().astype(np.float32)

# k-NN graph
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='cosine', n_jobs=-1)
knn.fit(X_train_arr)
_, indices = knn.kneighbors(X_train_arr, return_distance=True)

edge_u = []
edge_v = []
n_nodes = X_train_arr.shape[0]
for i in range(n_nodes):
    neighs = indices[i, 1:]
    for j in neighs:
        edge_u.append(i)
        edge_v.append(int(j))
        edge_u.append(int(j))
        edge_v.append(i)

edge_index = np.vstack([edge_u, edge_v]).astype(np.int64)

# Dedupe
pairs = set()
u_unique = []
v_unique = []
for a,b in zip(edge_index[0], edge_index[1]):
    if (a,b) not in pairs:
        pairs.add((a,b))
        u_unique.append(a)
        v_unique.append(b)
edge_index = np.vstack([u_unique, v_unique]).astype(np.int64)

print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges")

# Save plain graph (only x and edge_index for prediction)
plain = {
    "x": torch.from_numpy(X_train_arr),
    "edge_index": torch.from_numpy(edge_index)
}
torch.save(plain, GRAPH_PLAIN_PATH)
print("Saved graph data")

print("Preprocessing for deployment complete.")