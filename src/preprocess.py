import os
import re
import string
import warnings
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------- CONFIG ----------------------
DATA_INPUT_DIR = "data"
TRAIN_CSV = os.path.join(DATA_INPUT_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_INPUT_DIR, "test.csv")

OUTPUT_DIR = "sampled_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling sizes per class (adjust)
N_PER_LABEL_TRAIN = 2000
N_PER_LABEL_TEST  = 1000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

TFIDF_MAX_FEATURES = 5000
K_NEIGHBORS = 10

# Filenames
SAMP_TRAIN_CLEAN = os.path.join(OUTPUT_DIR, "sampled_train_cleaned.csv")
SAMP_TEST_CLEAN  = os.path.join(OUTPUT_DIR, "sampled_test_cleaned.csv")
TFIDF_TRAIN_CSV  = os.path.join(OUTPUT_DIR, "sampled_train_tfidf.csv")
TFIDF_TEST_CSV   = os.path.join(OUTPUT_DIR, "sampled_test_tfidf.csv")
VECTORIZER_PATH  = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
LABEL_MAP_PATH   = os.path.join(OUTPUT_DIR, "label_encoder_mapping.joblib")
GRAPH_PLAIN_PATH = os.path.join(OUTPUT_DIR, "graph_data_plain.pt")
PIPELINE_METADATA_PATH = os.path.join(OUTPUT_DIR, "pipeline_metadata.pkl")
# ----------------------------------------------------

# ---------- Utilities ----------
def find_label_column(df):
    """Try common label column names, otherwise pick a numeric-like column with small unique count."""
    candidates = ['label','Label','target','class','category','y']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose a column with numeric-like dtype or small unique count
    for c in df.columns:
        nunique = df[c].nunique()
        if 2 <= nunique <= 100 and df[c].dtype != object:
            return c
    # fallback: choose first object column that has small unique count
    for c in df.columns:
        nunique = df[c].nunique()
        if 2 <= nunique <= 100 and df[c].dtype == object:
            return c
    raise ValueError("Could not auto-detect a label column. Rename it to 'label' or 'target' or similar.")

# ensure nltk resources
def init_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

init_nltk()
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    s = str(text)
    s = re.sub(r'[^\x00-\x7F]+',' ', s)  # remove non-ascii
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = [lemmatizer.lemmatize(tok) for tok in s.split() if tok and tok not in STOPWORDS]
    return ' '.join(tokens)

# ---------- Load CSVs ----------
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Training CSV not found at {TRAIN_CSV}")
if not os.path.exists(TEST_CSV):
    warnings.warn(f"Test CSV not found at {TEST_CSV} — continuing without test sampling.")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV) if os.path.exists(TEST_CSV) else None

# ---------- Detect label column ----------
label_col = find_label_column(train_df)
print("Detected label column:", label_col)

# keep a raw copy for debugging and consistent processing
train_df['label_raw'] = train_df[label_col].astype(str).str.strip()
if test_df is not None:
    test_df['label_raw'] = test_df[label_col].astype(str).str.strip()

# ---------- Stratified sampling ----------
def stratified_sample(df, label_raw_col, n_per_label, random_state=RANDOM_STATE):
    groups = df.groupby(label_raw_col, group_keys=False)
    sampled = []
    insufficient = {}
    for label, g in groups:
        count = len(g)
        if count >= n_per_label:
            sampled.append(g.sample(n=n_per_label, random_state=random_state))
        else:
            sampled.append(g)
            need = n_per_label - count
            if count == 0:
                insufficient[label] = 0
            else:
                up = g.sample(n=need, replace=True, random_state=random_state)
                sampled.append(up)
                insufficient[label] = count
    if len(sampled) == 0:
        return pd.DataFrame(columns=df.columns), insufficient
    result = pd.concat(sampled).reset_index(drop=True)
    return result, insufficient

print("Sampling train:", N_PER_LABEL_TRAIN, "per label")
train_sampled, train_insuff = stratified_sample(train_df, 'label_raw', N_PER_LABEL_TRAIN)
if train_insuff:
    print("Warning: some labels had fewer than requested and were upsampled:", train_insuff)

if test_df is not None:
    print("Sampling test:", N_PER_LABEL_TEST, "per label")
    test_sampled, test_insuff = stratified_sample(test_df, 'label_raw', N_PER_LABEL_TEST)
    if test_insuff:
        print("Warning: some test labels had fewer than requested and were upsampled:", test_insuff)
else:
    test_sampled = None

# ---------- Detect & combine text columns ----------
def detect_text_columns(df):
    names = [c.lower() for c in df.columns]
    if 'text' in names:
        # return the original column name matching 'text'
        return df.columns[names.index('text')]
    title = None; content = None
    for c in df.columns:
        lc = c.lower()
        if lc in ('title','headline','subject'):
            title = c
        if lc in ('content','body','article','description'):
            content = c
    if title and content:
        return (title, content)
    # fallback: first object/string column that is not 'label_raw' or similar
    for c in df.columns:
        if df[c].dtype == object and c not in ('label_raw', label_col):
            return c
    raise ValueError("Could not detect text columns. Ensure CSV contains title/content/text columns.")

def make_text_series(df):
    cols = detect_text_columns(df)
    if isinstance(cols, (tuple, list)):
        tcol, ccol = cols
        return (df[tcol].fillna('') + ' ' + df[ccol].fillna('')).astype(str)
    else:
        return df[cols].fillna('').astype(str)

print("Detecting text columns and cleaning text (this may take a while)...")
train_text_series = make_text_series(train_sampled)
train_sampled['text'] = train_text_series.apply(clean_text)

if test_sampled is not None:
    test_text_series = make_text_series(test_sampled)
    test_sampled['text'] = test_text_series.apply(clean_text)

train_sampled.to_csv(SAMP_TRAIN_CLEAN, index=False)
print("Saved cleaned train sample to:", SAMP_TRAIN_CLEAN)
if test_sampled is not None:
    test_sampled.to_csv(SAMP_TEST_CLEAN, index=False)
    print("Saved cleaned test sample to:", SAMP_TEST_CLEAN)

# ---------- Create corrected label mapping ----------
# Unique raw labels from the sampled training set:
unique_raw_labels = list(train_sampled['label_raw'].unique())
print("Unique raw labels (examples):", unique_raw_labels[:30])
# Check if all unique labels are integer-like (e.g., '0','1','10')
def all_int_like(vals):
    try:
        for v in vals:
            int(v)
        return True
    except Exception:
        return False

if all_int_like(unique_raw_labels):
    # sort numerically by integer value
    unique_sorted = sorted(unique_raw_labels, key=lambda x: int(x))
    print("Labels appear numeric-like — sorting numerically.")
else:
    unique_sorted = sorted(unique_raw_labels)
    print("Labels are non-numeric — sorting lexically.")

# Map original raw label string -> integer ID 0..C-1 in sorted order
label2int = {lab: idx for idx, lab in enumerate(unique_sorted)}
int2label = {v: k for k, v in label2int.items()}
print("Final label mapping (label_raw -> int):", label2int)

# Save mapping
joblib.dump({"label2int": label2int, "int2label": int2label}, LABEL_MAP_PATH)
print("Saved label mapping to:", LABEL_MAP_PATH)

# Apply mapping to sampled dataframes
train_sampled['label_encoded'] = train_sampled['label_raw'].map(label2int).astype(int)
if test_sampled is not None:
    # Map test labels; unseen test labels -> -1
    test_sampled['label_encoded'] = test_sampled['label_raw'].map(label2int)
    unseen = test_sampled['label_encoded'].isna().sum()
    if unseen > 0:
        print(f"Warning: {unseen} test rows have labels unseen in train. Setting to -1.")
        test_sampled['label_encoded'] = test_sampled['label_encoded'].fillna(-1).astype(int)

# ---------- TF-IDF fit on train.text ----------
print(f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}) on training text...")
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train = vectorizer.fit_transform(train_sampled['text'].values)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("Saved TF-IDF vectorizer to:", VECTORIZER_PATH)

X_train_arr = X_train.toarray().astype(np.float32)
tfidf_train_df = pd.DataFrame(X_train_arr, columns=vectorizer.get_feature_names_out())
tfidf_train_df['label'] = train_sampled['label_encoded'].values
tfidf_train_df.to_csv(TFIDF_TRAIN_CSV, index=False)
print("Saved train TF-IDF CSV to:", TFIDF_TRAIN_CSV)

if test_sampled is not None:
    X_test = vectorizer.transform(test_sampled['text'].values)
    X_test_arr = X_test.toarray().astype(np.float32)
    tfidf_test_df = pd.DataFrame(X_test_arr, columns=vectorizer.get_feature_names_out())
    if 'label_raw' in test_sampled.columns:
        tfidf_test_df['label_raw'] = test_sampled['label_raw'].values
    tfidf_test_df['label_encoded'] = test_sampled['label_encoded'].values
    tfidf_test_df.to_csv(TFIDF_TEST_CSV, index=False)
    print("Saved test TF-IDF CSV to:", TFIDF_TEST_CSV)

# ---------- Diagnostics ----------
n_classes = len(label2int)
print("Number of classes in training (n_classes):", n_classes)
if n_classes <= 1:
    print("ERROR: <=1 class detected. Dumping label details for debugging:")
    print(train_sampled[['label_raw','label_encoded']].head(50))
    raise ValueError("Training contains <=1 class. Check input labels and detection.")

# ---------- Build k-NN graph (cosine) from training TF-IDF features ----------
print(f"Building k-NN graph (K={K_NEIGHBORS}) from training features...")
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='cosine', n_jobs=-1)
knn.fit(X_train_arr)
_, indices = knn.kneighbors(X_train_arr, return_distance=True)

edge_u = []
edge_v = []
n_nodes = X_train_arr.shape[0]
for i in range(n_nodes):
    neighs = indices[i, 1:]  # skip self
    for j in neighs:
        edge_u.append(i); edge_v.append(int(j))
        edge_u.append(int(j)); edge_v.append(i)

edge_index = np.vstack([edge_u, edge_v]).astype(np.int64)

# dedupe
pairs = set()
u_unique = []
v_unique = []
for a,b in zip(edge_index[0], edge_index[1]):
    if (a,b) not in pairs:
        pairs.add((a,b))
        u_unique.append(a); v_unique.append(b)
edge_index = np.vstack([u_unique, v_unique]).astype(np.int64)
print("Num nodes:", n_nodes, "Num edges:", edge_index.shape[1])

# ---------- Create stratified masks within sampled training set ----------
all_idx = np.arange(n_nodes)
train_idx, testval_idx, y_train_enc, y_testval_enc = train_test_split(
    all_idx, train_sampled['label_encoded'].values, test_size=0.30,
    stratify=train_sampled['label_encoded'].values, random_state=RANDOM_STATE
)
val_idx, test_idx, _, _ = train_test_split(testval_idx, y_testval_enc, test_size=0.5,
                                           stratify=y_testval_enc, random_state=RANDOM_STATE)

train_mask = torch.zeros(n_nodes, dtype=torch.bool); train_mask[train_idx] = True
val_mask = torch.zeros(n_nodes, dtype=torch.bool); val_mask[val_idx] = True
test_mask = torch.zeros(n_nodes, dtype=torch.bool); test_mask[test_idx] = True

# ---------- Save plain graph dict ----------
plain = {
    "x": torch.from_numpy(X_train_arr),                    # [n_nodes, n_features]
    "edge_index": torch.from_numpy(edge_index),            # [2, n_edges]
    "y": torch.from_numpy(train_sampled['label_encoded'].values).long(),
    "train_mask": train_mask,
    "val_mask": val_mask,
    "test_mask": test_mask
}
torch.save(plain, GRAPH_PLAIN_PATH)
print("Saved plain graph data to:", GRAPH_PLAIN_PATH)

pipeline_metadata = {
    "version": "1.0",
    "data_dir": OUTPUT_DIR,
    "train_samples": int(train_mask.sum().item()),
    "val_samples": int(val_mask.sum().item()),
    "test_samples": int(test_mask.sum().item()),
    "n_nodes": n_nodes,
    "n_features": X_train_arr.shape[1],
    "n_classes": n_classes,
    "k_neighbors": K_NEIGHBORS,
    "tfidf_max_features": TFIDF_MAX_FEATURES,
    "label2int": label2int,
    "int2label": int2label,
    "graph_edge_count": int(edge_index.shape[1]),
    "random_state": RANDOM_STATE,
}
with open(PIPELINE_METADATA_PATH, "wb") as f:
    pickle.dump(pipeline_metadata, f)
print("Saved pipeline metadata to:", PIPELINE_METADATA_PATH)

# Save cleaned sampled CSVs again with label_encoded (already saved earlier but update)
train_sampled.to_csv(SAMP_TRAIN_CLEAN, index=False)
print("Saved final cleaned sampled train CSV to:", SAMP_TRAIN_CLEAN)
if test_sampled is not None:
    test_sampled.to_csv(SAMP_TEST_CLEAN, index=False)
    print("Saved final cleaned sampled test CSV to:", SAMP_TEST_CLEAN)

print("Preprocessing pipeline finished successfully.")