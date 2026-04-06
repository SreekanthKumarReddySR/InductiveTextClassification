# DBpedia Entity Classification using GCNs and GraphSAGE

A multiclass text classification solution on the **DBpedia-14 Ontology Classification Dataset** using two advanced Graph Neural Network (GNN) architectures:
- **GCN (Graph Convolutional Network)** — Transductive setting
- **GraphSAGE (Graph Sample and Aggregated Embeddings)** — Inductive setting

---

## Table of Contents
1. [Dataset](#dataset)
2. [Data Shapes and Sizes](#data-shapes-and-sizes)
3. [Preprocessing](#preprocessing)
4. [Graph Construction](#graph-construction)
5. [Model Architectures](#model-architectures)
6. [Training](#training)
7. [Results](#results)
8. [Repository Structure](#repository-structure)
9. [Requirements](#requirements)

---

## Dataset

**DBpedia-14 Ontology Classification Dataset**

The DBpedia dataset is derived from Wikipedia articles and organized into 14 non-overlapping ontology classes. Each sample consists of a title and an abstract (content) extracted from a Wikipedia article.

| # | Class Label |
|---|-------------|
| 0 | Company |
| 1 | EducationalInstitution |
| 2 | Artist |
| 3 | Athlete |
| 4 | OfficeHolder |
| 5 | MeanOfTransportation |
| 6 | Building |
| 7 | NaturalPlace |
| 8 | Village |
| 9 | Animal |
| 10 | Plant |
| 11 | Album |
| 12 | Film |
| 13 | WrittenWork |

- **Total Classes:** 14
- **Source:** DBpedia / Hugging Face `datasets` library (`dbpedia_14`)
- **Text Fields Used:** `title` + `content` (concatenated)

---

## Data Shapes and Sizes

### Transductive Setting (GCN)

| Split | Samples per Class | Total Samples |
|-------|-------------------|---------------|
| Train | 2,000 | 28,000 |
| Test | 1,000 | 14,000 |
| **Total Nodes** | — | **42,000** |

- **TF-IDF Feature Matrix:** `(42,000 nodes, 5,000 features)`
- **Node Feature Dimension:** 5,000
- **Graph Edges (k-NN, K=10):** ~438,000 edges
- **Graph Type:** Homogeneous k-NN cosine similarity graph

### Inductive Setting (GraphSAGE)

| Split | Samples per Class | Total Samples |
|-------|-------------------|---------------|
| Train | 2,000 | 28,000 |
| Test | 1,000 | 14,000 |
| **Total Document Nodes** | — | **42,000** |

- **Vocabulary (Word Nodes):** ~5,000 word nodes
- **Total Graph Nodes:** ~47,000 (42,000 docs + 5,000 words)
- **Total Graph Edges:** ~1,670,000
  - Doc→Word edges: TF-IDF weighted
  - Word→Word edges: PMI (Pointwise Mutual Information) weighted
- **Node Feature Dimension:** 5,000 (TF-IDF)

---

## Preprocessing

Both pipelines share the following preprocessing steps:

### 1. Data Loading & Stratified Sampling
- Dataset loaded via Hugging Face `datasets` (`dbpedia_14`)
- **Stratified sampling** applied to ensure class balance:
  - 2,000 samples/class for training
  - 1,000 samples/class for testing

### 2. Text Cleaning
- Concatenate `title` and `content` fields
- Lowercase all text
- Remove punctuation and special characters
- Tokenize using NLTK
- Remove English stopwords (NLTK stopwords corpus)
- Retain only alphabetic tokens

### 3. TF-IDF Vectorization
- Applied `TfidfVectorizer` with `max_features=5000`
- Produces a sparse feature matrix of shape `(N_docs, 5000)`
- Used as node feature matrix for GNN input

---

## Graph Construction

### Transductive Graph (GCN)
- **Algorithm:** k-Nearest Neighbors (k-NN) on TF-IDF cosine similarity
- **K:** 10 neighbors per node
- **Edges:** Undirected cosine similarity edges
- **Total Edges:** ~438,000
- **Node Count:** 42,000 (train + test combined into one graph)
- All nodes are present during training (transductive)

### Inductive / Heterogeneous Graph (GraphSAGE)
- **Type:** Heterogeneous bipartite-style text graph
- **Node Types:**
  - Document nodes: 42,000
  - Word nodes: ~5,000 (top TF-IDF vocabulary terms)
- **Edge Types:**
  - **Doc → Word:** weighted by TF-IDF score
  - **Word → Word:** weighted by Pointwise Mutual Information (PMI) over a sliding window
- **Total Edges:** ~1,670,000
- Test nodes are unseen during training (inductive)

---

## Model Architectures

### GCN — Transductive

```
Input: Node features (N, 5000)
  └─> GCNConv Layer 1: 5000 → 256, ReLU, Dropout(0.5)
  └─> GCNConv Layer 2: 256 → 14 (num_classes)
  └─> Log Softmax → Class Prediction
```

- **Framework:** PyTorch Geometric
- **Optimizer:** Adam
- **Loss:** Negative Log-Likelihood (NLL) / CrossEntropy
- **Dropout:** 0.5

### GraphSAGE — Inductive

```
Input: Node features (N, 5000)
  └─> SAGEConv Layer 1: 5000 → 256, ReLU, Dropout(0.5)
  └─> SAGEConv Layer 2: 256 → 14 (num_classes)
  └─> Log Softmax → Class Prediction
```

- **Framework:** PyTorch Geometric
- **Sampling:** `NeighborLoader` for mini-batch neighbor sampling
- **Aggregation:** Mean aggregation (GraphSAGE default)
- **Optimizer:** Adam
- **Loss:** CrossEntropy
- **Dropout:** 0.5

---

## Training

### Transductive (GCN)
- **Mode:** Full-batch (entire graph in memory)
- **Train Mask:** 28,000 nodes
- **Val/Test Mask:** 14,000 nodes
- **Epochs:** 30
- **Hardware:** GPU (CUDA if available, else CPU)

### Inductive (GraphSAGE)
- **Mode:** Mini-batch with `NeighborLoader`
- **Batch Size:** Configurable per run
- **Neighbor Sampling:** 2-hop neighborhood
- **Train Nodes:** 28,000 document nodes
- **Test Nodes:** 14,000 unseen document nodes
- **Epochs:** 30
- **Hardware:** GPU (CUDA if available, else CPU)

---

## Results

### GCN — Transductive

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **94.93%** |
| Best Epoch | 26 / 30 |
| Node Features | TF-IDF (5000-dim) |
| Graph Edges | ~438,000 (k-NN, K=10) |

### GraphSAGE — Inductive

| Metric | Value |
|--------|-------|
| Setting | Inductive (unseen test nodes) |
| Node Features | TF-IDF (5000-dim) |
| Graph Edges | ~1,670,000 (TF-IDF + PMI) |
| Architecture | 2-layer SAGEConv (256 hidden) |

> Both models demonstrate strong classification performance on the 14-class DBpedia ontology dataset, showcasing the effectiveness of graph-based text representations over traditional flat classifiers.

---

## Repository Structure

```
DbPedia-Enitity-Classification-using-GCNs-and-GraphSage/
├── gcn/
│   └── transductive.ipynb      # GCN transductive classification pipeline
├── GraphSage/
│   └── inductive.ipynb         # GraphSAGE inductive classification pipeline
├── README.md
└── LICENSE
```

---

## Requirements

```
torch
torch-geometric
transformers
datasets
scikit-learn
nltk
numpy
pandas
matplotlib
fastapi
uvicorn
joblib
```

Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) based on your CUDA version.

---

## Deployment with MLOps

This project includes MLOps setup for deploying the inductive GraphSAGE model.

### Project Structure

```
DbPedia-Enitity-Classification-using-GCNs-and-GraphSage/
├── src/
│   ├── model.py          # GraphSAGE model definition
│   ├── preprocess.py     # Data preprocessing and graph construction
│   ├── train.py          # Model training script
│   └── predict.py        # Single prediction script
├── app/
│   └── main.py           # FastAPI application for serving predictions
├── sampled_data/         # Preprocessed data and trained model (generated)
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile            # Containerization
├── requirements.txt      # Python dependencies
└── README.md
```

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and preprocess data:**
   ```bash
   python -c "from datasets import load_dataset; ds = load_dataset('dbpedia_14'); ds['train'].to_csv('data/train.csv', index=False); ds['test'].to_csv('data/test.csv', index=False)"
   mkdir data
   # Download the dataset manually or use the above command
   ```

3. **Run preprocessing:**
   ```bash
   python src/preprocess.py
   ```

4. **Train the model:**
   ```bash
   python src/train.py
   ```

5. **Test prediction locally:**
   ```bash
   python src/predict.py "Your sample text here"
   ```

6. **Run the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://localhost:8000/docs` for interactive API documentation.

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t dbpedia-classifier .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 dbpedia-classifier
   ```

### Cloud Deployment (Azure Example)

The project includes GitHub Actions workflow for automated deployment to Azure Web Apps.

1. **Set up Azure resources:**
   - Create an Azure Container Registry (ACR)
   - Create an Azure Web App for Containers

2. **Configure secrets in GitHub:**
   - `ACR_LOGIN_SERVER`
   - `ACR_USERNAME`
   - `ACR_PASSWORD`
   - `AZURE_WEBAPP_NAME`
   - `AZURE_WEBAPP_PUBLISH_PROFILE`

3. **Push to main branch** to trigger deployment.

### API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Your text to classify"
}
```

**Response:**
```json
{
  "class_id": 0,
  "class_name": "Company",
  "probabilities": [0.95, 0.02, ...]
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
