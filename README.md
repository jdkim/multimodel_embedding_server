# Multi-Model Embedding Server

A web embedding server that provides text embeddings with similarity calculation capabilities through a FastAPI-based REST API. Compatible with Ollama's embedding API format.

## 🚀 Features

- **Multiple Models**: PubMedBERT and BiomedBERT for biomedical text analysis
- **Ollama-Compatible API**: Drop-in replacement for Ollama embedding endpoints
- **Document Similarity**: Advanced document chunking and similarity search capabilities
- **Similarity Calculations**: Built-in cosine, euclidean, manhattan, and chebyshev distance metrics
- **High Performance**: FastAPI with async support and concurrent request handling
- **Batch Processing**: Efficient batch embedding generation and similarity matrices
- **Model Selection**: Choose the best model for your specific use case

## 📋 Currently Supported Models

| Model | Key | Use Case | Embedding Dimension |
|-------|-----|----------|-------------------|
| PubMedBERT | `pubmedbert` | Biomedical text, scientific papers (optimized for embeddings) | 768 |
| BiomedBERT | `biomedbert` | Biomedical abstracts and full-text articles | 768 |

## 🛠️ Installation

This project uses [pip-tools](https://pip-tools.readthedocs.io/) to manage dependencies. Follow these steps to set up your development environment:

### Prerequisites

- Python 3.7 or higher
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone multimodel_embedding_server
   cd multimodel_embedding_server
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install pip-tools:**
   ```bash
   pip install pip-tools
   ```

4. **Install dependencies:**
   ```bash
   pip-sync requirements.txt
   ```

   If `requirements.txt` doesn't exist yet, compile it from `requirements.in`:
   ```bash
   pip-compile requirements.in
   pip-sync requirements.txt
   ```

### Development Dependencies (Optional)

If you have separate development dependencies in `requirements-dev.in`:

```bash
pip-compile requirements-dev.in
pip-sync requirements.txt requirements-dev.txt
```

### Adding New Dependencies

To add a new dependency:

1. Add it to `requirements.in` (or `requirements-dev.in` for dev dependencies)
2. Compile the requirements:
   ```bash
   pip-compile requirements.in
   ```
3. Sync your environment:
   ```bash
   pip-sync requirements.txt
   ```

### Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
pip-compile --upgrade requirements.in
pip-sync requirements.txt
```

## 🚀 Quick Start

### 1. Start the Server

**Development:**
```bash
uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435
```

**Production (gunicorn):**
```bash
gunicorn multimodel_embedding_server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:11435
```

The server will automatically download and load the models on the first startup (this may take a few minutes).

### 2. Test with curl

```bash
# Get embedding for biomedical text
curl -X POST "http://localhost:11435/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "myocardial infarction", "model": "pubmedbert"}'

# Get embedding with BiomedBERT for biomedical text
curl -X POST "http://localhost:11435/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "patient presents with chest pain", "model": "biomedbert"}'

# Calculate similarity between terms
curl -X POST "http://localhost:11435/api/similarity" \
  -H "Content-Type: application/json" \
  -d '{"text1": "myocardial infarction", "text2": "heart attack", "model": "pubmedbert", "metric": "cosine"}'
```

## 🔌 API Endpoints

### Core Endpoints

#### `POST /api/embeddings`
Get a single embedding (Ollama-compatible)

**Request:**
```json
{
  "prompt": "your text here",
  "model": "pubmedbert"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "model": "pubmedbert"
}
```

#### `POST /api/embed`
Get embeddings for multiple texts

**Request:**
```json
{
  "input": ["text 1", "text 2", "text 3"],
  "model": "pubmedbert"
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, 0.2, ...],
    [0.3, 0.4, ...],
    [0.5, 0.6, ...]
  ]
}
```

#### `POST /api/similarity`
Calculate similarity between two texts

**Request:**
```json
{
  "text1": "myocardial infarction",
  "text2": "heart attack",
  "model": "pubmedbert",
  "metric": "cosine"
}
```

**Response:**
```json
{
  "similarity": 0.8234,
  "distance": 0.1766,
  "metric": "cosine",
  "model": "pubmedbert",
  "text1": "myocardial infarction",
  "text2": "heart attack"
}
```

#### `POST /api/similarity/batch`
Calculate similarity matrix for multiple texts

**Request:**
```json
{
  "texts": ["diabetes", "hypertension", "heart attack", "stroke"],
  "model": "biomedbert",
  "metric": "cosine"
}
```

**Response:**
```json
{
  "similarity_matrix": [
    [1.0, 0.65, 0.42, 0.38],
    [0.65, 1.0, 0.48, 0.55],
    [0.42, 0.48, 1.0, 0.62],
    [0.38, 0.55, 0.62, 1.0]
  ],
  "distance_matrix": [
    [0.0, 0.35, 0.58, 0.62],
    [0.35, 0.0, 0.52, 0.45],
    [0.58, 0.52, 0.0, 0.38],
    [0.62, 0.45, 0.38, 0.0]
  ],
  "metric": "cosine",
  "model": "biomedbert",
  "texts": ["diabetes", "hypertension", "heart attack", "stroke"]
}
```

### Document Similarity Endpoints

#### `POST /api/document-similarity`
Find the most similar chunk in a document to a keyword using sliding window chunking

**Request:**
```json
{
  "keyword": "myocardial infarction",
  "document": "Long document text here...",
  "model": "pubmedbert",
  "chunk_size_tokens": 512,
  "overlap_percent": 50,
  "metric": "cosine"
}
```

**Response:**
```json
{
  "keyword": "myocardial infarction",
  "best_chunk": "The chunk of text most similar to the keyword...",
  "similarity_score": 0.8765,
  "chunk_index": 3,
  "total_chunks": 12,
  "chunk_start_offset": 1024,
  "chunk_end_offset": 1536,
  "metric": "cosine",
  "model": "pubmedbert"
}
```

#### `POST /api/document-similarity-recursive`
Find the most similar chunk using recursive hierarchical text splitting

**Request:**
```json
{
  "keyword": "diabetes",
  "document": "Long document text here...",
  "model": "pubmedbert",
  "max_tokens": 100,
  "separators": ["\n\n", "\n", ". ", " "],
  "metric": "cosine"
}
```

**Response:**
```json
{
  "keyword": "diabetes",
  "best_chunk": "The chunk of text most similar to the keyword...",
  "similarity_score": 0.9123,
  "chunk_index": 2,
  "total_chunks": 8,
  "chunk_start_offset": 512,
  "chunk_end_offset": 768,
  "metric": "cosine",
  "model": "pubmedbert"
}
```

### Management Endpoints

#### `GET /api/models`
List available models and their status

#### `GET /health`
Health check and system status

#### `GET /docs`
Interactive API documentation (Swagger UI)

## 📊 Supported Similarity Metrics

| Metric | Description | Best For | Range |
|--------|-------------|----------|--------|
| **cosine** | Cosine similarity | General text similarity | 0-1 (1=identical) |
| **euclidean** | Euclidean distance | Geometric similarity | 0-∞ (0=identical) |
| **manhattan** | Manhattan/L1 distance | Robust to outliers | 0-∞ (0=identical) |
| **chebyshev** | Chebyshev/L∞ distance | Maximum difference | 0-∞ (0=identical) |

## 📄 Document Chunking Strategies

The server supports two intelligent text chunking strategies for document similarity:

### Sliding Window Chunking
- **Token-based**: Chunks based on actual model tokens
- **Overlap**: Configurable percentage overlap between chunks
- **Best for**: Consistent chunk sizes, dense document analysis
- **Use case**: Research papers, continuous text analysis

### Recursive Hierarchical Chunking
- **Structure-aware**: Respects document structure (paragraphs, sentences, etc.)
- **Adaptive**: Only splits when necessary based on token limits
- **Hierarchical separators**: `["\n\n", "\n", ". ", " "]` (customizable)
- **Best for**: Maintaining semantic boundaries, structured documents
- **Use case**: Medical reports, structured documents with clear sections

## 💡 Usage Examples

### Biomedical Text Processing

```python
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_biomedical_terms(term1, term2, model="pubmedbert"):
    """Compare similarity between biomedical terms"""
    response = requests.post(
        "http://localhost:11435/api/similarity",
        json={
            "text1": term1,
            "text2": term2,
            "model": model,
            "metric": "cosine"
        }
    )

    result = response.json()
    return result["similarity"]

# Example
similarity = compare_biomedical_terms("myocardial infarction", "heart attack")
print(f"Similarity: {similarity:.3f}")

# Compare across models
pubmedbert_sim = compare_biomedical_terms("diabetes", "hyperglycemia", "pubmedbert")
biomedbert_sim = compare_biomedical_terms("diabetes", "hyperglycemia", "biomedbert")
print(f"PubMedBERT: {pubmedbert_sim:.3f}, BiomedBERT: {biomedbert_sim:.3f}")
```

### Clinical Text Analysis

```python
def analyze_clinical_similarity():
    """Analyze similarity between clinical terms using BiomedBERT"""

    clinical_terms = [
        "patient presents with chest pain",
        "acute myocardial infarction",
        "shortness of breath",
        "dyspnea on exertion",
        "elevated troponin levels"
    ]

    response = requests.post(
        "http://localhost:11435/api/similarity/batch",
        json={
            "texts": clinical_terms,
            "model": "biomedbert",
            "metric": "cosine"
        }
    )

    result = response.json()
    similarity_matrix = result['similarity_matrix']

    # Find most similar pairs
    for i in range(len(clinical_terms)):
        for j in range(i+1, len(clinical_terms)):
            similarity = similarity_matrix[i][j]
            if similarity > 0.7:  # High similarity threshold
                print(f"High similarity ({similarity:.3f}): {clinical_terms[i]} ↔ {clinical_terms[j]}")

analyze_clinical_similarity()
```

### Document Similarity Analysis

```python
def analyze_document_similarity():
    """Find most relevant sections in biomedical documents"""
    
    # Sample biomedical document
    document = """
    Diabetes mellitus is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time.
    Symptoms often include frequent urination, increased thirst and increased appetite. If left untreated, diabetes can cause many health complications.
    
    Type 1 diabetes results from failure of the pancreas to produce enough insulin due to loss of beta cells.
    This form was previously referred to as "insulin-dependent diabetes mellitus" or "juvenile diabetes".
    The loss of beta cells is caused by an autoimmune response.
    
    Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly.
    As the disease progresses, a lack of insulin may also develop. This form was previously referred to as "non-insulin-dependent diabetes mellitus".
    The most common cause is a combination of excessive body weight and insufficient exercise.
    """
    
    # Find most similar chunk to a keyword using sliding window
    response = requests.post(
        "http://localhost:11435/api/document-similarity",
        json={
            "keyword": "insulin resistance",
            "document": document,
            "model": "biomedbert",
            "chunk_size_tokens": 50,
            "overlap_percent": 25,
            "metric": "cosine"
        }
    )
    
    result = response.json()
    print(f"Best matching chunk (score: {result['similarity_score']:.3f}):")
    print(f"'{result['best_chunk']}'")
    print(f"Location: characters {result['chunk_start_offset']}-{result['chunk_end_offset']}")
    
    # Compare with recursive chunking strategy
    response = requests.post(
        "http://localhost:11435/api/document-similarity-recursive",
        json={
            "keyword": "autoimmune response",
            "document": document,
            "model": "pubmedbert",
            "max_tokens": 30,
            "separators": ["\n\n", "\n", ". "],
            "metric": "cosine"
        }
    )
    
    result = response.json()
    print(f"\nRecursive chunking result (score: {result['similarity_score']:.3f}):")
    print(f"'{result['best_chunk']}'")

analyze_document_similarity()
```

### Medical Literature Search

```python
class MedicalSearch:
    def __init__(self, model="pubmedbert"):
        self.model = model
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        """Add medical documents to the search index"""
        response = requests.post(
            "http://localhost:11435/api/embed",
            json={"input": docs, "model": self.model}
        )

        new_embeddings = response.json()["embeddings"]

        self.documents.extend(docs)
        self.embeddings.extend(new_embeddings)

    def search(self, query, top_k=5):
        """Search for similar medical documents"""

        # Calculate similarity with all documents
        similarities = []
        for doc_embedding in self.embeddings:
            response = requests.post(
                "http://localhost:11435/api/similarity",
                json={
                    "text1": query,
                    "text2": "",  # We'll use embeddings directly
                    "model": self.model,
                    "metric": "cosine"
                }
            )
            # In practice, you'd want to implement batch similarity
            # This is simplified for the example

        # Alternative: Use batch similarity
        docs_plus_query = self.documents + [query]
        response = requests.post(
            "http://localhost:11435/api/similarity/batch",
            json={
                "texts": docs_plus_query,
                "model": self.model,
                "metric": "cosine"
            }
        )

        similarity_matrix = response.json()["similarity_matrix"]
        query_similarities = similarity_matrix[-1][:-1]  # Last row, excluding self-similarity

        # Get top results
        top_indices = sorted(range(len(query_similarities)), 
                           key=lambda i: query_similarities[i], reverse=True)[:top_k]
        
        results = [
            {"document": self.documents[i], "score": query_similarities[i]}
            for i in top_indices
        ]

        return results

# Usage
search_engine = MedicalSearch(model="pubmedbert")
search_engine.add_documents([
    "Diabetes mellitus is a metabolic disorder characterized by high blood sugar",
    "Hypertension is persistently high blood pressure in the arteries",
    "Myocardial infarction occurs when blood flow to the heart muscle is blocked",
    "Stroke happens when blood supply to part of the brain is interrupted",
    "Pneumonia is an infection that inflames air sacs in the lungs"
])

results = search_engine.search("What causes heart attacks?")
for result in results:
    print(f"Score: {result['score']:.3f} - {result['document']}")
```

## ⚙️ Configuration

### Production Deployment

```bash
# Single worker (for GPU usage)
uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435 --workers 1

# Multiple workers (CPU only)
uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435 --workers 4

# With Gunicorn (production)
gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:11435 multimodel_embedding_server:app
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export TORCH_HOME=/path/to/models  # Cache directory for models
```

### Adding New Models

To add a new model, edit the `MODEL_CONFIGS` dictionary in `multimodel_embedding_server.py`:

```python
MODEL_CONFIGS = {
    "your-model": {
        "name": "huggingface/model-name",
        "type": "sentence_transformers",  # or "transformers"
        "description": "Your model description"
    }
}
```

## 📊 Performance

### Performance

| Model | Single Embedding | Batch (32 texts) | Memory Usage |
|-------|------------------|-------------------|---------------|
| PubMedBERT | ~50ms | ~1.5s | ~1.2GB |
| BiomedBERT | ~50ms | ~1.5s | ~1.2GB |

### Batch Processing Guidelines

| Hardware | Recommended Batch Size | Max Batch Size |
|----------|------------------------|----------------|
| CPU Only | 16 texts | 32 texts |
| GPU (4-8GB) | 32 texts | 64 texts |
| GPU (12GB+) | 64 texts | 128 texts |

### Optimization Tips

- Use batch processing (`/api/embed`) for multiple texts - much more efficient than individual calls
- Use `/api/similarity/batch` for similarity matrices instead of individual comparisons
- Choose the right model: PubMedBERT for optimized embeddings, BiomedBERT for biomedical abstracts and full-text
- Use document similarity endpoints for finding relevant sections in long texts
- Run on GPU for 3-5x better performance
- Use single worker mode when using GPU to avoid memory conflicts
- Consider model quantization for production deployments with memory constraints

### Recommended Batch Sizes

```python
# Good practice: chunk large lists
def process_large_list(texts, model="pubmedbert", chunk_size=32):
    all_embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        response = requests.post(
            "http://localhost:11435/api/embed",
            json={"input": chunk, "model": model}
        )
        all_embeddings.extend(response.json()["embeddings"])
    return all_embeddings
```

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Check if models are accessible
curl http://localhost:11435/api/models

# Check server logs for download progress

# Test document similarity
curl -X POST "http://localhost:11435/api/document-similarity" \
  -H "Content-Type: application/json" \
  -d '{"keyword": "diabetes", "document": "Long medical text...", "model": "pubmedbert"}'
```

**CUDA out of memory:**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in requests
curl -X POST "http://localhost:11435/api/embed" \
  -d '{"input": ["term1", "term2"], "model": "pubmedbert"}'  # Smaller batch
```

**Slow similarity calculations:**
```bash
# Use batch endpoint for multiple comparisons
curl -X POST "http://localhost:11435/api/similarity/batch" \
  -d '{"texts": ["term1", "term2", "term3"], "model": "pubmedbert"}'
```

**Model loading timeout:**
- Models download on first run (3-8 minutes total for both models)
- Increase `TimeoutStartSec=600` in systemd service for slower connections
- Check `/health` endpoint to monitor loading progress

### Logs

The server provides detailed logging:
```
🚀 Starting Multi-Model Embedding Server...
🔄 Loading pubmedbert (NeuML/pubmedbert-base-embeddings)...
✅ pubmedbert loaded in 45.32s
🔄 Loading biomedbert (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)...
✅ biomedbert loaded in 38.45s (109,482,240 parameters)
🎉 All models loaded successfully!
📝 [pubmedbert] Embedded text (len=19) in 0.043s
🔍 [biomedbert] Calculated cosine similarity in 0.089s: 0.8234
🔍 [pubmedbert] Found best chunk (3/12) with cosine similarity 0.8765 in 0.156s
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test with multiple models
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings) by NeuML for biomedical embeddings
- [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) by Microsoft for biomedical text analysis
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Sentence Transformers](https://www.sbert.net/) for embedding utilities
- [scikit-learn](https://scikit-learn.org/) for similarity metrics

## 📞 Support

- Create an issue for bugs or feature requests
- Check the `/docs` endpoint for interactive API documentation
- Monitor the `/health` endpoint for system status