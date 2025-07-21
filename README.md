# Multi-Model Embedding Server

A high-performance local embedding server that provides biomedical and multilingual text embeddings through a FastAPI-based REST API. Compatible with Ollama's embedding API format.

## 🚀 Features

- **Multiple Models**: BiomedBERT for biomedical text and Multilingual-E5-Large for general/multilingual text
- **Ollama-Compatible API**: Drop-in replacement for Ollama embedding endpoints
- **High Performance**: FastAPI with async support and concurrent request handling
- **Batch Processing**: Efficient batch embedding generation
- **Model Selection**: Choose the best model for your specific use case
- **Health Monitoring**: Built-in health checks and model status endpoints
- **Interactive Docs**: Auto-generated API documentation

## 📋 Supported Models

| Model | Key | Use Case | Embedding Dimension |
|-------|-----|----------|-------------------|
| BiomedBERT | `biomedbert` | Biomedical text, clinical notes, scientific papers | 768 |
| Multilingual-E5-Large | `multilingual-e5-large` | General text, multilingual content | 1024 |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, optional)

### Option 1: Using requirements.txt (Recommended)

```bash
# Clone or download the server files
git clone <repository-url>  # or download multimodel_embedding_server.py
cd multi-model-embedding-server

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435
```

### Option 2: Using Poetry

```bash
# Install Poetry first
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the server
poetry run uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435
```

### Option 3: Manual Installation

```bash
pip install fastapi uvicorn[standard] transformers torch sentence-transformers scikit-learn

uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435
```

## 📦 Dependencies

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.36.2
torch>=2.0.0
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy>=1.24.0
requests>=2.31.0
```

## 🚀 Quick Start

### 1. Start the Server

```bash
uvicorn multimodel_embedding_server:app --host 0.0.0.0 --port 11435
```

The server will automatically download and load both models on first startup (this may take a few minutes).

### 2. Test with curl

```bash
# Get embedding for biomedical text
curl -X POST "http://localhost:11435/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "myocardial infarction", "model": "biomedbert"}'

# Get embedding for multilingual text
curl -X POST "http://localhost:11435/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "model": "multilingual-e5-large"}'
```

### 3. Python Usage

```python
import requests

def get_embedding(text, model="biomedbert"):
    response = requests.post(
        "http://localhost:11435/api/embeddings",
        json={"prompt": text, "model": model}
    )
    return response.json()["embedding"]

# Example usage
embedding = get_embedding("diabetes mellitus type 2", "biomedbert")
print(f"Embedding dimension: {len(embedding)}")
```

## 🔌 API Endpoints

### Core Endpoints

#### `POST /api/embeddings`
Get a single embedding (Ollama-compatible)

**Request:**
```json
{
  "prompt": "your text here",
  "model": "biomedbert"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "model": "biomedbert"
}
```

#### `POST /api/embed`
Get embeddings for multiple texts

**Request:**
```json
{
  "input": ["text 1", "text 2", "text 3"],
  "model": "multilingual-e5-large"
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

### Management Endpoints

#### `GET /api/models`
List available models and their status

#### `GET /health`
Health check and system status

#### `GET /docs`
Interactive API documentation (Swagger UI)

## 💡 Usage Examples

### Biomedical Text Processing

```python
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_biomedical_terms(term1, term2):
    """Compare similarity between biomedical terms"""
    embeddings_response = requests.post(
        "http://localhost:11435/api/embed",
        json={
            "input": [term1, term2],
            "model": "biomedbert"
        }
    )
    
    embeddings = embeddings_response.json()["embeddings"]
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity

# Example
similarity = compare_biomedical_terms("myocardial infarction", "heart attack")
print(f"Similarity: {similarity:.3f}")
```

### Multilingual Text Processing

```python
def process_multilingual_text():
    texts = [
        "Hello, how are you?",          # English
        "Hola, ¿cómo estás?",          # Spanish  
        "Bonjour, comment ça va?",      # French
        "你好，你好吗？"                # Chinese
    ]
    
    response = requests.post(
        "http://localhost:11435/api/embed",
        json={
            "input": texts,
            "model": "multilingual-e5-large"
        }
    )
    
    return response.json()["embeddings"]

embeddings = process_multilingual_text()
print(f"Generated {len(embeddings)} multilingual embeddings")
```

### RAG (Retrieval-Augmented Generation) Setup

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingSearch:
    def __init__(self, model="biomedbert"):
        self.model = model
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs):
        """Add documents to the search index"""
        response = requests.post(
            "http://localhost:11435/api/embed",
            json={"input": docs, "model": self.model}
        )
        
        new_embeddings = response.json()["embeddings"]
        
        self.documents.extend(docs)
        self.embeddings.extend(new_embeddings)
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        query_response = requests.post(
            "http://localhost:11435/api/embeddings",
            json={"prompt": query, "model": self.model}
        )
        
        query_embedding = query_response.json()["embedding"]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {"document": self.documents[i], "score": similarities[i]}
            for i in top_indices
        ]
        
        return results

# Usage
search_engine = EmbeddingSearch(model="biomedbert")
search_engine.add_documents([
    "Diabetes is a metabolic disorder characterized by high blood sugar",
    "Hypertension is high blood pressure",
    "Myocardial infarction is commonly known as a heart attack"
])

results = search_engine.search("What is a heart attack?")
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

### Benchmarks (approximate, varies by hardware)

| Model | Single Embedding | Batch (10 texts) | Memory Usage |
|-------|------------------|-------------------|---------------|
| BiomedBERT | ~50ms | ~200ms | ~1.2GB |
| Multilingual-E5-Large | ~80ms | ~300ms | ~2.1GB |

### Optimization Tips

- Use batch processing (`/api/embed`) for multiple texts
- Run on GPU for better performance
- Use single worker mode when using GPU to avoid memory conflicts
- Consider model quantization for production deployments

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Check if models are accessible
curl http://localhost:11435/api/models

# Check server logs for download progress
```

**CUDA out of memory:**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**Slow startup:**
- Models are downloaded on first run (can take 5-10 minutes)
- Subsequent starts are much faster
- Check `/health` endpoint to monitor loading progress

### Logs

The server provides detailed logging:
```
🔄 Loading biomedbert (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)...
✅ biomedbert loaded in 45.32s (109,482,240 parameters)
📝 [biomedbert] Embedded text (len=19) in 0.043s
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

- [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) by Microsoft Research
- [Multilingual-E5-Large](https://huggingface.co/intfloat/multilingual-e5-large) by Beijing Academy of Artificial Intelligence
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Sentence Transformers](https://www.sbert.net/) for embedding utilities

## 📞 Support

- Create an issue for bugs or feature requests
- Check the `/docs` endpoint for interactive API documentation
- Monitor the `/health` endpoint for system status