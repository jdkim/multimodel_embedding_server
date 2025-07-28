# multimodel_embedding_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Dict, Any
import time
import asyncio
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model Embedding Server",
    description="Local embedding server with BiomedBERT, BlueBERT, and multilingual-e5-large",
    version="1.1.0"
)

# Global model storage
models = {}
tokenizers = {}

# Model configurations
MODEL_CONFIGS = {
    "pubmedbert": {
        "name": "NeuML/pubmedbert-base-embeddings",
        "type": "sentence_transformers",
        "description": "PubMed BERT optimized for embeddings and similarity search"
    },
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large", 
        "type": "sentence_transformers",
        "description": "Multilingual embedding model with strong performance"
    },
    "bluebert": {
        "name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "type": "transformers",
        "description": "BlueBERT pre-trained on PubMed abstracts and MIMIC-III clinical notes"
    }
}

class EmbeddingRequest(BaseModel):
    prompt: str
    model: str = "pubmedbert"

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "pubmedbert"

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model: str = "pubmedbert"
    metric: str = "cosine"  # cosine, euclidean, manhattan, chebyshev

class SimilarityResponse(BaseModel):
    similarity: float
    distance: float
    metric: str
    model: str
    text1: str
    text2: str

class BatchSimilarityRequest(BaseModel):
    texts: List[str]
    model: str = "pubmedbert"
    metric: str = "cosine"

class BatchSimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]]
    distance_matrix: List[List[float]]
    metric: str
    model: str
    texts: List[str]

class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]

async def load_transformers_model(model_key: str, model_name: str):
    """Load a transformers model"""
    logger.info(f"🔄 Loading {model_key} ({model_name})...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    tokenizers[model_key] = tokenizer
    models[model_key] = model

    load_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ {model_key} loaded in {load_time:.2f}s ({param_count:,} parameters)")

async def load_sentence_transformers_model(model_key: str, model_name: str):
    """Load a sentence-transformers model"""
    logger.info(f"🔄 Loading {model_key} ({model_name})...")
    start_time = time.time()

    model = SentenceTransformer(model_name)
    models[model_key] = model

    load_time = time.time() - start_time
    logger.info(f"✅ {model_key} loaded in {load_time:.2f}s")

@app.on_event("startup")
async def load_models():
    """Load all models at startup"""
    logger.info("🚀 Starting Multi-Model Embedding Server...")

    tasks = []
    for model_key, config in MODEL_CONFIGS.items():
        if config["type"] == "transformers":
            task = load_transformers_model(model_key, config["name"])
        elif config["type"] == "sentence_transformers":
            task = load_sentence_transformers_model(model_key, config["name"])
        tasks.append(task)

    # Load models concurrently
    await asyncio.gather(*tasks)
    logger.info("🎉 All models loaded successfully!")

def get_embedding_transformers(text: str, model_key: str) -> List[float]:
    """Get embedding using transformers model"""
    tokenizer = tokenizers[model_key]
    model = models[model_key]

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[0, 0, :].numpy().tolist()

    return embedding

def get_embeddings_batch_transformers(texts: List[str], model_key: str) -> List[List[float]]:
    """Get batch embeddings using transformers model"""
    tokenizer = tokenizers[model_key]
    model = models[model_key]

    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy().tolist()

    return embeddings

def get_embedding_sentence_transformers(text: str, model_key: str) -> List[float]:
    """Get embedding using sentence-transformers model"""
    model = models[model_key]
    embedding = model.encode([text])[0].tolist()
    return embedding

def get_embeddings_batch_sentence_transformers(texts: List[str], model_key: str) -> List[List[float]]:
    """Get batch embeddings using sentence-transformers model"""
    model = models[model_key]
    embeddings = model.encode(texts).tolist()
    return embeddings

def calculate_similarity_and_distance(embedding1: List[float], embedding2: List[float], metric: str = "cosine"):
    """Calculate similarity and distance between two embeddings"""
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)

    if metric == "cosine":
        similarity = cosine_similarity(emb1, emb2)[0][0]
        distance = 1 - similarity
    elif metric == "euclidean":
        distance = euclidean_distances(emb1, emb2)[0][0]
        # Convert to similarity (0-1 range, higher = more similar)
        similarity = 1 / (1 + distance)
    elif metric == "manhattan":
        distance = manhattan(emb1[0], emb2[0])
        similarity = 1 / (1 + distance)
    elif metric == "chebyshev":
        distance = chebyshev(emb1[0], emb2[0])
        similarity = 1 / (1 + distance)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return float(similarity), float(distance)

def calculate_similarity_matrix(embeddings: List[List[float]], metric: str = "cosine"):
    """Calculate similarity and distance matrices for multiple embeddings"""
    embeddings_array = np.array(embeddings)

    if metric == "cosine":
        similarity_matrix = cosine_similarity(embeddings_array)
        distance_matrix = 1 - similarity_matrix
    elif metric == "euclidean":
        distance_matrix = euclidean_distances(embeddings_array)
        # Convert to similarity
        similarity_matrix = 1 / (1 + distance_matrix)
    elif metric == "manhattan":
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = manhattan(embeddings_array[i], embeddings_array[j])
        similarity_matrix = 1 / (1 + distance_matrix)
    elif metric == "chebyshev":
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = chebyshev(embeddings_array[i], embeddings_array[j])
        similarity_matrix = 1 / (1 + distance_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return similarity_matrix.tolist(), distance_matrix.tolist()

def validate_model(model_key: str):
    """Validate that model exists and is loaded"""
    if model_key not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_key}' not available. Available models: {available_models}"
        )

    if model_key not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_key}' is not loaded yet. Please wait for startup to complete."
        )

@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    """Get single embedding"""
    validate_model(request.model)

    start_time = time.time()
    config = MODEL_CONFIGS[request.model]

    if config["type"] == "transformers":
        embedding = get_embedding_transformers(request.prompt, request.model)
    else:  # sentence_transformers
        embedding = get_embedding_sentence_transformers(request.prompt, request.model)

    processing_time = time.time() - start_time
    logger.info(f"📝 [{request.model}] Embedded text (len={len(request.prompt)}) in {processing_time:.3f}s")

    return EmbeddingResponse(
        embedding=embedding,
        model=request.model
    )

@app.post("/api/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Get batch embeddings"""
    validate_model(request.model)

    start_time = time.time()
    texts = request.input if isinstance(request.input, list) else [request.input]
    config = MODEL_CONFIGS[request.model]
    
    if config["type"] == "transformers":
        embeddings = get_embeddings_batch_transformers(texts, request.model)
    else:  # sentence_transformers
        embeddings = get_embeddings_batch_sentence_transformers(texts, request.model)

    processing_time = time.time() - start_time
    logger.info(f"📊 [{request.model}] Embedded {len(texts)} texts in {processing_time:.3f}s")

    return EmbedResponse(embeddings=embeddings)

@app.post("/api/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity between two texts"""
    validate_model(request.model)

    # Validate metric
    valid_metrics = ["cosine", "euclidean", "manhattan", "chebyshev"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{request.metric}'. Valid options: {valid_metrics}"
        )

    start_time = time.time()
    config = MODEL_CONFIGS[request.model]

    # Get embeddings for both texts
    if config["type"] == "transformers":
        embedding1 = get_embedding_transformers(request.text1, request.model)
        embedding2 = get_embedding_transformers(request.text2, request.model)
    else:  # sentence_transformers
        embedding1 = get_embedding_sentence_transformers(request.text1, request.model)
        embedding2 = get_embedding_sentence_transformers(request.text2, request.model)

    # Calculate similarity and distance
    similarity, distance = calculate_similarity_and_distance(embedding1, embedding2, request.metric)

    processing_time = time.time() - start_time
    logger.info(f"🔍 [{request.model}] Calculated {request.metric} similarity in {processing_time:.3f}s: {similarity:.4f}")

    return SimilarityResponse(
        similarity=similarity,
        distance=distance,
        metric=request.metric,
        model=request.model,
        text1=request.text1,
        text2=request.text2
    )

@app.post("/api/similarity/batch", response_model=BatchSimilarityResponse)
async def calculate_batch_similarity(request: BatchSimilarityRequest):
    """Calculate similarity matrix for multiple texts"""
    validate_model(request.model)

    # Validate metric
    valid_metrics = ["cosine", "euclidean", "manhattan", "chebyshev"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{request.metric}'. Valid options: {valid_metrics}"
        )

    if len(request.texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 texts are required for batch similarity calculation"
        )

    start_time = time.time()
    config = MODEL_CONFIGS[request.model]

    # Get embeddings for all texts
    if config["type"] == "transformers":
        embeddings = get_embeddings_batch_transformers(request.texts, request.model)
    else:  # sentence_transformers
        embeddings = get_embeddings_batch_sentence_transformers(request.texts, request.model)

    # Calculate similarity and distance matrices
    similarity_matrix, distance_matrix = calculate_similarity_matrix(embeddings, request.metric)

    processing_time = time.time() - start_time
    logger.info(f"🔍 [{request.model}] Calculated {request.metric} similarity matrix for {len(request.texts)} texts in {processing_time:.3f}s")

    return BatchSimilarityResponse(
        similarity_matrix=similarity_matrix,
        distance_matrix=distance_matrix,
        metric=request.metric,
        model=request.model,
        texts=request.texts
    )

@app.get("/api/models", response_model=ModelListResponse)
async def list_models():
    """List available models"""
    model_list = []
    for model_key, config in MODEL_CONFIGS.items():
        model_info = {
            "name": model_key,
            "full_name": config["name"],
            "type": config["type"],
            "description": config["description"],
            "loaded": model_key in models,
            "embedding_dim": 768 if config["type"] == "transformers" else (1024 if "e5-large" in config["name"] else 768)
        }
        model_list.append(model_info)

    return ModelListResponse(models=model_list)

@app.get("/health")
async def health():
    """Health check"""
    loaded_models = {k: k in models for k in MODEL_CONFIGS.keys()}
    return {
        "status": "healthy",
        "models": loaded_models,
        "total_models": len(MODEL_CONFIGS),
        "loaded_count": sum(loaded_models.values())
    }

@app.get("/")
async def root():
    """Root endpoint with info"""
    return {
        "message": "Multi-Model Embedding Server",
        "available_models": list(MODEL_CONFIGS.keys()),
        "endpoints": {
            "embeddings": "/api/embeddings",
            "batch_embed": "/api/embed",
            "similarity": "/api/similarity",
            "batch_similarity": "/api/similarity/batch",
            "list_models": "/api/models",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_metrics": ["cosine", "euclidean", "manhattan", "chebyshev"]
    }
