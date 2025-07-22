from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Dict, Any
import time
import asyncio

app = FastAPI(
    title="Multi-Model Embedding Server",
    description="Local embedding server with BiomedBERT and multilingual-e5-large",
    version="1.0.0"
)

# Global model storage
models = {}
tokenizers = {}

# Model configurations
MODEL_CONFIGS = {
    "biomedbert": {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "type": "transformers",
        "description": "Biomedical domain-specific BERT"
    },
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large", 
        "type": "sentence_transformers",
        "description": "Multilingual embedding model with strong performance"
    },
    "bluebert": {
        "name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "type": "transformers", 
        "description": "BERT pre-trained on PubMed abstracts and MIMIC-III clinical notes"
    }
}

class EmbeddingRequest(BaseModel):
    prompt: str
    model: str = "biomedbert"

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "biomedbert"

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]

async def load_transformers_model(model_key: str, model_name: str):
    """Load a transformers model"""
    print(f"🔄 Loading {model_key} ({model_name})...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    tokenizers[model_key] = tokenizer
    models[model_key] = model
    
    load_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ {model_key} loaded in {load_time:.2f}s ({param_count:,} parameters)")

async def load_sentence_transformers_model(model_key: str, model_name: str):
    """Load a sentence-transformers model"""
    print(f"🔄 Loading {model_key} ({model_name})...")
    start_time = time.time()
    
    model = SentenceTransformer(model_name)
    models[model_key] = model
    
    load_time = time.time() - start_time
    print(f"✅ {model_key} loaded in {load_time:.2f}s")

@app.on_event("startup")
async def load_models():
    """Load all models at startup"""
    print("🚀 Starting Multi-Model Embedding Server...")
    
    tasks = []
    for model_key, config in MODEL_CONFIGS.items():
        if config["type"] == "transformers":
            task = load_transformers_model(model_key, config["name"])
        elif config["type"] == "sentence_transformers":
            task = load_sentence_transformers_model(model_key, config["name"])
        tasks.append(task)
    
    # Load models concurrently
    await asyncio.gather(*tasks)
    print("🎉 All models loaded successfully!")

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
    print(f"📝 [{request.model}] Embedded text (len={len(request.prompt)}) in {processing_time:.3f}s")
    
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
    print(f"📊 [{request.model}] Embedded {len(texts)} texts in {processing_time:.3f}s")
    
    return EmbedResponse(embeddings=embeddings)

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
            "list_models": "/api/models",
            "health": "/health",
            "docs": "/docs"
        }
    }