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
from scipy.spatial.distance import cityblock as manhattan, chebyshev
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device detection for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🖥️ Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

app = FastAPI(
    title="Multi-Model Embedding Server",
    description="Local embedding server with BiomedBERT and PubMedBERT",
    version="1.3.0"
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
    "sapbert": {
        "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "type": "sentence_transformers",
        "description": "Self-alignment pretrained BERT for biomedical entity linking"
    },
    "biolord": {
        "name": "FremyCompany/BioLORD-2023",
        "type": "sentence_transformers",
        "description": "BioLORD-2023 for biomedical semantic similarity and entity linking"
    },
    "biomedbert": {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "type": "transformers",
        "description": "Microsoft BiomedBERT trained on biomedical abstracts and full-text"
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

class DocumentSimilarityRequest(BaseModel):
    keyword: str
    document: str
    model: str = "pubmedbert"
    chunk_size_tokens: int = 512
    overlap_percent: int = 50
    metric: str = "cosine"

class DocumentSimilarityResponse(BaseModel):
    keyword: str
    best_chunk: str
    similarity_score: float
    chunk_index: int
    total_chunks: int
    chunk_start_offset: int
    chunk_end_offset: int
    metric: str
    model: str

class DocumentSimilarityRecursiveRequest(BaseModel):
    keyword: str
    document: str
    model: str = "pubmedbert"
    max_tokens: int = 100
    separators: List[str] = ["\n\n", "\n", ". ", " "]
    metric: str = "cosine"

async def load_transformers_model(model_key: str, model_name: str):
    """Load a transformers model"""
    logger.info(f"🔄 Loading {model_key} ({model_name})...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    tokenizers[model_key] = tokenizer
    models[model_key] = model

    load_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ {model_key} loaded in {load_time:.2f}s ({param_count:,} parameters) on {device}")

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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()

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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

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

def chunk_document_with_sliding_window_tokens(document: str, model_key: str, chunk_size_tokens: int = 512, overlap_percent: int = 0) -> List[dict]:
    """
    Chunk document using sliding window with specified overlap percentage based on tokens.
    
    Args:
        document: The document to chunk
        model_key: The model key to get the appropriate tokenizer
        chunk_size_tokens: Size of each chunk in tokens
        overlap_percent: Percentage of overlap between chunks (0-100)
    
    Returns:
        List of dictionaries with 'text', 'start_offset', 'end_offset' keys
    """
    if overlap_percent < 0 or overlap_percent > 100:
        raise ValueError("overlap_percent must be between 0 and 100")
    
    config = MODEL_CONFIGS[model_key]
    
    # Get tokenizer based on model type
    if config["type"] == "sentence_transformers":
        # For sentence transformers, we'll use a basic tokenizer approach
        # Split by whitespace as approximation since we don't have direct access to model tokenizer
        tokens = document.split()
        
        if len(tokens) <= chunk_size_tokens:
            return [{"text": document, "start_offset": 0, "end_offset": len(document)}]
        
        chunks = []
        overlap_size = int(chunk_size_tokens * overlap_percent / 100)
        step_size = chunk_size_tokens - overlap_size
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens)
            
            # Find character offsets in original document
            if start == 0:
                start_offset = 0
            else:
                # Find the start of the first token in this chunk
                prefix = ' '.join(tokens[:start])
                start_offset = len(prefix) + (1 if prefix else 0)  # +1 for space
            
            # Find end offset
            if end >= len(tokens):
                end_offset = len(document)
            else:
                prefix_with_chunk = ' '.join(tokens[:end])
                end_offset = len(prefix_with_chunk)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "start_offset": start_offset,
                    "end_offset": end_offset
                })
            
            # If we've reached the end, break
            if end >= len(tokens):
                break
                
            start += step_size
        
        return chunks
    
    elif config["type"] == "transformers":
        # For transformers models, use the actual tokenizer
        tokenizer = tokenizers[model_key]
        
        # Get token-to-character mapping
        encoding = tokenizer(document, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.input_ids
        offsets = encoding.offset_mapping
        
        if len(tokens) <= chunk_size_tokens:
            return [{"text": document, "start_offset": 0, "end_offset": len(document)}]
        
        chunks = []
        overlap_size = int(chunk_size_tokens * overlap_percent / 100)
        step_size = chunk_size_tokens - overlap_size
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Get character offsets from token offsets
            start_offset = offsets[start][0]
            end_offset = offsets[end - 1][1]
            
            chunk_text = document[start_offset:end_offset]
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "start_offset": start_offset,
                    "end_offset": end_offset
                })
            
            # If we've reached the end, break
            if end >= len(tokens):
                break
                
            start += step_size
        
        return chunks
    
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")

def count_tokens_approximation(text: str, model_key: str) -> int:
    """
    Approximate token count for text using model-appropriate method.
    
    Args:
        text: The text to count tokens for
        model_key: The model key to determine tokenization method
    
    Returns:
        Approximate token count
    """
    config = MODEL_CONFIGS[model_key]
    
    if config["type"] == "sentence_transformers":
        # For sentence transformers, approximate using whitespace splitting
        return len(text.split())
    elif config["type"] == "transformers":
        # For transformers models, use actual tokenizer
        tokenizer = tokenizers[model_key]
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")

def chunk_document_recursive(document: str, model_key: str, max_tokens: int = 100, 
                           separators: List[str] = None) -> List[dict]:
    """
    Recursively chunk document using hierarchical separators that respect document structure.
    Only splits when necessary and tries to split around the middle.
    
    Args:
        document: The document to chunk
        model_key: The model key for tokenization
        max_tokens: Maximum tokens per chunk
        separators: List of separators in order of preference (largest to smallest)
    
    Returns:
        List of dictionaries with 'text', 'start_offset', 'end_offset' keys
    """
    if separators is None:
        # Default hierarchy: paragraphs, newlines, sentences, words
        separators = ["\n\n", "\n", ". ", " "]
    
    def _split_at_middle(text: str, separator: str, start_offset: int) -> List[dict]:
        """
        Split text around the middle using the given separator and track offsets.
        """
        parts = text.split(separator)
        if len(parts) <= 1:
            return [{"text": text, "start_offset": start_offset, "end_offset": start_offset + len(text)}]
        
        # Find the middle point
        middle = len(parts) // 2
        
        # Split into two parts around the middle
        left_parts = parts[:middle]
        right_parts = parts[middle:]
        
        # Reconstruct the text with separators and calculate offsets
        left_text = separator.join(left_parts)
        if separator != " " and left_text and right_parts:
            left_text += separator
        
        right_text = separator.join(right_parts)
        
        # Calculate character positions
        left_end = start_offset + len(left_text)
        right_start = start_offset + len(left_text)
        if separator != " " and left_text and right_parts:
            right_start -= len(separator)  # Adjust for added separator
        
        result = []
        if left_text.strip():
            result.append({
                "text": left_text,
                "start_offset": start_offset,
                "end_offset": left_end
            })
        
        if right_text.strip():
            result.append({
                "text": right_text,
                "start_offset": right_start,
                "end_offset": start_offset + len(text)
            })
        
        return result
    
    def _recursive_split(text: str, start_offset: int, sep_index: int) -> List[dict]:
        """
        Recursively split text only when necessary and track character offsets.
        """
        # Base case: if text is short enough, return as-is
        token_count = count_tokens_approximation(text, model_key)
        if token_count <= max_tokens:
            if text.strip():
                return [{"text": text, "start_offset": start_offset, "end_offset": start_offset + len(text)}]
            else:
                return []
        
        # If no more separators available, return the text as-is (can't split further)
        if sep_index >= len(separators):
            if text.strip():
                return [{"text": text, "start_offset": start_offset, "end_offset": start_offset + len(text)}]
            else:
                return []
        
        # Try to split with current separator around the middle
        separator = separators[sep_index]
        split_parts = _split_at_middle(text, separator, start_offset)
        
        # If splitting didn't help (only one chunk), try next separator
        if len(split_parts) <= 1:
            return _recursive_split(text, start_offset, sep_index + 1)
        
        # Process each split part recursively
        result = []
        for part_info in split_parts:
            part_text = part_info["text"]
            part_start = part_info["start_offset"]
            
            if not part_text.strip():
                continue
                
            part_tokens = count_tokens_approximation(part_text, model_key)
            if part_tokens > max_tokens:
                # This part is still too large, split it further
                sub_chunks = _recursive_split(part_text, part_start, sep_index)
                result.extend(sub_chunks)
            else:
                # This part is small enough
                result.append(part_info)
        
        return result
    
    # Start recursive splitting
    chunks = _recursive_split(document, 0, 0)
    
    # Filter out empty chunks and return
    return [chunk for chunk in chunks if chunk["text"].strip()]

def find_most_similar_chunk_recursive(keyword: str, document: str, model_key: str, 
                                    max_tokens: int = 100, separators: List[str] = None,
                                    metric: str = "cosine"):
    """
    Find the most similar chunk in a document to a keyword using recursive chunking strategy.
    Only works with sentence_transformers models.
    
    Args:
        keyword: The keyword to compare against
        document: The document to search in
        model_key: The model to use (must be sentence_transformers type)
        max_tokens: Maximum tokens per chunk
        separators: List of separators for recursive splitting
        metric: Similarity metric to use
    
    Returns:
        Tuple of (best_chunk_text, max_similarity, chunk_index, total_chunks, start_offset, end_offset)
    """
    config = MODEL_CONFIGS[model_key]
    if config["type"] != "sentence_transformers":
        raise ValueError(f"Model '{model_key}' must be of type 'sentence_transformers' for document similarity")
    
    # Chunk the document using recursive strategy
    chunks = chunk_document_recursive(document, model_key, max_tokens, separators)
    
    if not chunks:
        return "", 0.0, -1, 0, 0, 0
    
    # Get embeddings for keyword and all chunks
    model = models[model_key]
    keyword_embedding = model.encode([keyword])[0]
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    # Calculate similarities
    max_similarity = -1
    best_chunk_idx = 0
    
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity, _ = calculate_similarity_and_distance(
            keyword_embedding.tolist(), 
            chunk_embedding.tolist(), 
            metric
        )
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_chunk_idx = i
    
    best_chunk = chunks[best_chunk_idx]
    return (best_chunk["text"], max_similarity, best_chunk_idx, len(chunks), 
            best_chunk["start_offset"], best_chunk["end_offset"])

def find_most_similar_chunk(keyword: str, document: str, model_key: str, chunk_size_tokens: int = 512, 
                           overlap_percent: int = 50, metric: str = "cosine"):
    """
    Find the most similar chunk in a document to a keyword using sliding window strategy.
    Only works with sentence_transformers models.
    
    Args:
        keyword: The keyword to compare against
        document: The document to search in
        model_key: The model to use (must be sentence_transformers type)
        chunk_size_tokens: Size of each chunk in tokens
        overlap_percent: Percentage of overlap between chunks
        metric: Similarity metric to use
    
    Returns:
        Tuple of (best_chunk_text, max_similarity, chunk_index, total_chunks, start_offset, end_offset)
    """
    config = MODEL_CONFIGS[model_key]
    if config["type"] != "sentence_transformers":
        raise ValueError(f"Model '{model_key}' must be of type 'sentence_transformers' for document similarity")
    
    # Chunk the document using token-based chunking
    chunks = chunk_document_with_sliding_window_tokens(document, model_key, chunk_size_tokens, overlap_percent)
    
    if not chunks:
        return "", 0.0, -1, 0, 0, 0
    
    # Get embeddings for keyword and all chunks
    model = models[model_key]
    keyword_embedding = model.encode([keyword])[0]
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    # Calculate similarities
    max_similarity = -1
    best_chunk_idx = 0
    
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity, _ = calculate_similarity_and_distance(
            keyword_embedding.tolist(), 
            chunk_embedding.tolist(), 
            metric
        )
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_chunk_idx = i
    
    best_chunk = chunks[best_chunk_idx]
    return (best_chunk["text"], max_similarity, best_chunk_idx, len(chunks), 
            best_chunk["start_offset"], best_chunk["end_offset"])

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

@app.post("/api/document-similarity", response_model=DocumentSimilarityResponse)
async def document_similarity(request: DocumentSimilarityRequest):
    """Find the most similar chunk in a document to a keyword"""
    validate_model(request.model)
    
    # Validate empty inputs
    if not request.keyword or not request.keyword.strip():
        raise HTTPException(
            status_code=400,
            detail="keyword cannot be empty or whitespace only"
        )
    
    if not request.document or not request.document.strip():
        raise HTTPException(
            status_code=400,
            detail="document cannot be empty or whitespace only"
        )
    
    # Validate that model is sentence_transformers type
    config = MODEL_CONFIGS[request.model]
    if config["type"] != "sentence_transformers":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' must be of type 'sentence_transformers' for document similarity. Available sentence_transformers models: {[k for k, v in MODEL_CONFIGS.items() if v['type'] == 'sentence_transformers']}"
        )
    
    # Validate metric
    valid_metrics = ["cosine", "euclidean", "manhattan", "chebyshev"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{request.metric}'. Valid options: {valid_metrics}"
        )
    
    # Validate parameters
    if request.chunk_size_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="chunk_size_tokens must be positive"
        )
    
    if request.overlap_percent < 0 or request.overlap_percent > 100:
        raise HTTPException(
            status_code=400,
            detail="overlap_percent must be between 0 and 100"
        )
    
    start_time = time.time()
    
    try:
        best_chunk, similarity_score, chunk_index, total_chunks, start_offset, end_offset = find_most_similar_chunk(
            keyword=request.keyword,
            document=request.document,
            model_key=request.model,
            chunk_size_tokens=request.chunk_size_tokens,
            overlap_percent=request.overlap_percent,
            metric=request.metric
        )
        
        processing_time = time.time() - start_time
        logger.info(f"🔍 [{request.model}] Found best chunk ({chunk_index+1}/{total_chunks}) with {request.metric} similarity {similarity_score:.4f} in {processing_time:.3f}s")
        
        return DocumentSimilarityResponse(
            keyword=request.keyword,
            best_chunk=best_chunk,
            similarity_score=similarity_score,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_start_offset=start_offset,
            chunk_end_offset=end_offset,
            metric=request.metric,
            model=request.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document similarity: {str(e)}")

@app.post("/api/document-similarity-recursive", response_model=DocumentSimilarityResponse)
async def document_similarity_recursive(request: DocumentSimilarityRecursiveRequest):
    """Find the most similar chunk in a document to a keyword using recursive chunking strategy"""
    validate_model(request.model)
    
    # Validate empty inputs
    if not request.keyword or not request.keyword.strip():
        raise HTTPException(
            status_code=400,
            detail="keyword cannot be empty or whitespace only"
        )
    
    if not request.document or not request.document.strip():
        raise HTTPException(
            status_code=400,
            detail="document cannot be empty or whitespace only"
        )
    
    # Validate that model is sentence_transformers type
    config = MODEL_CONFIGS[request.model]
    if config["type"] != "sentence_transformers":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' must be of type 'sentence_transformers' for document similarity. Available sentence_transformers models: {[k for k, v in MODEL_CONFIGS.items() if v['type'] == 'sentence_transformers']}"
        )
    
    # Validate metric
    valid_metrics = ["cosine", "euclidean", "manhattan", "chebyshev"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{request.metric}'. Valid options: {valid_metrics}"
        )
    
    # Validate parameters
    if request.max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be positive"
        )
    
    if not request.separators or len(request.separators) == 0:
        raise HTTPException(
            status_code=400,
            detail="separators list cannot be empty"
        )
    
    start_time = time.time()
    
    try:
        best_chunk, similarity_score, chunk_index, total_chunks, start_offset, end_offset = find_most_similar_chunk_recursive(
            keyword=request.keyword,
            document=request.document,
            model_key=request.model,
            max_tokens=request.max_tokens,
            separators=request.separators,
            metric=request.metric
        )
        
        processing_time = time.time() - start_time
        logger.info(f"🔍🔄 [{request.model}] Found best recursive chunk ({chunk_index+1}/{total_chunks}) with {request.metric} similarity {similarity_score:.4f} in {processing_time:.3f}s")
        
        return DocumentSimilarityResponse(
            keyword=request.keyword,
            best_chunk=best_chunk,
            similarity_score=similarity_score,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_start_offset=start_offset,
            chunk_end_offset=end_offset,
            metric=request.metric,
            model=request.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recursive document similarity: {str(e)}")

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
    device_info = {
        "type": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
    return {
        "status": "healthy",
        "models": loaded_models,
        "total_models": len(MODEL_CONFIGS),
        "loaded_count": sum(loaded_models.values()),
        "device": device_info
    }

@app.get("/")
async def root():
    """Root endpoint with info"""
    return {
        "message": "Multi-Model Embedding Server",
        "version": "1.3.0",
        "features": ["Batch embeddings", "Multiple similarity metrics"],
        "available_models": list(MODEL_CONFIGS.keys()),
        "endpoints": {
            "embeddings": "/api/embeddings",
            "batch_embed": "/api/embed",
            "similarity": "/api/similarity",
            "batch_similarity": "/api/similarity/batch",
            "document_similarity": "/api/document-similarity",
            "document_similarity_recursive": "/api/document-similarity-recursive",
            "list_models": "/api/models",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_metrics": ["cosine", "euclidean", "manhattan", "chebyshev"]
    }
