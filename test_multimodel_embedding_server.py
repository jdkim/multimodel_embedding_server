import pytest
from starlette.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
from scipy.spatial.distance import cityblock, chebyshev

from multimodel_embedding_server import (
    app,
    calculate_similarity_and_distance,
    calculate_similarity_matrix,
    chunk_document_with_sliding_window_tokens,
    count_tokens_approximation,
    chunk_document_recursive,
    get_span_embeddings,
    validate_model,
    MODEL_CONFIGS,
    models,
    tokenizers,
)

# Test client - created per test to avoid startup issues
@pytest.fixture(scope="function")
def test_client():
    """Create test client for FastAPI app"""
    client = TestClient(app, raise_server_exceptions=False)
    yield client


# ============================================================================
# Similarity Calculation Tests
# ============================================================================

class TestSimilarityCalculations:
    """Tests for similarity and distance calculations"""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity, distance = calculate_similarity_and_distance(emb1, emb2, "cosine")
        assert abs(similarity - 1.0) < 0.001
        assert abs(distance - 0.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity, distance = calculate_similarity_and_distance(emb1, emb2, "cosine")
        assert abs(similarity - 0.0) < 0.001
        assert abs(distance - 1.0) < 0.001

    def test_euclidean_distance(self):
        """Test euclidean distance calculation"""
        emb1 = [0.0, 0.0, 0.0]
        emb2 = [3.0, 4.0, 0.0]
        similarity, distance = calculate_similarity_and_distance(emb1, emb2, "euclidean")
        assert abs(distance - 5.0) < 0.001
        assert similarity > 0  # Should be positive

    def test_manhattan_distance(self):
        """Test manhattan distance calculation"""
        emb1 = [0.0, 0.0]
        emb2 = [3.0, 4.0]
        similarity, distance = calculate_similarity_and_distance(emb1, emb2, "manhattan")
        assert abs(distance - 7.0) < 0.001
        assert similarity > 0  # Should be positive

    def test_chebyshev_distance(self):
        """Test chebyshev distance calculation"""
        emb1 = [0.0, 0.0]
        emb2 = [3.0, 4.0]
        similarity, distance = calculate_similarity_and_distance(emb1, emb2, "chebyshev")
        assert abs(distance - 4.0) < 0.001
        assert similarity > 0  # Should be positive

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError"""
        emb1 = [1.0, 0.0]
        emb2 = [0.0, 1.0]
        with pytest.raises(ValueError, match="Unsupported metric"):
            calculate_similarity_and_distance(emb1, emb2, "invalid_metric")

    def test_similarity_matrix_cosine(self):
        """Test similarity matrix calculation"""
        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        sim_matrix, dist_matrix = calculate_similarity_matrix(embeddings, "cosine")

        # Check diagonal (self-similarity should be 1.0)
        assert abs(sim_matrix[0][0] - 1.0) < 0.001
        assert abs(sim_matrix[1][1] - 1.0) < 0.001

        # Check identical vectors
        assert abs(sim_matrix[0][1] - 1.0) < 0.001

        # Check orthogonal vectors
        assert abs(sim_matrix[0][2] - 0.0) < 0.001

    def test_similarity_matrix_euclidean(self):
        """Test similarity matrix with euclidean metric"""
        embeddings = [
            [0.0, 0.0],
            [3.0, 4.0]
        ]
        sim_matrix, dist_matrix = calculate_similarity_matrix(embeddings, "euclidean")

        # Distance should be 5.0
        assert abs(dist_matrix[0][1] - 5.0) < 0.001


# ============================================================================
# Document Chunking Tests
# ============================================================================

class TestDocumentChunking:
    """Tests for document chunking functions"""

    @pytest.mark.skip(reason="Complex tokenizer mocking required for transformers models")
    def test_chunk_document_transformers_no_overlap(self):
        """Test chunking with transformers model without overlap"""
        # This test requires complex mocking of tokenizers which return encoding objects
        # Skipping for now as it requires deep mocking of HuggingFace transformers
        pass

    def test_chunk_document_sentence_transformers(self):
        """Test chunking with sentence transformers model"""
        document = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunk_document_with_sliding_window_tokens(
            document, "pubmedbert", chunk_size_tokens=3, overlap_percent=0
        )

        assert len(chunks) >= 2
        assert all('text' in chunk for chunk in chunks)
        assert all('start_offset' in chunk for chunk in chunks)
        assert all('end_offset' in chunk for chunk in chunks)

    def test_chunk_document_with_overlap(self):
        """Test chunking with overlap"""
        document = "word1 word2 word3 word4 word5 word6"
        chunks = chunk_document_with_sliding_window_tokens(
            document, "pubmedbert", chunk_size_tokens=3, overlap_percent=33
        )

        # With overlap, should have more chunks
        assert len(chunks) >= 2

    def test_chunk_document_small_document(self):
        """Test chunking when document is smaller than chunk size"""
        document = "short document"
        chunks = chunk_document_with_sliding_window_tokens(
            document, "pubmedbert", chunk_size_tokens=100, overlap_percent=0
        )

        assert len(chunks) == 1
        assert chunks[0]["text"] == document

    def test_chunk_document_invalid_overlap(self):
        """Test that invalid overlap raises ValueError"""
        with pytest.raises(ValueError, match="overlap_percent must be between 0 and 100"):
            chunk_document_with_sliding_window_tokens(
                "test", "pubmedbert", chunk_size_tokens=10, overlap_percent=150
            )

    @patch('multimodel_embedding_server.tokenizers')
    def test_count_tokens_transformers(self, mock_tokenizers):
        """Test token counting for transformers model"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizers.__getitem__.return_value = mock_tokenizer

        count = count_tokens_approximation("test text", "biomedbert")
        assert count == 5

    def test_count_tokens_sentence_transformers(self):
        """Test token counting for sentence transformers model"""
        text = "word1 word2 word3"
        count = count_tokens_approximation(text, "pubmedbert")
        assert count == 3

    def test_chunk_recursive_small_document(self):
        """Test recursive chunking with document smaller than max_tokens"""
        document = "Short text"
        chunks = chunk_document_recursive(document, "pubmedbert", max_tokens=100)

        assert len(chunks) == 1
        assert chunks[0]["text"] == document

    def test_chunk_recursive_splits_on_separators(self):
        """Test that recursive chunking respects separator hierarchy"""
        document = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunk_document_recursive(
            document, "pubmedbert", max_tokens=5, separators=["\n\n", ". "]
        )

        # Should split into multiple chunks
        assert len(chunks) >= 2


# ============================================================================
# Span Embeddings Tests
# ============================================================================

class TestSpanEmbeddings:
    """Tests for span embeddings functionality"""

    @patch('multimodel_embedding_server.models')
    def test_span_embeddings_endpoint_empty_text(self, mock_models, test_client):
        """Test span embeddings with empty text"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "",
                "spans": [{"begin": 0, "end": 5}],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @patch('multimodel_embedding_server.models')
    def test_span_embeddings_endpoint_empty_spans(self, mock_models, test_client):
        """Test span embeddings with empty spans list"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "some text",
                "spans": [],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @patch('multimodel_embedding_server.models')
    def test_span_embeddings_endpoint_negative_begin(self, mock_models, test_client):
        """Test span embeddings with negative begin offset"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "some text",
                "spans": [{"begin": -1, "end": 5}],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "non-negative" in response.json()["detail"]

    @patch('multimodel_embedding_server.models')
    def test_span_embeddings_endpoint_begin_gte_end(self, mock_models, test_client):
        """Test span embeddings with begin >= end"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "some text",
                "spans": [{"begin": 5, "end": 5}],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "less than" in response.json()["detail"]

    @patch('multimodel_embedding_server.models')
    def test_span_embeddings_endpoint_end_exceeds_text_length(self, mock_models, test_client):
        """Test span embeddings with end exceeding text length"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "short",
                "spans": [{"begin": 0, "end": 100}],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "exceeds text length" in response.json()["detail"]

    def test_span_embeddings_endpoint_invalid_model(self, test_client):
        """Test span embeddings with invalid model"""
        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "some text",
                "spans": [{"begin": 0, "end": 4}],
                "model": "nonexistent_model"
            }
        )
        assert response.status_code == 400

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.get_span_embeddings')
    def test_span_embeddings_endpoint_success(self, mock_get_span, mock_models, test_client):
        """Test successful span embeddings endpoint call"""
        mock_models.__contains__.return_value = True
        mock_get_span.return_value = [
            {
                'begin': 23,
                'end': 33,
                'span_text': 'galectin-3',
                'embedding': [0.1] * 768,
                'tokens': ['gal', '##ect', '##in', '-', '3']
            }
        ]

        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "NMR-based insight into galectin-3 binding",
                "spans": [{"begin": 23, "end": 33}],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "spans" in data
        assert "model" in data
        assert len(data["spans"]) == 1
        assert data["spans"][0]["span_text"] == "galectin-3"
        assert data["spans"][0]["begin"] == 23
        assert data["spans"][0]["end"] == 33
        assert "embedding" in data["spans"][0]
        assert "tokens" in data["spans"][0]

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.get_span_embeddings')
    def test_span_embeddings_multiple_spans(self, mock_get_span, mock_models, test_client):
        """Test span embeddings with multiple spans"""
        mock_models.__contains__.return_value = True
        mock_get_span.return_value = [
            {
                'begin': 0,
                'end': 3,
                'span_text': 'NMR',
                'embedding': [0.1] * 768,
                'tokens': ['NMR']
            },
            {
                'begin': 23,
                'end': 33,
                'span_text': 'galectin-3',
                'embedding': [0.2] * 768,
                'tokens': ['gal', '##ect', '##in', '-', '3']
            }
        ]

        response = test_client.post(
            "/api/span-embeddings",
            json={
                "text": "NMR-based insight into galectin-3 binding",
                "spans": [
                    {"begin": 0, "end": 3},
                    {"begin": 23, "end": 33}
                ],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["spans"]) == 2
        assert data["spans"][0]["span_text"] == "NMR"
        assert data["spans"][1]["span_text"] == "galectin-3"


# ============================================================================
# Span Similarity Tests
# ============================================================================

class TestSpanSimilarity:
    """Tests for span similarity functionality"""

    @patch('multimodel_embedding_server.models')
    def test_span_similarity_empty_query(self, mock_models, test_client):
        """Test span similarity with empty query"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-similarity",
            json={
                "text": "some text",
                "spans": [{"begin": 0, "end": 4}],
                "query": "",
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400
        assert "query" in response.json()["detail"].lower()

    @patch('multimodel_embedding_server.models')
    def test_span_similarity_invalid_metric(self, mock_models, test_client):
        """Test span similarity with invalid metric"""
        mock_models.__contains__.return_value = True
        response = test_client.post(
            "/api/span-similarity",
            json={
                "text": "some text",
                "spans": [{"begin": 0, "end": 4}],
                "query": "test query",
                "model": "pubmedbert",
                "metric": "invalid_metric"
            }
        )
        assert response.status_code == 400
        assert "metric" in response.json()["detail"].lower()

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.get_span_embeddings')
    @patch('multimodel_embedding_server.get_embedding_sentence_transformers')
    def test_span_similarity_success(self, mock_get_query_emb, mock_get_span, mock_models, test_client):
        """Test successful span similarity call"""
        mock_models.__contains__.return_value = True
        mock_get_span.return_value = [
            {
                'begin': 23,
                'end': 33,
                'span_text': 'galectin-3',
                'embedding': [1.0] + [0.0] * 767,
                'tokens': ['galectin', '-', '3']
            }
        ]
        mock_get_query_emb.return_value = [1.0] + [0.0] * 767  # Same embedding = similarity 1.0

        response = test_client.post(
            "/api/span-similarity",
            json={
                "text": "NMR-based insight into galectin-3 binding",
                "spans": [{"begin": 23, "end": 33}],
                "query": "galectin-3",
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "query" in data
        assert "spans" in data
        assert "metric" in data
        assert len(data["spans"]) == 1
        assert data["spans"][0]["span_text"] == "galectin-3"
        assert "similarity" in data["spans"][0]
        assert "distance" in data["spans"][0]
        assert data["spans"][0]["similarity"] > 0.99  # Should be ~1.0

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.get_span_embeddings')
    @patch('multimodel_embedding_server.get_embedding_sentence_transformers')
    def test_span_similarity_multiple_spans(self, mock_get_query_emb, mock_get_span, mock_models, test_client):
        """Test span similarity with multiple spans"""
        mock_models.__contains__.return_value = True
        mock_get_span.return_value = [
            {
                'begin': 0,
                'end': 3,
                'span_text': 'NMR',
                'embedding': [0.0, 1.0] + [0.0] * 766,
                'tokens': ['NMR']
            },
            {
                'begin': 23,
                'end': 33,
                'span_text': 'galectin-3',
                'embedding': [1.0, 0.0] + [0.0] * 766,
                'tokens': ['galectin', '-', '3']
            }
        ]
        mock_get_query_emb.return_value = [1.0, 0.0] + [0.0] * 766

        response = test_client.post(
            "/api/span-similarity",
            json={
                "text": "NMR-based insight into galectin-3 binding",
                "spans": [
                    {"begin": 0, "end": 3},
                    {"begin": 23, "end": 33}
                ],
                "query": "galectin-3",
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["spans"]) == 2
        # galectin-3 should have higher similarity
        assert data["spans"][1]["similarity"] > data["spans"][0]["similarity"]


# ============================================================================
# Model Validation Tests
# ============================================================================

class TestModelValidation:
    """Tests for model validation"""

    def test_validate_model_invalid(self):
        """Test that invalid model raises HTTPException"""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_model("nonexistent_model")
        assert exc_info.value.status_code == 400

    def test_validate_model_not_loaded(self):
        """Test that unloaded model raises HTTPException"""
        from fastapi import HTTPException
        # Clear models dict to simulate unloaded state
        original_models = models.copy()
        models.clear()

        try:
            with pytest.raises(HTTPException) as exc_info:
                validate_model("pubmedbert")
            assert exc_info.value.status_code == 503
        finally:
            # Restore original models
            models.update(original_models)


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns correct information"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Multi-Model Embedding Server"
        assert data["version"] == "1.3.0"
        assert "available_models" in data
        assert "endpoints" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models" in data
        assert "total_models" in data

    def test_list_models_endpoint(self, test_client):
        """Test models listing endpoint"""
        response = test_client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

        # Check model structure
        model = data["models"][0]
        assert "name" in model
        assert "full_name" in model
        assert "type" in model
        assert "description" in model

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.MODEL_CONFIGS')
    def test_embeddings_endpoint_invalid_model(self, mock_configs, mock_models, test_client):
        """Test embeddings endpoint with invalid model"""
        mock_configs.__getitem__.side_effect = KeyError("invalid_model")

        response = test_client.post(
            "/api/embeddings",
            json={"prompt": "test", "model": "invalid_model"}
        )
        assert response.status_code == 400

    @patch('multimodel_embedding_server.models')
    @patch('multimodel_embedding_server.get_embedding_sentence_transformers')
    def test_embeddings_endpoint_success(self, mock_get_embedding, mock_models, test_client):
        """Test successful embeddings endpoint call"""
        mock_models.__contains__.return_value = True
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]

        response = test_client.post(
            "/api/embeddings",
            json={"prompt": "test text", "model": "pubmedbert"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "model" in data
        assert data["model"] == "pubmedbert"

    def test_similarity_endpoint_invalid_metric(self, test_client):
        """Test similarity endpoint with invalid metric"""
        response = test_client.post(
            "/api/similarity",
            json={
                "text1": "hello",
                "text2": "world",
                "model": "pubmedbert",
                "metric": "invalid_metric"
            }
        )
        assert response.status_code == 400

    def test_batch_similarity_endpoint_insufficient_texts(self, test_client):
        """Test batch similarity with less than 2 texts"""
        response = test_client.post(
            "/api/similarity/batch",
            json={
                "texts": ["only one text"],
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400

    def test_document_similarity_empty_keyword(self, test_client):
        """Test document similarity with empty keyword"""
        response = test_client.post(
            "/api/document-similarity",
            json={
                "keyword": "",
                "document": "some text",
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400

    def test_document_similarity_empty_document(self, test_client):
        """Test document similarity with empty document"""
        response = test_client.post(
            "/api/document-similarity",
            json={
                "keyword": "test",
                "document": "",
                "model": "pubmedbert"
            }
        )
        assert response.status_code == 400

    def test_document_similarity_invalid_chunk_size(self, test_client):
        """Test document similarity with invalid chunk size"""
        response = test_client.post(
            "/api/document-similarity",
            json={
                "keyword": "test",
                "document": "some text",
                "model": "pubmedbert",
                "chunk_size_tokens": -1
            }
        )
        assert response.status_code == 400

    def test_document_similarity_invalid_overlap(self, test_client):
        """Test document similarity with invalid overlap percent"""
        response = test_client.post(
            "/api/document-similarity",
            json={
                "keyword": "test",
                "document": "some text",
                "model": "pubmedbert",
                "overlap_percent": 150
            }
        )
        assert response.status_code == 400

    def test_document_similarity_wrong_model_type(self, test_client):
        """Test document similarity with transformers model (requires sentence_transformers)"""
        response = test_client.post(
            "/api/document-similarity",
            json={
                "keyword": "test",
                "document": "some text",
                "model": "biomedbert"  # This is a transformers model
            }
        )
        assert response.status_code == 400
        assert "sentence_transformers" in response.json()["detail"]

    def test_document_similarity_recursive_empty_separators(self, test_client):
        """Test recursive document similarity with empty separators"""
        response = test_client.post(
            "/api/document-similarity-recursive",
            json={
                "keyword": "test",
                "document": "some text",
                "model": "pubmedbert",
                "separators": []
            }
        )
        assert response.status_code == 400

    def test_document_similarity_recursive_invalid_max_tokens(self, test_client):
        """Test recursive document similarity with invalid max_tokens"""
        response = test_client.post(
            "/api/document-similarity-recursive",
            json={
                "keyword": "test",
                "document": "some text",
                "model": "pubmedbert",
                "max_tokens": -1
            }
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=multimodel_embedding_server", "--cov-report=term-missing"])
