"""
API endpoint tests for the RAG system FastAPI app.

Endpoints covered:
  POST /api/query
  GET  /api/courses
  POST /api/session/reset
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_query_creates_session_when_none_provided(self, client, mock_rag_system, sample_query_payload):
        """When no session_id is sent the endpoint creates a new one and returns it."""
        response = client.post("/api/query", json=sample_query_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_reuses_provided_session(self, client, mock_rag_system, sample_query_payload_with_session):
        """An explicit session_id must be forwarded to RAGSystem.query and echoed back."""
        response = client.post("/api/query", json=sample_query_payload_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_42"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            sample_query_payload_with_session["query"], "session_42"
        )

    def test_query_returns_answer_and_sources(self, client, sample_query_payload):
        """Response body must contain answer text and a non-empty sources list."""
        response = client.post("/api/query", json=sample_query_payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert data["sources"][0]["label"] == "Python Basics - Lesson 1"

    def test_query_missing_required_field_returns_422(self, client):
        """Omitting the required 'query' field must trigger a validation error."""
        response = client.post("/api/query", json={"session_id": "session_1"})
        assert response.status_code == 422

    def test_query_empty_string_is_accepted(self, client):
        """An empty string is a valid (though unusual) query value."""
        response = client.post("/api/query", json={"query": ""})
        assert response.status_code == 200

    def test_query_rag_exception_returns_500(self, client, mock_rag_system, sample_query_payload):
        """If RAGSystem.query raises, the endpoint must return HTTP 500."""
        mock_rag_system.query.side_effect = RuntimeError("ChromaDB unavailable")

        response = client.post("/api/query", json=sample_query_payload)

        assert response.status_code == 500
        assert "ChromaDB unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_courses_returns_stats(self, client):
        """GET /api/courses should return total_courses count and title list."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert "Python Basics" in data["course_titles"]
        assert "Advanced RAG" in data["course_titles"]

    def test_courses_returns_correct_count(self, client):
        """total_courses must match the length of course_titles."""
        response = client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == len(data["course_titles"])

    def test_courses_analytics_exception_returns_500(self, client, mock_rag_system):
        """If get_course_analytics raises, the endpoint must return HTTP 500."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/session/reset
# ---------------------------------------------------------------------------

class TestSessionResetEndpoint:

    def test_reset_returns_success(self, client):
        """A valid session reset must return {"success": true}."""
        response = client.post("/api/session/reset", json={"session_id": "session_1"})

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_reset_calls_clear_session(self, client, mock_rag_system):
        """The endpoint must delegate to session_manager.clear_session with the given ID."""
        client.post("/api/session/reset", json={"session_id": "session_99"})

        mock_rag_system.session_manager.clear_session.assert_called_once_with("session_99")

    def test_reset_missing_session_id_returns_422(self, client):
        """Omitting session_id must trigger a validation error."""
        response = client.post("/api/session/reset", json={})
        assert response.status_code == 422
