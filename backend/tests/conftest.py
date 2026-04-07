"""
Shared fixtures for the RAG system test suite.

A separate test app is created here to avoid import-time side effects from
app.py (RAGSystem initialisation touching ChromaDB, StaticFiles mounting a
frontend directory that does not exist in CI).
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Pydantic models (mirrors app.py — kept local so the test app is self-contained)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    label: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


class SessionResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_app(rag_system: MagicMock) -> FastAPI:
    """Return a FastAPI instance wired to the given mock RAG system.

    This mirrors the endpoints defined in app.py without any static-file
    mounts or startup document loading.
    """
    app = FastAPI(title="Test RAG App")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/reset")
    async def reset_session(request: SessionResetRequest):
        rag_system.session_manager.clear_session(request.session_id)
        return {"success": True}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """A fully-mocked RAGSystem with sensible default return values."""
    mock = MagicMock()

    # session_manager behaviour
    mock.session_manager.create_session.return_value = "session_1"
    mock.session_manager.clear_session.return_value = None

    # query returns (answer, sources)
    mock.query.return_value = (
        "This course covers Python fundamentals.",
        [{"label": "Python Basics - Lesson 1", "url": "https://example.com/lesson1"}],
    )

    # analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Advanced RAG"],
    }

    return mock


@pytest.fixture
def client(mock_rag_system):
    """TestClient backed by the test app and a fresh mock RAG system."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def sample_query_payload():
    """Minimal valid payload for POST /api/query."""
    return {"query": "What is covered in the Python course?"}


@pytest.fixture
def sample_query_payload_with_session():
    """Valid payload for POST /api/query that includes an existing session."""
    return {"query": "Tell me more.", "session_id": "session_42"}
