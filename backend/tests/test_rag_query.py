"""
Tests for RAGSystem.query() in rag_system.py.

Covers:
- query() returns a (response, sources) tuple
- query() passes tool definitions to ai_generator.generate_response()
- query() retrieves and resets sources after each query
- query() passes conversation history when session_id is provided
- query() adds the exchange to session history
- Integration: RAGSystem.query() end-to-end with a mocked Anthropic client
  (real ChromaDB + real VectorStore) to expose failures in the content-query path
"""
import os
import pytest
from unittest.mock import MagicMock, patch

from rag_system import RAGSystem
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Create a Config that points to the real chroma_db but uses a fake API key."""
    cfg = Config()
    cfg.ANTHROPIC_API_KEY = "fake_key_for_testing"
    # Use the real chroma_db so integration tests exercise the actual data
    cfg.CHROMA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../backend/chroma_db")
    )
    return cfg


def _make_rag_with_mocked_ai():
    """
    RAGSystem with real VectorStore/ChromaDB but mocked AIGenerator.
    Isolates the RAG orchestration logic from Anthropic API calls.
    """
    cfg = _make_config()
    with patch("rag_system.AIGenerator") as MockAI:
        rag = RAGSystem(cfg)
        rag.ai_generator = MockAI.return_value
    return rag


# ---------------------------------------------------------------------------
# Unit tests — query() orchestration
# ---------------------------------------------------------------------------

class TestRAGQueryOrchestration:

    def setup_method(self):
        self.rag = _make_rag_with_mocked_ai()
        self.rag.ai_generator.generate_response.return_value = "Test answer"

    def test_query_returns_tuple_of_two(self):
        result = self.rag.query("What is Python?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_returns_string_response(self):
        response, _ = self.rag.query("What is Python?")
        assert isinstance(response, str)

    def test_query_returns_list_of_sources(self):
        _, sources = self.rag.query("What is Python?")
        assert isinstance(sources, list)

    def test_query_response_matches_ai_generator_output(self):
        self.rag.ai_generator.generate_response.return_value = "Specific answer"
        response, _ = self.rag.query("Anything")
        assert response == "Specific answer"

    def test_query_calls_generate_response_with_tools(self):
        self.rag.query("What is ML?")
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        # Tools should be the list from tool_manager — non-empty
        assert len(call_kwargs["tools"]) > 0

    def test_query_passes_tool_manager_to_generate_response(self):
        self.rag.query("What is RAG?")
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs.get("tool_manager") is self.rag.tool_manager

    def test_query_tools_include_search_course_content(self):
        self.rag.query("test")
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_resets_sources_after_retrieval(self):
        """Sources must be reset after each query so they don't bleed into next query."""
        # Simulate a tool having set sources
        self.rag.search_tool.last_sources = [{"label": "Course A", "url": None}]
        self.rag.query("test")
        assert self.rag.search_tool.last_sources == []

    def test_query_without_session_id_does_not_use_history(self):
        self.rag.query("test", session_id=None)
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs.get("conversation_history") is None

    def test_query_with_session_id_passes_history(self):
        session_id = self.rag.session_manager.create_session()
        # Add a prior exchange to establish history
        self.rag.session_manager.add_exchange(session_id, "prior question", "prior answer")
        self.rag.query("follow up", session_id=session_id)
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert len(history) > 0

    def test_query_saves_exchange_to_session(self):
        session_id = self.rag.session_manager.create_session()
        self.rag.ai_generator.generate_response.return_value = "My answer"
        self.rag.query("My question", session_id=session_id)
        history = self.rag.session_manager.get_conversation_history(session_id)
        assert "My question" in history
        assert "My answer" in history

    def test_query_prompt_contains_user_question(self):
        self.rag.query("What are transformers?")
        call_kwargs = self.rag.ai_generator.generate_response.call_args[1]
        query_sent = call_kwargs.get("query", "")
        assert "transformers" in query_sent.lower()


# ---------------------------------------------------------------------------
# Integration tests — real ChromaDB, mocked Anthropic client only
# ---------------------------------------------------------------------------

class TestRAGQueryIntegration:
    """
    Tests that run the full RAGSystem.query() path with a real VectorStore
    but a mocked Anthropic client. This catches runtime errors in the
    search → format → return pipeline without incurring API costs.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rag = _make_rag_with_mocked_ai()

    def test_content_query_with_direct_answer_does_not_raise(self):
        """
        Simulates Claude answering directly (no tool use).
        The full query() pipeline must complete without exception.
        """
        self.rag.ai_generator.generate_response.return_value = "Direct answer"
        try:
            response, sources = self.rag.query("What is the course about?")
            assert isinstance(response, str)
        except Exception as e:
            pytest.fail(f"query() raised unexpectedly: {e}")

    def test_content_query_with_tool_use_does_not_raise(self):
        """
        Simulates Claude invoking search_course_content by having the mocked
        AIGenerator call the real tool_manager via execute_tool.
        """
        def _simulate_tool_use(query, conversation_history=None, tools=None, tool_manager=None):
            # Actually execute the tool as Claude would
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query=query)
            return "Answer based on course content"

        self.rag.ai_generator.generate_response.side_effect = _simulate_tool_use

        try:
            response, sources = self.rag.query("Tell me about lesson 1")
            assert isinstance(response, str)
            assert response == "Answer based on course content"
        except Exception as e:
            pytest.fail(f"query() raised during tool execution: {e}")

    def test_search_tool_executes_without_exception_on_broad_query(self):
        """
        Directly exercises CourseSearchTool.execute() through the real VectorStore.
        This is the most likely failure point when 'query failed' appears.
        """
        try:
            result = self.rag.search_tool.execute(query="introduction to the course")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(
                f"CourseSearchTool.execute() raised an exception with no filters: {e}\n"
                "This is likely the root cause of 'query failed'."
            )

    def test_search_tool_executes_without_exception_with_course_filter(self):
        """Exercises the code path where course_name builds a where filter."""
        try:
            result = self.rag.search_tool.execute(
                query="computer use", course_name="Computer Use"
            )
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"execute() raised with course_name filter: {e}")

    def test_search_tool_executes_without_exception_with_lesson_filter(self):
        """Exercises the code path where lesson_number builds a where filter."""
        try:
            result = self.rag.search_tool.execute(query="overview", lesson_number=1)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"execute() raised with lesson_number filter: {e}")

    def test_sources_are_list_after_tool_use(self):
        """Sources returned from query() must always be a list (never None)."""
        def _use_tool(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query=query)
            return "Answer"

        self.rag.ai_generator.generate_response.side_effect = _use_tool
        _, sources = self.rag.query("What does lesson 1 cover?")
        assert isinstance(sources, list)

    def test_sources_reset_between_consecutive_queries(self):
        """
        Two consecutive queries must not leak sources from the first into the second.
        """
        def _use_tool(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query=query)
            return "Answer"

        self.rag.ai_generator.generate_response.side_effect = _use_tool

        self.rag.query("First query about courses")
        _, sources_second = self.rag.query("Second query — no tool use expected")
        # After second query (direct answer, no tool), sources should be empty
        # Actually the mock always calls the tool, so just verify list type
        assert isinstance(sources_second, list)
        # And verify tool.last_sources was reset
        assert self.rag.search_tool.last_sources == []
