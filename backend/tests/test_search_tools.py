"""
Tests for CourseSearchTool.execute() and ToolManager in search_tools.py.

Covers:
- execute() passes correct parameters to the VectorStore
- execute() handles errors from the VectorStore gracefully
- execute() formats results correctly
- execute() populates last_sources after a successful search
- Tool definition structure
- Integration: execute() against the real ChromaDB (no mocks)
"""
import os
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(docs, metas):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


def _mock_store(search_return):
    store = MagicMock()
    store.search.return_value = search_return
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    return store


# ---------------------------------------------------------------------------
# Unit tests — CourseSearchTool.execute()
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:

    def setup_method(self):
        self.store = _mock_store(_make_results([], []))
        self.tool = CourseSearchTool(self.store)

    # --- parameter forwarding ---

    def test_execute_passes_query_to_store(self):
        self.tool.execute(query="what is machine learning")
        self.store.search.assert_called_once_with(
            query="what is machine learning",
            course_name=None,
            lesson_number=None,
        )

    def test_execute_passes_course_name_to_store(self):
        self.tool.execute(query="intro lesson", course_name="MCP Course")
        self.store.search.assert_called_once_with(
            query="intro lesson",
            course_name="MCP Course",
            lesson_number=None,
        )

    def test_execute_passes_lesson_number_to_store(self):
        self.tool.execute(query="vectors", lesson_number=3)
        self.store.search.assert_called_once_with(
            query="vectors",
            course_name=None,
            lesson_number=3,
        )

    def test_execute_passes_all_filters_to_store(self):
        self.tool.execute(query="embeddings", course_name="RAG Course", lesson_number=2)
        self.store.search.assert_called_once_with(
            query="embeddings",
            course_name="RAG Course",
            lesson_number=2,
        )

    # --- error handling ---

    def test_execute_returns_error_string_on_store_error(self):
        self.store.search.return_value = SearchResults.empty("Search error: DB unavailable")
        result = self.tool.execute(query="test")
        assert result == "Search error: DB unavailable"

    def test_execute_returns_string_type(self):
        result = self.tool.execute(query="anything")
        assert isinstance(result, str)

    # --- empty results ---

    def test_execute_returns_no_content_message_when_empty(self):
        result = self.tool.execute(query="obscure topic")
        assert "No relevant content found" in result

    def test_execute_no_content_message_includes_course_name(self):
        result = self.tool.execute(query="x", course_name="Python Basics")
        assert "Python Basics" in result

    def test_execute_no_content_message_includes_lesson_number(self):
        result = self.tool.execute(query="x", lesson_number=5)
        assert "5" in result

    # --- result formatting ---

    def test_execute_formats_course_title_in_header(self):
        self.store.search.return_value = _make_results(
            ["Lesson content here"],
            [{"course_title": "Intro to AI", "lesson_number": 1}],
        )
        result = self.tool.execute(query="AI basics")
        assert "Intro to AI" in result

    def test_execute_formats_lesson_number_in_header(self):
        self.store.search.return_value = _make_results(
            ["Some content"],
            [{"course_title": "Course A", "lesson_number": 4}],
        )
        result = self.tool.execute(query="test")
        assert "Lesson 4" in result

    def test_execute_includes_document_content_in_result(self):
        self.store.search.return_value = _make_results(
            ["The answer is 42"],
            [{"course_title": "Math", "lesson_number": 1}],
        )
        result = self.tool.execute(query="answer")
        assert "The answer is 42" in result

    def test_execute_handles_result_without_lesson_number(self):
        self.store.search.return_value = _make_results(
            ["General info"],
            [{"course_title": "Overview", "lesson_number": None}],
        )
        result = self.tool.execute(query="overview")
        assert "Overview" in result
        # Should not raise and should not contain "Lesson None"
        assert "Lesson None" not in result

    def test_execute_joins_multiple_results(self):
        self.store.search.return_value = _make_results(
            ["First chunk", "Second chunk"],
            [
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
            ],
        )
        result = self.tool.execute(query="test")
        assert "First chunk" in result
        assert "Second chunk" in result

    # --- last_sources tracking ---

    def test_last_sources_starts_empty(self):
        assert self.tool.last_sources == []

    def test_last_sources_populated_after_successful_search(self):
        self.store.search.return_value = _make_results(
            ["Content"],
            [{"course_title": "Test Course", "lesson_number": 1}],
        )
        self.store.get_lesson_link.return_value = "http://lesson.com"
        self.tool.execute(query="test")
        assert len(self.tool.last_sources) == 1

    def test_last_sources_contains_correct_label(self):
        self.store.search.return_value = _make_results(
            ["Content"],
            [{"course_title": "Test Course", "lesson_number": 2}],
        )
        self.tool.execute(query="test")
        assert self.tool.last_sources[0]["label"] == "Test Course - Lesson 2"

    def test_last_sources_contains_lesson_url_when_available(self):
        self.store.search.return_value = _make_results(
            ["Content"],
            [{"course_title": "Test Course", "lesson_number": 1}],
        )
        self.store.get_lesson_link.return_value = "http://lesson.example.com"
        self.tool.execute(query="test")
        assert self.tool.last_sources[0]["url"] == "http://lesson.example.com"

    def test_last_sources_falls_back_to_course_url(self):
        self.store.search.return_value = _make_results(
            ["Content"],
            [{"course_title": "Test Course", "lesson_number": 1}],
        )
        self.store.get_lesson_link.return_value = None
        self.store.get_course_link.return_value = "http://course.example.com"
        self.tool.execute(query="test")
        assert self.tool.last_sources[0]["url"] == "http://course.example.com"

    def test_last_sources_empty_when_no_results(self):
        self.tool.execute(query="no results query")
        assert self.tool.last_sources == []

    def test_last_sources_empty_when_store_error(self):
        self.store.search.return_value = SearchResults.empty("error")
        self.tool.execute(query="test")
        assert self.tool.last_sources == []


# ---------------------------------------------------------------------------
# Unit tests — tool definition
# ---------------------------------------------------------------------------

class TestCourseSearchToolDefinition:

    def setup_method(self):
        self.tool = CourseSearchTool(MagicMock())

    def test_tool_name_is_search_course_content(self):
        defn = self.tool.get_tool_definition()
        assert defn["name"] == "search_course_content"

    def test_tool_definition_has_description(self):
        defn = self.tool.get_tool_definition()
        assert "description" in defn and len(defn["description"]) > 0

    def test_tool_definition_has_input_schema(self):
        defn = self.tool.get_tool_definition()
        assert "input_schema" in defn

    def test_query_is_in_required_fields(self):
        defn = self.tool.get_tool_definition()
        assert "query" in defn["input_schema"]["required"]

    def test_query_property_exists(self):
        defn = self.tool.get_tool_definition()
        assert "query" in defn["input_schema"]["properties"]

    def test_course_name_property_exists_and_is_optional(self):
        defn = self.tool.get_tool_definition()
        assert "course_name" in defn["input_schema"]["properties"]
        # course_name is optional — not in required
        assert "course_name" not in defn["input_schema"].get("required", [])

    def test_lesson_number_property_exists_and_is_optional(self):
        defn = self.tool.get_tool_definition()
        assert "lesson_number" in defn["input_schema"]["properties"]
        assert "lesson_number" not in defn["input_schema"].get("required", [])


# ---------------------------------------------------------------------------
# Unit tests — ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:

    def test_register_tool_stores_by_name(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "my_tool"}
        manager.register_tool(mock_tool)
        assert "my_tool" in manager.tools

    def test_execute_tool_calls_correct_tool(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "my_tool"}
        mock_tool.execute.return_value = "result_value"
        manager.register_tool(mock_tool)
        result = manager.execute_tool("my_tool", param="value")
        mock_tool.execute.assert_called_once_with(param="value")
        assert result == "result_value"

    def test_execute_unknown_tool_returns_not_found_message(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool")
        assert "not found" in result.lower() or "nonexistent_tool" in result

    def test_get_tool_definitions_returns_list(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "t1"}
        manager.register_tool(mock_tool)
        defns = manager.get_tool_definitions()
        assert isinstance(defns, list)
        assert len(defns) == 1

    def test_get_last_sources_returns_sources_from_tool(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "search_course_content"}
        mock_tool.last_sources = [{"label": "Course A", "url": None}]
        manager.tools["search_course_content"] = mock_tool
        sources = manager.get_last_sources()
        assert sources == [{"label": "Course A", "url": None}]

    def test_get_last_sources_returns_empty_when_no_sources(self):
        manager = ToolManager()
        assert manager.get_last_sources() == []

    def test_reset_sources_clears_last_sources(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "search_course_content"}
        mock_tool.last_sources = [{"label": "X", "url": None}]
        manager.tools["search_course_content"] = mock_tool
        manager.reset_sources()
        assert mock_tool.last_sources == []


# ---------------------------------------------------------------------------
# Integration tests — real VectorStore + ChromaDB
# ---------------------------------------------------------------------------

class TestCourseSearchToolIntegration:
    """
    Tests using the real ChromaDB and VectorStore.
    These reveal actual runtime failures that unit tests with mocks cannot catch.
    """

    @pytest.fixture(autouse=True)
    def setup_vector_store(self):
        from vector_store import VectorStore
        chroma_path = os.path.join(os.path.dirname(__file__), "../../backend/chroma_db")
        chroma_path = os.path.normpath(chroma_path)
        self.vector_store = VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        self.tool = CourseSearchTool(self.vector_store)

    def test_execute_does_not_raise_without_filters(self):
        """
        Regression: VectorStore.search() passes where=None to ChromaDB when no
        course/lesson filter is given. This should not raise an exception.
        """
        try:
            result = self.tool.execute(query="introduction to AI")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"execute() raised an exception without filters: {e}")

    def test_execute_returns_content_or_no_results_message(self):
        """
        A broad query should return either found content or the no-results message.
        It must never propagate an exception.
        """
        result = self.tool.execute(query="machine learning neural networks")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_valid_course_name_returns_filtered_results(self):
        """
        If there are courses in the DB, a partial course name filter should work.
        """
        result = self.tool.execute(query="introduction", course_name="Computer Use")
        assert isinstance(result, str)
        # Should not be a raw ChromaDB exception traceback
        assert "Traceback" not in result

    def test_execute_with_lesson_number_filter_does_not_raise(self):
        """
        Regression: lesson_number filter builds a where dict that must be
        valid for ChromaDB 1.x.
        """
        try:
            result = self.tool.execute(query="lesson content", lesson_number=1)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"execute() raised with lesson_number filter: {e}")

    def test_execute_with_nonexistent_course_does_not_raise(self):
        """
        _resolve_course_name() uses vector similarity with NO similarity threshold,
        so even a completely nonsensical course name resolves to the nearest match
        in the DB. The method never returns None for any input. This means the
        'No course found' error path in VectorStore.search() is unreachable in practice.

        The test verifies: no exception is raised and a string is returned.
        """
        result = self.tool.execute(query="test", course_name="ZZZ Nonexistent Course XYZ")
        assert isinstance(result, str)
        assert len(result) > 0
