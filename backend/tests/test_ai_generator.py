"""
Tests for AIGenerator in ai_generator.py.

Covers:
- generate_response() returns text directly when stop_reason != "tool_use"
- generate_response() delegates to _handle_tool_execution() when stop_reason == "tool_use"
- _handle_tool_execution() calls tool_manager.execute_tool() with correct name and inputs
- _handle_tool_execution() adds tool results to messages with the correct structure
- The second API call (follow-up) does NOT include the tools parameter
- The second API call preserves the system prompt
- Edge case: stop_reason == "tool_use" but tool_manager is None
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------

def _text_response(text, stop_reason="end_turn"):
    """A response where Claude answered directly (no tool use)."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = [block]
    return resp


def _tool_use_response(tool_name, tool_input, tool_id="toolu_abc123"):
    """A response where Claude wants to call a tool."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id

    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def _make_ai(model="claude-test") -> AIGenerator:
    """Create an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic"):
        ai = AIGenerator(api_key="fake_key", model=model)
    return ai


# ---------------------------------------------------------------------------
# Direct response (no tool use)
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def setup_method(self):
        self.ai = _make_ai()

    def test_returns_text_when_stop_reason_is_end_turn(self):
        self.ai.client.messages.create.return_value = _text_response("Hello!")
        result = self.ai.generate_response(query="Hello")
        assert result == "Hello!"

    def test_returns_text_when_stop_reason_is_max_tokens(self):
        self.ai.client.messages.create.return_value = _text_response("Truncated...", stop_reason="max_tokens")
        result = self.ai.generate_response(query="long question")
        assert result == "Truncated..."

    def test_only_one_api_call_made_when_no_tool_use(self):
        self.ai.client.messages.create.return_value = _text_response("Direct answer")
        self.ai.generate_response(query="What is 2+2?")
        assert self.ai.client.messages.create.call_count == 1

    def test_api_called_with_correct_model(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="test")
        kwargs = self.ai.client.messages.create.call_args[1]
        assert kwargs["model"] == "claude-test"

    def test_api_called_with_user_query_in_messages(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="What is Python?")
        kwargs = self.ai.client.messages.create.call_args[1]
        messages = kwargs["messages"]
        assert any("Python" in str(m.get("content", "")) for m in messages)

    def test_system_prompt_included_in_api_call(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="test")
        kwargs = self.ai.client.messages.create.call_args[1]
        assert "system" in kwargs
        assert len(kwargs["system"]) > 0

    def test_conversation_history_appended_to_system_prompt(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="follow-up", conversation_history="User: hi\nAssistant: hello")
        kwargs = self.ai.client.messages.create.call_args[1]
        assert "hi" in kwargs["system"]

    def test_tools_added_to_api_call_when_provided(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        tools = [{"name": "search_course_content", "description": "search", "input_schema": {}}]
        self.ai.generate_response(query="test", tools=tools)
        kwargs = self.ai.client.messages.create.call_args[1]
        assert "tools" in kwargs
        assert kwargs["tools"] == tools

    def test_tools_not_in_api_call_when_none(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="test", tools=None)
        kwargs = self.ai.client.messages.create.call_args[1]
        assert "tools" not in kwargs

    def test_tool_choice_auto_set_when_tools_provided(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="test", tools=[{"name": "t"}])
        kwargs = self.ai.client.messages.create.call_args[1]
        assert kwargs.get("tool_choice") == {"type": "auto"}


# ---------------------------------------------------------------------------
# Tool use pathway
# ---------------------------------------------------------------------------

class TestToolExecution:

    def setup_method(self):
        self.ai = _make_ai()
        self.tool_manager = MagicMock()
        self.tool_manager.execute_tool.return_value = "Search results: found 3 lessons"

    def _run_with_tool(self, tool_name="search_course_content", tool_input=None, tool_id="toolu_001"):
        if tool_input is None:
            tool_input = {"query": "AI history"}
        first = _tool_use_response(tool_name, tool_input, tool_id)
        second = _text_response("Here is what I found.")
        self.ai.client.messages.create.side_effect = [first, second]
        return self.ai.generate_response(
            query="Tell me about AI history",
            tools=[{"name": tool_name}],
            tool_manager=self.tool_manager,
        )

    def test_two_api_calls_made_on_tool_use(self):
        self._run_with_tool()
        assert self.ai.client.messages.create.call_count == 2

    def test_returns_text_from_second_api_call(self):
        result = self._run_with_tool()
        assert result == "Here is what I found."

    def test_tool_manager_execute_called_with_correct_tool_name(self):
        self._run_with_tool(tool_name="search_course_content", tool_input={"query": "vectors"})
        self.tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="vectors"
        )

    def test_tool_manager_execute_called_with_all_tool_inputs(self):
        self._run_with_tool(
            tool_input={"query": "embeddings", "course_name": "RAG", "lesson_number": 2}
        )
        self.tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="embeddings",
            course_name="RAG",
            lesson_number=2,
        )

    def test_second_api_call_excludes_tools_parameter(self):
        """
        Critical: the follow-up call must NOT include 'tools' to avoid
        an infinite tool-call loop.
        """
        self._run_with_tool()
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_kwargs

    def test_second_api_call_preserves_system_prompt(self):
        self._run_with_tool()
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        assert "system" in second_kwargs
        assert len(second_kwargs["system"]) > 0

    def test_tool_result_message_has_role_user(self):
        self._run_with_tool(tool_id="toolu_xyz")
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        tool_result_msg = next(
            (m for m in messages if m["role"] == "user" and isinstance(m["content"], list)),
            None,
        )
        assert tool_result_msg is not None, "No user message with tool results found"

    def test_tool_result_message_has_correct_type(self):
        self._run_with_tool(tool_id="toolu_xyz")
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        tool_result_msg = next(
            m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        )
        assert tool_result_msg["content"][0]["type"] == "tool_result"

    def test_tool_result_message_links_to_correct_tool_use_id(self):
        self._run_with_tool(tool_id="toolu_xyz")
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        tool_result_msg = next(
            m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        )
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_xyz"

    def test_tool_result_message_contains_tool_output(self):
        self._run_with_tool()
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        tool_result_msg = next(
            m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        )
        assert tool_result_msg["content"][0]["content"] == "Search results: found 3 lessons"

    def test_assistant_tool_use_response_added_to_message_chain(self):
        """The assistant's tool_use response must be in the conversation before the tool result."""
        first = _tool_use_response("search_course_content", {"query": "test"})
        second = _text_response("Answer")
        self.ai.client.messages.create.side_effect = [first, second]
        self.ai.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=self.tool_manager,
        )
        second_kwargs = self.ai.client.messages.create.call_args_list[1][1]
        messages = second_kwargs["messages"]
        # There should be an assistant message with the tool_use block
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # The content should be the original tool_use content block
        assert assistant_msgs[0]["content"] == first.content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def setup_method(self):
        self.ai = _make_ai()

    def test_tool_use_with_no_tool_manager_falls_through_to_text(self):
        """
        When stop_reason is "tool_use" but tool_manager is None,
        the code falls through to `response.content[0].text`.
        A tool_use block has no .text attribute → AttributeError.
        This is a latent bug that should be flagged.
        """
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        # Simulate missing .text attribute on a tool_use block
        del tool_block.text

        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [tool_block]
        self.ai.client.messages.create.return_value = resp

        with pytest.raises(AttributeError):
            self.ai.generate_response(
                query="test",
                tools=[{"name": "search_course_content"}],
                tool_manager=None,  # intentionally no tool_manager
            )

    def test_empty_tools_list_does_not_add_tools_to_api_call(self):
        self.ai.client.messages.create.return_value = _text_response("ok")
        self.ai.generate_response(query="test", tools=[])
        kwargs = self.ai.client.messages.create.call_args[1]
        # Empty list is falsy; tools should not be added
        assert "tools" not in kwargs
