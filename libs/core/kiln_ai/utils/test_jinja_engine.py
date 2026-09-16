import pytest
from jinja2 import Undefined, UndefinedError
from jinja2.sandbox import SecurityError

from kiln_ai.datamodel.input_transform import JinjaInputTransform
from kiln_ai.utils.jinja_engine import (
    JinjaExtractionError,
    compile_expression_or_raise,
    compile_template_or_raise,
    extract,
    render_input_transform,
)


class TestCompileTemplateOrRaise:
    def test_valid_template(self):
        compile_template_or_raise("Hello {{ name }}")

    def test_valid_template_with_blocks(self):
        compile_template_or_raise("{% if x %}yes{% endif %}")

    def test_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="line"):
            compile_template_or_raise("{% if ")

    def test_unclosed_block_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid Jinja2 template"):
            compile_template_or_raise("{% for x in items %}")


class TestCompileExpressionOrRaise:
    def test_valid_expression(self):
        compile_expression_or_raise("foo")

    def test_valid_expression_with_filter(self):
        compile_expression_or_raise("items | length")

    def test_syntax_error_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid Jinja2 expression"):
            compile_expression_or_raise("foo ||| bar")


class TestRenderInputTransform:
    def test_dict_input_attribute_access(self):
        transform = JinjaInputTransform(template="{{ input.foo }}")
        result = render_input_transform(transform, {"foo": 1})
        assert result == "1"

    def test_dict_input_multiple_fields(self):
        transform = JinjaInputTransform(template="{{ input.a }} and {{ input.b }}")
        result = render_input_transform(transform, {"a": "hello", "b": "world"})
        assert result == "hello and world"

    def test_list_input_index_access(self):
        transform = JinjaInputTransform(template="{{ input[0] }}")
        result = render_input_transform(transform, [1, 2, 3])
        assert result == "1"

    def test_list_input_length_filter(self):
        transform = JinjaInputTransform(template="{{ input | length }}")
        result = render_input_transform(transform, [1, 2, 3])
        assert result == "3"

    def test_string_input(self):
        transform = JinjaInputTransform(template="{{ input }}")
        result = render_input_transform(transform, "hello")
        assert result == "hello"

    def test_plaintext_json_dict_auto_parsed(self):
        transform = JinjaInputTransform(template="{{ input.foo }}")
        result = render_input_transform(transform, '{"foo": 1}')
        assert result == "1"

    def test_plaintext_json_list_auto_parsed(self):
        transform = JinjaInputTransform(template="{{ input[0] }}")
        result = render_input_transform(transform, "[10, 20]")
        assert result == "10"

    def test_plaintext_json_scalar_auto_parsed(self):
        transform = JinjaInputTransform(template="{{ input }}")
        result = render_input_transform(transform, "42")
        assert result == "42"

    def test_plaintext_non_json_fallback(self):
        transform = JinjaInputTransform(template="{{ input }}")
        result = render_input_transform(transform, "not json")
        assert result == "not json"

    def test_undefined_error_on_missing_attribute(self):
        transform = JinjaInputTransform(template="{{ input.missing }}")
        with pytest.raises(UndefinedError):
            render_input_transform(transform, {"foo": 1})

    def test_sandbox_violation_dunder_class(self):
        transform = JinjaInputTransform(template="{{ input.__class__ }}")
        with pytest.raises(SecurityError):
            render_input_transform(transform, {})

    def test_sandbox_violation_dunder_bases(self):
        transform = JinjaInputTransform(template="{{ input.__class__.__bases__ }}")
        with pytest.raises(SecurityError):
            render_input_transform(transform, {})

    def test_sandbox_violation_mro_walking(self):
        transform = JinjaInputTransform(
            template="{{ input.__class__.__mro__[1].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            render_input_transform(transform, {})

    def test_sandbox_violation_subclasses_traversal(self):
        transform = JinjaInputTransform(
            template="{{ ''.__class__.__mro__[2].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            render_input_transform(transform, "")

    def test_empty_template_returns_empty_string(self):
        transform = JinjaInputTransform(template="")
        result = render_input_transform(transform, {"foo": 1})
        assert result == ""

    def test_unknown_transform_variant_raises(self):
        class FakeTransform:
            type = "fake"

        with pytest.raises(ValueError, match="Unknown InputTransform variant"):
            render_input_transform(FakeTransform(), "input")  # type: ignore


class TestExtract:
    def test_returns_value(self):
        assert extract("foo", {"foo": 42}) == 42

    def test_returns_string(self):
        assert extract("name", {"name": "alice"}) == "alice"

    def test_returns_list(self):
        assert extract("items", {"items": [1, 2, 3]}) == [1, 2, 3]

    def test_returns_dict(self):
        assert extract("nested", {"nested": {"a": 1}}) == {"a": 1}

    def test_missing_key_returns_undefined(self):
        result = extract("foo", {})
        assert isinstance(result, Undefined)

    def test_explicit_none_returns_none(self):
        result = extract("foo", {"foo": None})
        assert result is None

    def test_materializes_generators(self):
        result = extract(
            "data | map(attribute='x') | list",
            {"data": [{"x": 1}, {"x": 2}]},
        )
        assert result == [1, 2]

    def test_materializes_generator_without_list_filter(self):
        result = extract(
            "data | map(attribute='x')",
            {"data": [{"x": 1}, {"x": 2}]},
        )
        assert isinstance(result, list)
        assert result == [1, 2]

    def test_nested_attribute_access(self):
        result = extract("a.b", {"a": {"b": "deep"}})
        assert result == "deep"

    def test_malformed_expression_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid Jinja2 expression"):
            extract("{{ invalid", {})


class TestTrimAndLstripBlocks:
    def test_trim_blocks_strips_newline_after_block_tag(self):
        template = "{% if True %}\nyes\n{% endif %}"
        transform = JinjaInputTransform(template=template)
        result = render_input_transform(transform, {})
        assert result == "yes\n"

    def test_lstrip_blocks_strips_leading_whitespace(self):
        template = "  {% if True %}\nyes\n  {% endif %}"
        transform = JinjaInputTransform(template=template)
        result = render_input_transform(transform, {})
        assert result == "yes\n"

    def test_combined_trim_and_lstrip(self):
        template = (
            "start\n  {% for item in input %}\n  - {{ item }}\n  {% endfor %}\nend"
        )
        transform = JinjaInputTransform(template=template)
        result = render_input_transform(transform, ["a", "b"])
        assert result == "start\n  - a\n  - b\nend"


class TestFromjsonFilter:
    def test_parse_dict(self):
        result = extract(
            "final_message | fromjson",
            {"final_message": '{"key": "value"}'},
        )
        assert result == {"key": "value"}

    def test_parse_list(self):
        result = extract(
            "final_message | fromjson",
            {"final_message": "[1, 2, 3]"},
        )
        assert result == [1, 2, 3]

    def test_parse_scalar(self):
        result = extract(
            "final_message | fromjson",
            {"final_message": "42"},
        )
        assert result == 42

    def test_nested_field_access(self):
        result = extract(
            "(final_message | fromjson).user.status",
            {"final_message": '{"user": {"status": "active"}}'},
        )
        assert result == "active"

    def test_array_length(self):
        result = extract(
            '(final_message | fromjson)["items"] | length',
            {"final_message": '{"items": [1, 2, 3]}'},
        )
        assert result == 3

    def test_invalid_json_raises(self):
        with pytest.raises(JinjaExtractionError, match="not valid JSON"):
            extract("final_message | fromjson", {"final_message": "not json {"})

    def test_non_string_input_raises(self):
        with pytest.raises(JinjaExtractionError, match="expected a string"):
            extract("data | fromjson", {"data": 42})

    def test_fromjson_in_template_env(self):
        transform = JinjaInputTransform(
            template="{{ (input.json_field | fromjson).name }}"
        )
        result = render_input_transform(transform, {"json_field": '{"name": "Alice"}'})
        assert result == "Alice"

    def test_fromjson_invalid_json_in_template_raises(self):
        transform = JinjaInputTransform(
            template="{{ (input.json_field | fromjson).name }}"
        )
        with pytest.raises(JinjaExtractionError, match="not valid JSON"):
            render_input_transform(transform, {"json_field": "not json"})


class TestCanonicalExpressions:
    """Ensure the canonical UI example expressions all work correctly."""

    def test_extract_field_from_json(self):
        result = extract(
            "(final_message | fromjson).user.status",
            {"final_message": '{"user": {"status": "active"}}'},
        )
        assert result == "active"

    def test_truncate_long_output(self):
        long_text = "x" * 500
        result = extract(
            "final_message | truncate(200)",
            {"final_message": long_text},
        )
        assert len(result) <= 200
        assert result.endswith("...")

    def test_last_message_in_trace(self):
        trace = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = extract("trace[-1].content", {"trace": trace})
        assert result == "world"

    def test_uppercase_the_output(self):
        result = extract("final_message | upper", {"final_message": "hello"})
        assert result == "HELLO"

    def test_count_messages_in_trace(self):
        trace = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "thanks"},
        ]
        result = extract("trace | length", {"trace": trace})
        assert result == 3

    def test_tool_call_name_in_trace(self):
        trace = [
            {"role": "user", "content": "add 1 and 2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        "type": "function",
                    }
                ],
            },
        ]
        result = extract("trace[-1].tool_calls[0].function.name", {"trace": trace})
        assert result == "add"

    def test_json_array_length(self):
        result = extract(
            '(final_message | fromjson)["items"] | length',
            {"final_message": '{"items": ["a", "b", "c", "d"]}'},
        )
        assert result == 4


class TestFormatTraceFilter:
    """{{ trace | format_trace }} renders the canonical role-labelled
    transcript (EvalTraceFormatter's format) inside templates, so every LLM
    that sees a conversation sees the same rendering."""

    def test_renders_role_labelled_turns(self):
        trace = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = render_input_transform(
            JinjaInputTransform(template="{{ input.trace | format_trace }}"),
            {"trace": trace},
        )
        assert result == (
            "user:\n<user_message>\nhello\n</user_message>\n\n"
            "assistant:\n<assistant_message>\nworld\n</assistant_message>"
        )

    def test_matches_eval_trace_formatter(self):
        # The filter IS the formatter — a drift between the two would split
        # the canonical rendering back into per-consumer variants.
        from kiln_ai.adapters.eval.eval_utils.eval_trace_formatter import (
            EvalTraceFormatter,
        )

        trace = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
        rendered = render_input_transform(
            JinjaInputTransform(template="{{ input.trace | format_trace }}"),
            {"trace": trace},
        )
        assert rendered == EvalTraceFormatter.trace_to_formatted_conversation_history(
            trace
        )

    def test_non_list_raises_extraction_error(self):
        with pytest.raises(JinjaExtractionError, match="got str"):
            render_input_transform(
                JinjaInputTransform(template="{{ input.trace | format_trace }}"),
                {"trace": "not a list"},
            )

    def test_non_dict_member_raises_error_naming_the_member(self):
        # The error must name the malformed member, not the (valid) container.
        with pytest.raises(JinjaExtractionError, match="message 1 is str, not a dict"):
            render_input_transform(
                JinjaInputTransform(template="{{ input.trace | format_trace }}"),
                {"trace": [{"role": "user", "content": "hi"}, "not a dict"]},
            )

    def test_canonical_rendering_is_frozen(self):
        # CROSS-REPO DRIFT ALARM. The TAG VOCABULARY below (system/user/
        # assistant messages, assistant reasoning, requested tool calls, tool
        # results) is what the kiln_server claim-builder task's instruction
        # documents to its LLM (buildClaimEvidence "Multi-turn transcripts"
        # section; its tests pin the same tag list). Changing a tag here
        # without updating that instruction silently degrades citations, so a
        # tag change must land in BOTH repos together.
        #
        # Role labels are not part of that contract: the instruction describes
        # them generically ("the role: labels and <…> tags are added by the
        # rendering software — never cite them") and enumerates no specific
        # one. The labels below therefore changed on their own when tool
        # results gained their tool's name, and the tags did not.
        trace = [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Is the X200 in stock?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "check_stock",
                            "arguments": '{"sku": "X200"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "In stock: 4"},
            {
                "role": "assistant",
                "content": "Yes — 4 units in stock.",
                "reasoning_content": "Inventory says 4.",
            },
        ]
        rendered = render_input_transform(
            JinjaInputTransform(template="{{ input.trace | format_trace }}"),
            {"trace": trace},
        )
        assert rendered == (
            "system:\n<system_message>\nYou are a support agent.\n</system_message>\n\n"
            "user:\n<user_message>\nIs the X200 in stock?\n</user_message>\n\n"
            "assistant requested tool calls:\n<assistant_requested_tool_calls>\n"
            '- Tool Name: check_stock\n- Arguments: {"sku": "X200"}\n'
            "</assistant_requested_tool_calls>\n\n"
            "tool result from check_stock:\n<tool_tool_message>\n"
            "In stock: 4\n</tool_tool_message>\n\n"
            "assistant reasoning:\n<assistant_reasoning_message>\n"
            "Inventory says 4.\n</assistant_reasoning_message>\n\n"
            "assistant:\n<assistant_message>\nYes — 4 units in stock.\n</assistant_message>"
        )
