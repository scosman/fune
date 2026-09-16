"""Tests for v2_eval_helpers -- extract_value, extract_output_value, check_reference_key, references_trace, stringify_for_match."""

from kiln_ai.adapters.eval.eval_utils.v2_eval_helpers import (
    check_reference_key,
    extract_output_value,
    extract_value,
    references_trace,
    stringify_for_match,
)
from kiln_ai.datamodel.datamodel_enums import TaskOutputRatingType
from kiln_ai.datamodel.eval import EvalOutputScore, EvalTaskInput, SkippedReason


def _make_input(**kwargs) -> EvalTaskInput:
    defaults = {"final_message": "hello"}
    defaults.update(kwargs)
    return EvalTaskInput(**defaults)


# ---------------------------------------------------------------------------
# extract_value
# ---------------------------------------------------------------------------


class TestExtractValue:
    def test_default_returns_final_message(self):
        inp = _make_input(final_message="the answer")
        value, skip, detail = extract_value(None, inp)
        assert value == "the answer"
        assert skip is None
        assert detail is None

    def test_valid_expression(self):
        inp = _make_input(final_message="hi", task_input="my input")
        value, skip, _detail = extract_value("task_input", inp)
        assert value == "my input"
        assert skip is None

    def test_undefined_expression_skips(self):
        inp = _make_input()
        value, skip, detail = extract_value("nonexistent_field", inp)
        assert value is None
        assert skip == SkippedReason.extraction_failed
        assert detail is not None and "undefined" in detail

    def test_missing_trace_skips_with_missing_trace_reason(self):
        inp = _make_input(trace=None)
        value, skip, detail = extract_value("trace", inp)
        assert value is None
        assert skip == SkippedReason.missing_trace
        assert detail is not None and "trace" in detail

    def test_non_none_result_passes(self):
        inp = _make_input(trace=[{"role": "user", "content": "hi"}])
        value, skip, _detail = extract_value("trace", inp)
        assert value == [{"role": "user", "content": "hi"}]
        assert skip is None

    def test_trace_as_string_data_not_treated_as_trace_dependency(self):
        # 'trace' inside a string literal is data, not a variable reference —
        # a traceless run must evaluate the expression instead of skipping.
        inp = _make_input(final_message="the trace is fine", trace=None)
        value, skip, _detail = extract_value("'trace' in final_message", inp)
        assert value is True
        assert skip is None


# ---------------------------------------------------------------------------
# references_trace
# ---------------------------------------------------------------------------


class TestReferencesTrace:
    def test_bare_variable(self):
        assert references_trace("trace") is True

    def test_subscript(self):
        assert references_trace("trace[-1]") is True

    def test_filter(self):
        assert references_trace("trace | length") is True

    def test_conditional(self):
        assert references_trace("trace[0].content if trace else final_message") is True

    def test_quoted_string_is_not_a_reference(self):
        assert references_trace("'trace' in final_message") is False

    def test_dict_key_is_not_a_reference(self):
        assert references_trace("{'trace': 1}") is False

    def test_attribute_of_other_variable_is_not_a_reference(self):
        assert references_trace("final_message.trace") is False

    def test_subscript_key_of_other_variable_is_not_a_reference(self):
        assert references_trace("reference_data['trace']") is False

    def test_similar_identifier_is_not_a_reference(self):
        assert references_trace("retrace") is False

    def test_invalid_expression_is_not_a_reference(self):
        # Syntax errors are the extraction path's to report; no trace
        # dependency is claimed for them.
        assert references_trace("trace [") is False


# ---------------------------------------------------------------------------
# check_reference_key
# ---------------------------------------------------------------------------


class TestCheckReferenceKey:
    def test_key_present_with_value(self):
        inp = _make_input(reference_data={"expected": "gold"})
        value, skip, _detail = check_reference_key("expected", inp)
        assert value == "gold"
        assert skip is None

    def test_no_reference_data(self):
        inp = _make_input(reference_data=None)
        value, skip, _detail = check_reference_key("expected", inp)
        assert value is None
        assert skip == SkippedReason.missing_reference_key

    def test_key_missing(self):
        inp = _make_input(reference_data={"other": "value"})
        value, skip, _detail = check_reference_key("expected", inp)
        assert value is None
        assert skip == SkippedReason.missing_reference_key

    def test_key_value_is_none_skips(self):
        inp = _make_input(reference_data={"expected": None})
        value, skip, detail = check_reference_key("expected", inp)
        assert value is None
        assert skip == SkippedReason.missing_reference_key
        assert detail is not None and "None" in detail


# ---------------------------------------------------------------------------
# fromjson extraction error → clean skip
# ---------------------------------------------------------------------------


class TestFromjsonExtractionSkip:
    def test_extract_value_fromjson_invalid_json_skips(self):
        inp = _make_input(final_message="not valid json {")
        value, skip, detail = extract_value("(final_message | fromjson).field", inp)
        assert value is None
        assert skip == SkippedReason.extraction_failed
        assert detail is not None
        assert "not valid JSON" in detail
        assert "(final_message | fromjson).field" in detail

    def test_extract_value_fromjson_valid_json_passes(self):
        inp = _make_input(final_message='{"status": "ok"}')
        value, skip, _detail = extract_value("(final_message | fromjson).status", inp)
        assert value == "ok"
        assert skip is None


# ---------------------------------------------------------------------------
# extract_output_value — output failures become FAILs, not skips
# ---------------------------------------------------------------------------

_SAMPLE_SCORES = [
    EvalOutputScore(name="s1", instruction="i", type=TaskOutputRatingType.pass_fail),
    EvalOutputScore(name="s2", instruction="i", type=TaskOutputRatingType.pass_fail),
]


class TestExtractOutputValue:
    def test_success_returns_value_and_no_fail(self):
        inp = _make_input(final_message="ok")
        value, fail_result = extract_output_value(None, inp, _SAMPLE_SCORES)
        assert value == "ok"
        assert fail_result is None

    def test_expression_success(self):
        inp = _make_input(task_input="my input")
        value, fail_result = extract_output_value("task_input", inp, _SAMPLE_SCORES)
        assert value == "my input"
        assert fail_result is None

    def test_undefined_expression_fails_not_skips(self):
        inp = _make_input()
        value, fail_result = extract_output_value(
            "nonexistent_field", inp, _SAMPLE_SCORES
        )
        assert value is None
        assert fail_result is not None
        assert fail_result.skipped_reason is None
        assert fail_result.scores == {"s1": 0.0, "s2": 0.0}

    def test_missing_trace_skips_not_fails(self):
        inp = _make_input(trace=None)
        value, fail_result = extract_output_value("trace", inp, _SAMPLE_SCORES)
        assert value is None
        assert fail_result is not None
        # Missing trace propagates as a skip, not a scored FAIL.
        assert fail_result.skipped_reason == SkippedReason.missing_trace
        assert fail_result.scores == {}

    def test_fromjson_invalid_json_fails_not_skips(self):
        inp = _make_input(final_message="not json")
        value, fail_result = extract_output_value(
            "(final_message | fromjson).field", inp, _SAMPLE_SCORES
        )
        assert value is None
        assert fail_result is not None
        assert fail_result.skipped_reason is None
        assert fail_result.scores == {"s1": 0.0, "s2": 0.0}

    def test_fromjson_valid_json_succeeds(self):
        inp = _make_input(final_message='{"key": "val"}')
        value, fail_result = extract_output_value(
            "(final_message | fromjson).key", inp, _SAMPLE_SCORES
        )
        assert value == "val"
        assert fail_result is None


# ---------------------------------------------------------------------------
# stringify_for_match
# ---------------------------------------------------------------------------


class TestStringifyForMatch:
    def test_string_passthrough(self):
        assert stringify_for_match("hello") == "hello"

    def test_dict_to_json(self):
        result = stringify_for_match({"a": 1, "b": 2})
        assert result == '{"a": 1, "b": 2}'
        assert '"' in result

    def test_list_to_json(self):
        result = stringify_for_match([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_bool_true(self):
        assert stringify_for_match(True) == "true"

    def test_bool_false(self):
        assert stringify_for_match(False) == "false"

    def test_number(self):
        assert stringify_for_match(42) == "42"

    def test_float(self):
        assert stringify_for_match(3.14) == "3.14"

    def test_none(self):
        assert stringify_for_match(None) == "null"

    def test_non_serializable_falls_back_to_str(self):
        assert stringify_for_match(object.__class__) == str(object.__class__)
