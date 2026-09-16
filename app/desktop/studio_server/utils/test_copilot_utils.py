"""Tests for app/desktop/studio_server/utils/copilot_utils.py."""

import logging
import random
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from kiln_ai.datamodel import GradedClaim, Project, Task, TaskRun
from kiln_ai.datamodel.datamodel_enums import (
    FeedbackSource,
    TaskOutputRatingType,
    TurnMode,
)
from kiln_ai.datamodel.eval import MultiTurnDriveConfig
from kiln_ai.datamodel.external_tool_server import ExternalToolServer, ToolServerType
from kiln_ai.datamodel.run_config import (
    McpRunConfigProperties,
    MCPToolReference,
    ToolsRunConfig,
)
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_output import (
    DataSource,
    DataSourceType,
    TaskOutput,
    TaskOutputRating,
)
from kiln_ai.run_context import (
    clear_agent_run_id,
    get_agent_run_id,
    set_agent_run_id,
)
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from app.desktop.studio_server.api_models.copilot_models import (
    ClaimReviewApi,
    DrivenSyntheticCaseApi,
    ReviewedChainApi,
    ReviewedExample,
    SampleApi,
    TaskInfoApi,
    TaskSkillInfoApi,
    TaskToolInfoApi,
)
from app.desktop.studio_server.utils.copilot_utils import (
    GOLDEN_TARGET_FRACTION,
    KILN_ADAPTER_NAME,
    KILN_COPILOT_MODEL_NAME,
    KILN_COPILOT_MODEL_PROVIDER,
    build_multi_turn_eval_inputs,
    build_single_turn_batch_eval_inputs,
    build_single_turn_eval_inputs,
    create_single_turn_dataset,
    create_task_run_from_reviewed,
    create_task_run_from_sample,
    deal_pool_train_val,
    delete_multi_turn_batch_chains,
    delete_single_turn_batch_runs,
    find_single_turn_batch_runs,
    get_copilot_api_key,
    persist_eval_slice,
    rate_reviewed_batch_runs,
    select_golden_runs,
    split_and_tag_batch_runs,
    split_pool_train_eval,
    tag_single_turn_drive_run,
    task_capabilities_for_task,
    task_info_payload,
    unrate_reviewed_batch_runs,
    warn_if_golden_below_target,
)


class TestGetCopilotApiKey:
    def test_returns_api_key_when_configured(self):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config:
            mock_config.return_value.kiln_copilot_api_key = "test_api_key_123"
            result = get_copilot_api_key()
            assert result == "test_api_key_123"

    def test_raises_401_when_not_configured(self):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config:
            mock_config.return_value.kiln_copilot_api_key = None
            with pytest.raises(HTTPException) as exc_info:
                get_copilot_api_key()
            assert exc_info.value.status_code == 401
            assert "API key not configured" in exc_info.value.detail

    def test_raises_401_when_empty_string(self):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config:
            mock_config.return_value.kiln_copilot_api_key = ""
            with pytest.raises(HTTPException) as exc_info:
                get_copilot_api_key()
            assert exc_info.value.status_code == 401


class TestSplitPoolTrainEval:
    @pytest.mark.parametrize(
        "n,expected_train,expected_eval",
        [
            (0, 0, 0),  # empty pool
            (1, 1, 0),  # 1//3 == 0 → all to train
            (2, 2, 0),  # 2//3 == 0 → all to train
            (3, 2, 1),  # exact 2:1
            (4, 3, 1),
            (6, 4, 2),
            (9, 6, 3),  # exact 2:1
        ],
    )
    def test_splits_two_to_one(self, n, expected_train, expected_eval):
        pool = list(range(n))
        train, eval_items = split_pool_train_eval(pool, random.Random(0))
        assert len(train) == expected_train
        assert len(eval_items) == expected_eval
        # Partition: disjoint and complete.
        assert sorted(train + eval_items) == pool

    def test_does_not_mutate_input(self):
        pool = list(range(9))
        original = list(pool)
        split_pool_train_eval(pool, random.Random(1))
        assert pool == original

    def test_deterministic_under_seed(self):
        pool = list(range(9))
        a = split_pool_train_eval(pool, random.Random(42))
        b = split_pool_train_eval(pool, random.Random(42))
        assert a == b


class TestSelectGoldenLeaves:
    """select_golden_runs carves golden (rated-only, capped at 25%) off the
    leaves; the remainder feeds the train/eval split."""

    def test_all_rated_golden_capped_at_quarter(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 8)
        rated = {leaf.id for leaf in leaves}
        golden, remaining = select_golden_runs(leaves, rated, random.Random(0))
        assert len(golden) == 2  # 8 // 4
        assert len(remaining) == 6
        # Golden is drawn from rated; disjoint from remaining; covers all.
        assert all(leaf.id in rated for leaf in golden)
        assert {leaf.id for leaf in golden}.isdisjoint({leaf.id for leaf in remaining})
        assert len(golden) + len(remaining) == 8

    def test_rated_below_cap_golden_is_all_rated(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 8)
        rated = {leaves[0].id}  # 1 rated, cap is 2
        golden, remaining = select_golden_runs(leaves, rated, random.Random(0))
        assert {leaf.id for leaf in golden} == {leaves[0].id}
        assert len(remaining) == 7

    def test_unrated_never_enters_golden(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 8)
        golden, remaining = select_golden_runs(leaves, set(), random.Random(0))
        assert golden == []
        assert len(remaining) == 8

    def test_excess_rated_falls_into_remaining(self, multiturn_task):
        # All 8 rated, cap 2 → 6 rated leaves land in remaining (still held out).
        leaves = _make_su_leaves(multiturn_task, 8)
        rated = {leaf.id for leaf in leaves}
        golden, remaining = select_golden_runs(leaves, rated, random.Random(1))
        assert len(golden) == 2
        assert all(leaf.id in rated for leaf in remaining)


class TestWarnIfGoldenBelowTarget:
    def test_warns_when_below_target(self, caplog):
        # 1 of 10 rated == 10%, below the 25% floor.
        with caplog.at_level("WARNING"):
            warn_if_golden_below_target(1, 10)
        assert any("below the" in r.message for r in caplog.records)

    def test_silent_at_or_above_target(self, caplog):
        # 25% is the target — not below it.
        target_count = int(GOLDEN_TARGET_FRACTION * 100)
        with caplog.at_level("WARNING"):
            warn_if_golden_below_target(target_count, 100)
        assert caplog.records == []

    def test_silent_when_total_zero(self, caplog):
        with caplog.at_level("WARNING"):
            warn_if_golden_below_target(0, 0)
        assert caplog.records == []


class TestCreateTaskRunFromSample:
    def test_creates_task_run_with_correct_input(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(sample, "some_tag")
        assert task_run.input == "test input"

    def test_creates_task_run_with_correct_output(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(sample, "some_tag")
        assert task_run.output.output == "test output"

    def test_creates_task_run_with_tag(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(sample, "some_tag")
        assert "some_tag" in task_run.tags

    def test_creates_task_run_with_extra_tags(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(
            sample, "some_tag", extra_tags=["session_123", "other_tag"]
        )
        assert "some_tag" in task_run.tags
        assert "session_123" in task_run.tags
        assert "other_tag" in task_run.tags

    def test_creates_task_run_without_extra_tags(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(sample, "some_tag", extra_tags=None)
        assert task_run.tags == ["some_tag"]

    def test_creates_task_run_with_synthetic_data_source(self):
        sample = SampleApi(input="test input", output="test output")
        task_run = create_task_run_from_sample(sample, "some_tag")
        assert task_run.input_source.type == DataSourceType.synthetic
        assert task_run.input_source.properties["model_name"] == KILN_COPILOT_MODEL_NAME
        assert (
            task_run.input_source.properties["model_provider"]
            == KILN_COPILOT_MODEL_PROVIDER
        )
        assert task_run.input_source.properties["adapter_name"] == KILN_ADAPTER_NAME


class TestCreateTaskRunFromReviewed:
    def test_creates_task_run_with_correct_input(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="Good example",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        assert task_run.input == "test input"

    def test_creates_task_run_with_correct_output(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        assert task_run.output.output == "test output"

    def test_creates_task_run_with_pass_rating_when_meets_spec(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        rating_key = "named::My Spec"
        assert rating_key in task_run.output.rating.requirement_ratings
        assert task_run.output.rating.requirement_ratings[rating_key].value == 1.0

    def test_creates_task_run_with_fail_rating_when_not_meets_spec(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=False,
            user_says_meets_spec=False,
            feedback="Bad example",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        rating_key = "named::My Spec"
        assert rating_key in task_run.output.rating.requirement_ratings
        assert task_run.output.rating.requirement_ratings[rating_key].value == 0.0

    def test_creates_task_run_with_tag(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        assert "golden_tag" in task_run.tags

    def test_creates_task_run_with_extra_tags(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        task_run, _ = create_task_run_from_reviewed(
            example, "golden_tag", "My Spec", extra_tags=["session_456"]
        )
        assert "golden_tag" in task_run.tags
        assert "session_456" in task_run.tags

    def test_creates_task_run_with_pass_fail_rating_type(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        task_run, _ = create_task_run_from_reviewed(example, "golden_tag", "My Spec")
        rating_key = "named::My Spec"
        assert (
            task_run.output.rating.requirement_ratings[rating_key].type
            == TaskOutputRatingType.pass_fail
        )

    def test_returns_feedback_text_when_present(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=False,
            feedback="This fails because the output is too vague",
        )
        _, feedback_text = create_task_run_from_reviewed(
            example, "golden_tag", "My Spec"
        )
        assert feedback_text == "This fails because the output is too vague"

    def test_returns_none_feedback_when_empty(self):
        example = ReviewedExample(
            input="test input",
            output="test output",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        _, feedback_text = create_task_run_from_reviewed(
            example, "golden_tag", "My Spec"
        )
        assert feedback_text is None


def _samples(n: int) -> list[SampleApi]:
    return [SampleApi(input=f"input_{i}", output=f"output_{i}") for i in range(n)]


def _reviewed(n: int) -> list[ReviewedExample]:
    return [
        ReviewedExample(
            input=f"reviewed_input_{i}",
            output=f"reviewed_output_{i}",
            model_says_meets_spec=True,
            user_says_meets_spec=True,
            feedback="",
        )
        for i in range(n)
    ]


def _make_dataset(
    all_examples: list[SampleApi],
    reviewed_examples: list[ReviewedExample],
    seed: int = 0,
):
    return create_single_turn_dataset(
        all_examples,
        reviewed_examples,
        "eval_tag",
        "train_tag",
        "golden_tag",
        "Test Spec",
        rng=random.Random(seed),
    )


def _by_split(task_runs):
    """Bucket runs by their single split tag. The eval slice is never here:
    it is EvalInput items, not runs."""
    return {
        "train": [tr for tr in task_runs if "train_tag" in tr.tags],
        "golden": [tr for tr in task_runs if "golden_tag" in tr.tags],
    }


class TestCreateSingleTurnDataset:
    def test_run_count_is_train_plus_rated(self):
        # Golden holds the rated examples; the unrated pool splits into the
        # train runs and the eval slice (EvalInputs, not runs).
        dataset = _make_dataset(_samples(300), _reviewed(4))
        assert len(dataset.task_runs) == 204
        assert len(dataset.eval_inputs) == 100

    def test_reviewed_examples_are_golden_and_rated(self):
        dataset = _make_dataset(_samples(60), _reviewed(1))
        golden = _by_split(dataset.task_runs)["golden"]
        # Golden == exactly the rated set, no unrated padding.
        assert len(golden) == 1
        assert golden[0].input == "reviewed_input_0"
        assert golden[0].output.rating is not None

    def test_golden_is_rated_only_no_unrated_padding(self):
        # Golden holds exactly the rated count — never topped up with unrated
        # machine examples, even when that leaves it small.
        dataset = _make_dataset(_samples(60), _reviewed(2))
        golden = _by_split(dataset.task_runs)["golden"]
        assert len(golden) == 2
        assert all(tr.output.rating is not None for tr in golden)

    def test_zero_rated_yields_no_golden(self):
        dataset = _make_dataset(_samples(30), _reviewed(0))
        assert _by_split(dataset.task_runs)["golden"] == []

    def test_splits_are_disjoint_and_complete(self):
        dataset = _make_dataset(_samples(60), _reviewed(4))
        for tr in dataset.task_runs:
            split_tags = {"train_tag", "golden_tag"} & set(tr.tags)
            assert len(split_tags) == 1, f"run {tr.input} has splits {split_tags}"
            # The eval slice never rides on a run — a stray eval tag here would
            # put the same example in two splits, one of them frozen output.
            assert "eval_tag" not in tr.tags
        buckets = _by_split(dataset.task_runs)
        assert len(buckets["train"]) + len(buckets["golden"]) == len(dataset.task_runs)
        # Every example lands in exactly one slice across both stores.
        assert len(dataset.task_runs) + len(dataset.eval_inputs) == 64

    def test_unrated_pool_splits_two_to_one(self):
        dataset = _make_dataset(_samples(60), _reviewed(0))
        assert len(dataset.eval_inputs) == 20  # 60 // 3
        assert len(_by_split(dataset.task_runs)["train"]) == 40

    def test_one_rated_small_pool(self):
        # golden=1 (rated); the 3 unrated split 2:1 → train 2, eval 1.
        dataset = _make_dataset(_samples(3), _reviewed(1))
        buckets = _by_split(dataset.task_runs)
        assert len(buckets["golden"]) == 1
        assert len(buckets["train"]) == 2
        assert len(dataset.eval_inputs) == 1

    def test_tiny_pool_all_train_no_eval(self):
        # 2 unrated → 2//3 == 0 eval, both to train (documented small-N edge).
        dataset = _make_dataset(_samples(2), _reviewed(0))
        assert len(_by_split(dataset.task_runs)["train"]) == 2
        assert dataset.eval_inputs == []

    def test_handles_insufficient_examples(self):
        dataset = _make_dataset(_samples(5), _reviewed(0))
        assert len(dataset.task_runs) == 4
        assert len(dataset.eval_inputs) == 1

    def test_does_not_mutate_all_examples(self):
        all_examples = _samples(30)
        original = list(all_examples)
        _make_dataset(all_examples, _reviewed(2))
        assert all_examples == original

    def test_every_item_shares_one_session_tag(self):
        # Runs and eval inputs alike carry the generation session's tag, so a
        # saved spec's whole dataset stays traceable to one batch.
        dataset = _make_dataset(_samples(60), _reviewed(2))
        tagged = [*dataset.task_runs, *dataset.eval_inputs]
        session_tags = {
            tag
            for item in tagged
            for tag in item.tags
            if tag.startswith("synthetic_session_")
        }
        assert len(session_tags) == 1
        for item in tagged:
            assert sum(t.startswith("synthetic_session_") for t in item.tags) == 1

    def test_warns_when_golden_below_target(self, caplog):
        with caplog.at_level("WARNING"):
            _make_dataset(_samples(99), _reviewed(1))  # golden 1% << 25%
        assert any("below the" in r.message for r in caplog.records)


class TestBuildSingleTurnEvalInputs:
    def test_carries_the_input_verbatim(self):
        # Inputs only, by construction: the builder takes input strings and
        # the runner produces a fresh output per run config at eval time.
        eval_inputs = build_single_turn_eval_inputs(
            ["input_0", "input_1"], "eval_myspec", ["synthetic_session_7"]
        )
        assert len(eval_inputs) == 2
        assert [ei.data.type for ei in eval_inputs] == ["single_turn", "single_turn"]
        assert [ei.data.user_message.text for ei in eval_inputs] == [
            "input_0",
            "input_1",
        ]
        assert all(ei.reference is None for ei in eval_inputs)

    def test_tags_carry_the_eval_slice_and_session(self):
        eval_inputs = build_single_turn_eval_inputs(
            ["input_0"], "eval_myspec", ["synthetic_session_7"]
        )
        assert eval_inputs[0].tags == ["eval_myspec", "synthetic_session_7"]

    def test_builds_unsaved_models(self):
        # Built and validated here, persisted inside the save unit of work.
        eval_inputs = build_single_turn_eval_inputs(
            ["input_0", "input_1", "input_2"], "eval_myspec", []
        )
        assert all(ei.path is None for ei in eval_inputs)


class TestBuildSingleTurnBatchEvalInputs:
    """The wizard save's eval-slice builder: same inputs-only items, with
    drive-batch provenance and the task parented on."""

    def test_carries_batch_provenance_and_parents_the_task(self, tmp_path):
        project_path = tmp_path / "p" / "project.kiln"
        project_path.parent.mkdir()
        project = Project(name="P", path=project_path)
        project.save_to_file()
        task = Task(name="T", instruction="i", parent=project)
        task.save_to_file()

        eval_inputs = build_single_turn_batch_eval_inputs(
            ["input_0", "input_1"], "batch123", task, "eval_myspec"
        )
        assert len(eval_inputs) == 2
        assert [ei.data.user_message.text for ei in eval_inputs] == [
            "input_0",
            "input_1",
        ]
        assert all(
            ei.tags == ["eval_myspec", "single_turn_drive_batch:batch123"]
            for ei in eval_inputs
        )
        assert all(ei.parent is task for ei in eval_inputs)
        # Built unsaved — persistence happens in the save unit of work.
        assert all(ei.path is None for ei in eval_inputs)


def _claim_review_api(judge_score: str = "fail") -> ClaimReviewApi:
    return ClaimReviewApi(
        judge_score=judge_score,
        judge_reasoning="Stated an unverified policy as fact.",
        overview="The user asked about returns and the agent quoted a window.",
        claims=[
            GradedClaim(
                text="The agent stated a specific return window as fact [1].",
                human_grade="disagree",
                human_feedback="The policy quoted is actually correct.",
            ),
            GradedClaim(
                text="It fails because the window was never verified [1].",
                human_grade="agree",
                human_feedback=None,
            ),
        ],
        human_verdict=judge_score,
    )


@pytest.fixture
def task_with_leaves(tmp_path):
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    task = Task(name="Test Task", instruction="Test instruction", parent=project)
    task.save_to_file()
    source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "haiku",
            "model_provider": "openrouter",
            "adapter_name": "kiln_synthetic_user_runner",
        },
    )
    leaves = []
    for i in range(2):
        run = TaskRun(
            parent=task,
            input=f"input {i}",
            input_source=source,
            output=TaskOutput(output=f"output {i}", source=source),
        )
        run.save_to_file()
        leaves.append(run)
    return task, leaves


class TestRateMultiTurnChainLeaves:
    def test_writes_rating_feedback_and_claim_review(self, task_with_leaves):
        _, leaves = task_with_leaves
        reviewed = [
            ReviewedChainApi(
                leaf_run_id=leaves[0].id,
                user_says_meets_spec=False,
                feedback="Fabricated the return window.",
                claim_review=_claim_review_api(),
            ),
            ReviewedChainApi(
                leaf_run_id=leaves[1].id,
                user_says_meets_spec=True,
            ),
        ]

        rated_out: list = []
        rate_reviewed_batch_runs(
            leaves, reviewed, spec_name="My Spec", rated_out=rated_out
        )

        # Leaf 0: FAIL rating + feedback + claim review persisted.
        rating = leaves[0].output.rating
        assert rating is not None
        req = rating.requirement_ratings["named::My Spec"]
        assert req.type == TaskOutputRatingType.pass_fail
        assert req.value == 0.0
        feedback = leaves[0].feedback()
        assert len(feedback) == 1
        assert feedback[0].source == FeedbackSource.spec_feedback
        assert feedback[0].feedback == "Fabricated the return window."
        reviews = leaves[0].claim_reviews()
        assert len(reviews) == 1
        assert reviews[0].judge_score == "fail"
        assert reviews[0].overview.startswith("The user asked")
        assert reviews[0].claims[0].human_grade == "disagree"
        assert (
            reviews[0].claims[0].human_feedback
            == "The policy quoted is actually correct."
        )
        assert reviews[0].human_verdict == "fail"

        # Leaf 1: PASS rating, no feedback/claim-review children.
        rating = leaves[1].output.rating
        assert rating is not None
        assert rating.requirement_ratings["named::My Spec"].value == 1.0
        assert leaves[1].feedback() == []
        assert leaves[1].claim_reviews() == []

        # Both mutations were captured for rollback.
        assert len(rated_out) == 2

    def test_unknown_leaf_id_raises_404(self, task_with_leaves):
        _, leaves = task_with_leaves
        reviewed = [
            ReviewedChainApi(leaf_run_id="no_such_run", user_says_meets_spec=True)
        ]
        with pytest.raises(HTTPException) as exc:
            rate_reviewed_batch_runs(leaves, reviewed, spec_name="My Spec")
        assert exc.value.status_code == 404

    def test_unrate_restores_prior_state(self, task_with_leaves):
        _, leaves = task_with_leaves
        # Leaf 0 starts with a pre-existing rating that must survive rollback.
        prior = TaskOutputRating(
            type=TaskOutputRatingType.five_star,
            value=None,
            requirement_ratings={
                "named::Other Spec": {
                    "type": TaskOutputRatingType.pass_fail,
                    "value": 1.0,
                }
            },
        )
        leaves[0].output.rating = prior
        leaves[0].save_to_file()

        reviewed = [
            ReviewedChainApi(
                leaf_run_id=leaves[0].id,
                user_says_meets_spec=False,
                feedback="why",
                claim_review=_claim_review_api(),
            ),
        ]
        rated_out: list = []
        rate_reviewed_batch_runs(
            leaves, reviewed, spec_name="My Spec", rated_out=rated_out
        )
        assert "named::My Spec" in leaves[0].output.rating.requirement_ratings
        assert len(leaves[0].feedback()) == 1
        assert len(leaves[0].claim_reviews()) == 1

        unrate_reviewed_batch_runs(rated_out)

        rating = leaves[0].output.rating
        assert rating is not None
        assert "named::My Spec" not in rating.requirement_ratings
        assert "named::Other Spec" in rating.requirement_ratings
        assert leaves[0].feedback() == []
        assert leaves[0].claim_reviews() == []

    def test_failure_mid_children_still_rolls_back_the_rating(self, task_with_leaves):
        # The rating is persisted before the Feedback/ClaimReview children; a
        # failure saving a child must still leave the leaf recoverable via
        # rated_out (recorded as soon as the rating hits disk).
        _, leaves = task_with_leaves
        reviewed = [
            ReviewedChainApi(
                leaf_run_id=leaves[0].id,
                user_says_meets_spec=False,
                feedback="why",
                claim_review=_claim_review_api(),
            ),
        ]
        rated_out: list = []
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.save_claim_review",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                rate_reviewed_batch_runs(
                    leaves, reviewed, spec_name="My Spec", rated_out=rated_out
                )

        # The mutated leaf was captured despite the mid-children failure...
        assert len(rated_out) == 1
        assert "named::My Spec" in leaves[0].output.rating.requirement_ratings

        # ...so rollback restores it (rating gone, feedback child deleted).
        unrate_reviewed_batch_runs(rated_out)
        assert leaves[0].output.rating is None
        assert leaves[0].feedback() == []
        assert leaves[0].claim_reviews() == []


class TestSavePendingChildren:
    def test_persists_feedback_and_claim_review(self, task_with_leaves):
        task, _ = task_with_leaves
        reviewed = ReviewedExample(
            input="What's the return window?",
            output="30 days.",
            model_says_meets_spec=False,
            user_says_meets_spec=False,
            feedback="Fabricated the window.",
            claim_review=_claim_review_api(),
        )
        dataset = create_single_turn_dataset(
            [], [reviewed], "eval_tag", "train_tag", "golden_tag", "My Spec"
        )
        assert len(dataset.task_runs) == 1
        run = dataset.task_runs[0]
        run.parent = task
        run.save_to_file()
        dataset.save_pending_children(run)

        feedback = run.feedback()
        assert len(feedback) == 1
        assert feedback[0].feedback == "Fabricated the window."
        reviews = run.claim_reviews()
        assert len(reviews) == 1
        assert reviews[0].judge_score == "fail"
        assert [c.human_grade for c in reviews[0].claims] == ["disagree", "agree"]
        assert reviews[0].human_verdict == "fail"


def _make_su_leaves(task: Task, n: int) -> list[TaskRun]:
    """Persist n synthetic-user chain leaves under a task, tagged like the runner."""
    source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "haiku",
            "model_provider": "openrouter",
            "adapter_name": "kiln_synthetic_user_runner",
        },
    )
    leaves = []
    for i in range(n):
        run = TaskRun(
            parent=task,
            input=f"input {i}",
            input_source=source,
            output=TaskOutput(output=f"output {i}", source=source),
            tags=["synthetic_user_case", "synthetic_user_batch:b1"],
        )
        run.save_to_file()
        leaves.append(run)
    return leaves


def _leaf_split(leaves: list[TaskRun]) -> dict[str, list[TaskRun]]:
    return {
        "eval": [x for x in leaves if "eval_tag" in (x.tags or [])],
        "train": [x for x in leaves if "train_tag" in (x.tags or [])],
        "val": [x for x in leaves if "val_tag" in (x.tags or [])],
        "golden": [x for x in leaves if "golden_tag" in (x.tags or [])],
    }


class TestDealPoolTrainVal:
    """The non-golden pool is dealt train:val at 40:25 by largest remainder."""

    @pytest.mark.parametrize(
        "pool_size,expected_train,expected_val",
        [
            # 0 and 1 are the degenerate pools: 1 run's remainders are
            # 40/65 vs 25/65, so the single seat goes to train.
            (0, 0, 0),
            (1, 1, 0),
            (2, 1, 1),
            # 30 * 40 // 65 = 18, 30 * 25 // 65 = 11, leftover seat to val
            # (remainders 30 train vs 35 val) → the 18/12 the scheme wants.
            (30, 18, 12),
            # 65 divides exactly: no leftover seat to award.
            (65, 40, 25),
        ],
    )
    def test_counts_by_largest_remainder(self, pool_size, expected_train, expected_val):
        pool = list(range(pool_size))
        train, val = deal_pool_train_val(pool, random.Random(0))
        assert (len(train), len(val)) == (expected_train, expected_val)
        # Largest remainder drops nobody and duplicates nobody.
        assert sorted(train + val) == pool

    def test_does_not_mutate_input(self):
        pool = list(range(30))
        deal_pool_train_val(pool, random.Random(1))
        assert pool == list(range(30))


class TestSplitAndTagMultiTurnChains:
    def test_all_reviewed_splits_golden_cap_rest_dealt(self, multiturn_task):
        # Mirrors the real UI: every chain reviewed before save. golden caps
        # at 25% of 8 = 2; the 6 left over are dealt train:val (the eval slice
        # is EvalInput items minted from the cases, not chains).
        leaves = _make_su_leaves(multiturn_task, 8)
        reviewed_ids = {leaf.id for leaf in leaves}

        split_and_tag_batch_runs(
            leaves,
            reviewed_ids,
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(0),
        )

        buckets = _leaf_split(leaves)
        assert len(buckets["golden"]) == 2
        assert buckets["eval"] == []
        # 6 * 40 // 65 = 3 train, 6 * 25 // 65 = 2 val, leftover seat to train.
        assert len(buckets["train"]) == 4
        assert len(buckets["val"]) == 2
        # Golden is a subset of the reviewed leaves (rated-only answer key).
        assert {x.id for x in buckets["golden"]} <= reviewed_ids

    def test_each_leaf_gets_exactly_one_split_tag(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 5)
        split_and_tag_batch_runs(
            leaves,
            {leaves[0].id},
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(1),
        )
        for leaf in leaves:
            split_tags = {"train_tag", "golden_tag", "val_tag"} & set(leaf.tags)
            assert len(split_tags) == 1

    def test_golden_capped_even_when_all_reviewed(self, multiturn_task):
        # 4 leaves all reviewed → golden caps at 1 (not 4); the dealt slices
        # are never starved to empty (the bug the cap fixes).
        leaves = _make_su_leaves(multiturn_task, 4)
        split_and_tag_batch_runs(
            leaves,
            {leaf.id for leaf in leaves},
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(7),
        )
        buckets = _leaf_split(leaves)
        assert len(buckets["golden"]) == 1
        assert len(buckets["train"]) == 2
        assert len(buckets["val"]) == 1

    def test_zero_rated_no_golden(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 3)
        split_and_tag_batch_runs(
            leaves,
            set(),
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(2),
        )
        buckets = _leaf_split(leaves)
        assert buckets["golden"] == []
        assert len(buckets["train"]) == 2
        assert len(buckets["val"]) == 1

    def test_deal_is_driven_by_the_injected_rng(self, multiturn_task):
        # The injected rng, and only it, decides who is held out: the same
        # seed reproduces a save's val membership, a different seed does not.
        # A deal that read the pool in order instead would pass the first
        # assertion and fail the second.
        def tagged_val_inputs(seed: int) -> set[str]:
            leaves = _make_su_leaves(multiturn_task, 30)
            split_and_tag_batch_runs(
                leaves,
                set(),
                "train_tag",
                "golden_tag",
                "val_tag",
                rng=random.Random(seed),
            )
            return {leaf.input for leaf in _leaf_split(leaves)["val"]}

        first = tagged_val_inputs(11)
        assert len(first) == 12  # 30 unreviewed → 18 train / 12 val
        assert tagged_val_inputs(11) == first
        assert tagged_val_inputs(999) != first

    def test_preserves_existing_runner_tags(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 4)
        split_and_tag_batch_runs(
            leaves,
            {leaf.id for leaf in leaves},
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(3),
        )
        for leaf in leaves:
            assert "synthetic_user_case" in leaf.tags
            assert "synthetic_user_batch:b1" in leaf.tags

    def test_tagged_out_captures_additions_for_rollback(self, multiturn_task):
        leaves = _make_su_leaves(multiturn_task, 4)
        tagged_out: list = []
        split_and_tag_batch_runs(
            leaves,
            {leaf.id for leaf in leaves},
            "train_tag",
            "golden_tag",
            "val_tag",
            rng=random.Random(4),
            tagged_out=tagged_out,
        )
        # Every leaf was mutated exactly once (one split tag added each), and
        # val rides the same ledger as train so rollback reverses it too.
        assert len(tagged_out) == 4
        assert all(len(added) == 1 for _, added in tagged_out)
        assert {tag for _, added in tagged_out for tag in added} == {
            "golden_tag",
            "train_tag",
            "val_tag",
        }


# ───────────────── delete_multi_turn_batch_chains ─────────────────


def _su_source(turn_index: int) -> DataSource:
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "haiku",
            "model_provider": "openrouter",
            "adapter_name": "kiln_synthetic_user_runner",
            "batch_tag": "batch1",
            "turn_index": turn_index,
        },
    )


def _build_chain(task: Task, batch_tag: str, turns: int = 2) -> list[TaskRun]:
    """A root→leaf chain shaped like the SU runner's output: only the leaf
    carries the discovery tags."""
    chain: list[TaskRun] = []
    parent_id = None
    for i in range(turns):
        run = TaskRun(
            parent=task,
            input=f"turn input {i}",
            input_source=_su_source(i + 1),
            output=TaskOutput(output=f"turn output {i}", source=_su_source(i + 1)),
            parent_task_run_id=parent_id,
        )
        run.save_to_file()
        chain.append(run)
        parent_id = str(run.id)
    leaf = chain[-1]
    leaf.tags = sorted({"synthetic_user_case", f"synthetic_user_batch:{batch_tag}"})
    leaf.save_to_file()
    return chain


@pytest.fixture
def multiturn_task(tmp_path):
    project_path = tmp_path / "mt_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="MT Project", path=project_path)
    project.save_to_file()
    task = Task(
        name="MT Task",
        instruction="Test instruction",
        turn_mode=TurnMode.multiturn,
        parent=project,
    )
    task.save_to_file()
    return task


class TestDeleteMultiTurnBatchChains:
    def test_deletes_whole_chains_of_the_batch(self, multiturn_task):
        chain_a = _build_chain(multiturn_task, "old-batch", turns=3)
        chain_b = _build_chain(multiturn_task, "old-batch", turns=2)

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 5
        assert multiturn_task.runs(include_intermediate_runs=True) == []
        for run in [*chain_a, *chain_b]:
            assert run.path is not None and not run.path.exists()

    def test_other_batches_survive(self, multiturn_task):
        _build_chain(multiturn_task, "old-batch", turns=2)
        keep = _build_chain(multiturn_task, "new-batch", turns=2)

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 2
        remaining_ids = {
            str(r.id) for r in multiturn_task.runs(include_intermediate_runs=True)
        }
        assert remaining_ids == {str(r.id) for r in keep}

    def test_skips_chain_claimed_by_another_flow(self, multiturn_task):
        """A leaf with tags beyond the runner's own (an eval save tagged it)
        is part of a dataset, not an abandoned drive — left alone."""
        chain = _build_chain(multiturn_task, "old-batch", turns=2)
        leaf = chain[-1]
        leaf.tags = sorted({*(leaf.tags or []), "eval_config_my_spec"})
        leaf.save_to_file()

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 0
        assert len(multiturn_task.runs(include_intermediate_runs=True)) == 2

    def test_skips_rated_leaf(self, multiturn_task):
        """A rated leaf is answer-key material — never delete it."""
        chain = _build_chain(multiturn_task, "old-batch", turns=2)
        leaf = chain[-1]
        leaf.output.rating = TaskOutputRating(
            type=TaskOutputRatingType.pass_fail, value=1.0
        )
        leaf.save_to_file()

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 0
        assert len(multiturn_task.runs(include_intermediate_runs=True)) == 2

    def test_unknown_batch_tag_is_a_noop(self, multiturn_task):
        _build_chain(multiturn_task, "some-batch", turns=2)
        assert delete_multi_turn_batch_chains(multiturn_task, "nonexistent") == 0
        assert len(multiturn_task.runs(include_intermediate_runs=True)) == 2

    def test_skips_leaf_with_descendants(self, multiturn_task):
        """A tagged run that gained children (a continued conversation) is no
        longer a chain leaf — deleting it would strand its descendants."""
        chain = _build_chain(multiturn_task, "old-batch", turns=2)
        continued = TaskRun(
            parent=multiturn_task,
            input="follow-up turn",
            input_source=_su_source(3),
            output=TaskOutput(output="reply", source=_su_source(3)),
            parent_task_run_id=str(chain[-1].id),
        )
        continued.save_to_file()

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 0
        assert len(multiturn_task.runs(include_intermediate_runs=True)) == 3

    def test_skips_chain_forked_mid_conversation(self, multiturn_task):
        """A conversation continued from a MID-chain turn parents a run
        outside the chain — deleting the ancestors would dangle the fork's
        parent_task_run_id, so the whole chain is left alone."""
        chain = _build_chain(multiturn_task, "old-batch", turns=3)
        fork = TaskRun(
            parent=multiturn_task,
            input="fork from the first turn",
            input_source=_su_source(9),
            output=TaskOutput(output="reply", source=_su_source(9)),
            parent_task_run_id=str(chain[0].id),
        )
        fork.save_to_file()

        deleted = delete_multi_turn_batch_chains(multiturn_task, "old-batch")

        assert deleted == 0
        assert len(multiturn_task.runs(include_intermediate_runs=True)) == 4


# ───────────── single-turn drive tags + delete_single_turn_batch_runs ───────


def _single_turn_source() -> DataSource:
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt_5_5_mini",
            "model_provider": "openrouter",
            "adapter_name": "kiln_eval_builder_single_turn",
            "batch_tag": "batch1",
        },
    )


def _build_single_turn_run(task: Task, batch_tag: str, i: int = 0) -> TaskRun:
    """One driven run shaped like the single-turn pipeline's output: saved,
    then tagged through the real tagging helper."""
    run = TaskRun(
        parent=task,
        input=f"input {i}",
        input_source=_single_turn_source(),
        output=TaskOutput(output=f"output {i}", source=_single_turn_source()),
    )
    run.save_to_file()
    tag_single_turn_drive_run(run, batch_tag)
    return run


@pytest.fixture
def singleturn_task(tmp_path):
    project_path = tmp_path / "st_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="ST Project", path=project_path)
    project.save_to_file()
    task = Task(
        name="ST Task",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()
    return task


class TestSingleTurnDriveTags:
    def test_tags_and_persists(self, singleturn_task):
        run = _build_single_turn_run(singleturn_task, "batch42")
        reloaded = singleturn_task.runs()[0]
        assert reloaded.tags == sorted(
            ["single_turn_drive", "single_turn_drive_batch:batch42"]
        )
        assert str(reloaded.id) == str(run.id)

    def test_retagging_is_idempotent(self, singleturn_task):
        run = _build_single_turn_run(singleturn_task, "batch42")
        tag_single_turn_drive_run(run, "batch42")
        assert run.tags == sorted(
            ["single_turn_drive", "single_turn_drive_batch:batch42"]
        )

    def test_find_returns_only_the_batch(self, singleturn_task):
        run_a = _build_single_turn_run(singleturn_task, "batch-a", 0)
        _build_single_turn_run(singleturn_task, "batch-b", 1)
        found = find_single_turn_batch_runs(singleturn_task, "batch-a")
        assert [str(r.id) for r in found] == [str(run_a.id)]


class TestDeleteSingleTurnBatchRuns:
    def test_deletes_the_batch(self, singleturn_task):
        run_a = _build_single_turn_run(singleturn_task, "old-batch", 0)
        run_b = _build_single_turn_run(singleturn_task, "old-batch", 1)

        deleted = delete_single_turn_batch_runs(singleturn_task, "old-batch")

        assert deleted == 2
        assert singleturn_task.runs() == []
        for run in (run_a, run_b):
            assert run.path is not None and not run.path.exists()

    def test_other_batches_survive(self, singleturn_task):
        _build_single_turn_run(singleturn_task, "old-batch", 0)
        keep = _build_single_turn_run(singleturn_task, "new-batch", 1)

        deleted = delete_single_turn_batch_runs(singleturn_task, "old-batch")

        assert deleted == 1
        assert [str(r.id) for r in singleturn_task.runs()] == [str(keep.id)]

    def test_skips_run_claimed_by_another_flow(self, singleturn_task):
        """A run with tags beyond the pipeline's own (an eval save tagged it
        golden/train) is dataset material, not an abandoned drive artifact —
        left alone. The exact-set match fails CLOSED."""
        run = _build_single_turn_run(singleturn_task, "old-batch")
        run.tags = sorted({*(run.tags or []), "eval_config_my_spec"})
        run.save_to_file()

        deleted = delete_single_turn_batch_runs(singleturn_task, "old-batch")

        assert deleted == 0
        assert len(singleturn_task.runs()) == 1

    def test_skips_rated_run(self, singleturn_task):
        """A rated run is answer-key material — never delete it."""
        run = _build_single_turn_run(singleturn_task, "old-batch")
        run.output.rating = TaskOutputRating(
            type=TaskOutputRatingType.pass_fail, value=1.0
        )
        run.save_to_file()

        deleted = delete_single_turn_batch_runs(singleturn_task, "old-batch")

        assert deleted == 0
        assert len(singleturn_task.runs()) == 1

    def test_unknown_batch_tag_is_a_noop(self, singleturn_task):
        _build_single_turn_run(singleturn_task, "some-batch")
        assert delete_single_turn_batch_runs(singleturn_task, "nonexistent") == 0
        assert len(singleturn_task.runs()) == 1


# ───────────────── multi-turn eval slice (EvalInput writer) ─────────────────


def _driven_case(idx: int, scenario_index: int | None = None) -> DrivenSyntheticCaseApi:
    return DrivenSyntheticCaseApi(
        seed_prompt=f"seed {idx}",
        synthetic_user_info=(
            f"<persona>persona {idx}</persona>"
            f"<goal>goal {idx}</goal>"
            f"<behavior_guidance>guidance {idx}</behavior_guidance>"
        ),
        scenario_index=scenario_index,
    )


_DRIVE_CONFIG = MultiTurnDriveConfig(
    model_name="claude_4_5_haiku", model_provider="openrouter", turns=5
)


class TestBuildMultiTurnEvalInputs:
    def test_mints_one_eval_input_per_case(self, multiturn_task):
        cases = [_driven_case(0, scenario_index=2), _driven_case(1)]
        eval_inputs = build_multi_turn_eval_inputs(
            cases, "batch99", multiturn_task, "eval_myspec", _DRIVE_CONFIG
        )

        assert len(eval_inputs) == 2
        first = eval_inputs[0]
        assert first.data.type == "multi_turn_synthetic"
        assert first.data.first_message is not None
        assert first.data.first_message.text == "seed 0"
        assert first.data.synthetic_user_info.persona == "persona 0"
        assert first.data.synthetic_user_info.goal == "goal 0"
        assert first.data.synthetic_user_info.behavior_guidance == "guidance 0"
        # Every minted item is stamped with the batch's drive settings.
        assert all(ei.data.drive_config == _DRIVE_CONFIG for ei in eval_inputs)
        # Slice tag + provenance: the synthetic-user batch the case was
        # driven in, and the batch-plan scenario it came from.
        assert first.tags == [
            "eval_myspec",
            "synthetic_user_batch:batch99",
            "scenario:2",
        ]
        # No scenario_index → no scenario tag.
        assert eval_inputs[1].tags == ["eval_myspec", "synthetic_user_batch:batch99"]
        # Built, validated, NOT saved — persistence is the unit of work's job.
        assert multiturn_task.eval_inputs(readonly=True) == []

    def test_malformed_blob_is_422(self, multiturn_task):
        bad = DrivenSyntheticCaseApi(
            seed_prompt="seed", synthetic_user_info="no tags at all"
        )
        with pytest.raises(HTTPException) as exc:
            build_multi_turn_eval_inputs(
                [_driven_case(0), bad], "b1", multiturn_task, "eval_x", _DRIVE_CONFIG
            )
        assert exc.value.status_code == 422
        assert "Case 1" in exc.value.detail
        assert multiturn_task.eval_inputs(readonly=True) == []


class TestPersistEvalSlice:
    def test_persists_and_ledgers_each_item(self, multiturn_task):
        eval_inputs = build_multi_turn_eval_inputs(
            [_driven_case(0), _driven_case(1)],
            "b1",
            multiturn_task,
            "eval_x",
            _DRIVE_CONFIG,
        )
        saved_out: list = []
        persist_eval_slice(eval_inputs, saved_out)

        on_disk = multiturn_task.eval_inputs(readonly=True)
        assert len(on_disk) == 2
        # Every persisted item is in the rollback ledger.
        assert saved_out == eval_inputs


@pytest.fixture
def project_and_task(tmp_path):
    """An empty saved project + task — the starting point for the capability
    tests, which build their own run config on top."""
    project = Project(name="Capability Project", path=tmp_path / "project.kiln")
    project.save_to_file()
    task = Task(name="Capability Task", instruction="Do the thing.", parent=project)
    task.save_to_file()
    return project, task


class TestTaskCapabilitiesForTask:
    async def test_reads_tools_and_skills_from_default_run_config(
        self, project_and_task, give_task_one_tool_and_skill
    ):
        project, task = project_and_task
        give_task_one_tool_and_skill(project, task)

        tools, skills = await task_capabilities_for_task(task)

        assert tools == [
            TaskToolInfoApi(
                name="add", description="Add two numbers together and return the result"
            )
        ]
        assert skills == [
            TaskSkillInfoApi(
                name="refund-policy", description="How and when refunds are issued."
            )
        ]

    async def test_collection_is_logged_with_counts(
        self, project_and_task, give_task_one_tool_and_skill, caplog
    ):
        """Resolving tools can dial MCP servers, so the cost of collection is
        logged rather than capped — it has to be visible in the logs."""
        project, task = project_and_task
        give_task_one_tool_and_skill(project, task)

        with caplog.at_level(logging.INFO):
            await task_capabilities_for_task(task)

        assert "1 tools, 1 skills" in caplog.text

    async def test_tool_order_follows_the_run_config(
        self, project_and_task, agent_run_config_properties, set_default_run_config
    ):
        """Tools are reported in the order the run config lists them, so the
        payload matches the surface the model is actually given."""
        _, task = project_and_task
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(
                    tools=["kiln_tool::multiply_numbers", "kiln_tool::add_numbers"]
                )
            ),
        )

        tools, _ = await task_capabilities_for_task(task)

        assert [tool.name for tool in tools or []] == ["multiply", "add"]

    async def test_no_default_run_config_is_not_collected(self, project_and_task):
        """No default config means the capabilities are unknown, not empty."""
        _, task = project_and_task
        assert await task_capabilities_for_task(task) == (None, None)

    async def test_dangling_default_run_config_id_is_not_collected(
        self, project_and_task
    ):
        _, task = project_and_task
        task.default_run_config_id = "does-not-exist"
        task.save_to_file()
        assert await task_capabilities_for_task(task) == (None, None)

    async def test_unreadable_storage_degrades_to_not_collected(
        self, project_and_task, caplog
    ):
        """Collection reads run configs and skills off disk. A corrupt or
        forward-versioned file must degrade to the un-enriched prompt rather
        than failing the whole spec-building request."""
        _, task = project_and_task
        task.default_run_config_id = "rc-1"
        with patch.object(
            type(task), "run_configs", side_effect=ValueError("corrupt run_config.kiln")
        ):
            with caplog.at_level(logging.WARNING):
                result = await task_capabilities_for_task(task)

        assert result == (None, None)
        assert "corrupt run_config.kiln" in caplog.text

    async def test_non_agent_run_config_has_no_capabilities(
        self, project_and_task, set_default_run_config
    ):
        """An MCP run config carries no tools_config and loads no skills, so
        its surface is genuinely empty rather than uncollected."""
        _, task = project_and_task
        set_default_run_config(
            task,
            McpRunConfigProperties(
                tool_reference=MCPToolReference(tool_id="mcp::local::server::do_thing")
            ),
        )
        assert await task_capabilities_for_task(task) == ([], [])

    async def test_agent_config_without_tools_config_has_no_capabilities(
        self, project_and_task, agent_run_config_properties, set_default_run_config
    ):
        _, task = project_and_task
        set_default_run_config(task, agent_run_config_properties())
        assert await task_capabilities_for_task(task) == ([], [])

    async def test_skill_only_config_reports_skills_and_no_tools(
        self,
        project_and_task,
        save_skill,
        agent_run_config_properties,
        set_default_run_config,
    ):
        """Skill ids live in the same tools list but must never be resolved as
        tools — tool_from_id rejects them."""
        project, task = project_and_task
        skill = save_skill(project, "escalation", "When to escalate.")
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(tools=[f"kiln_tool::skill::{skill.id}"])
            ),
        )

        tools, skills = await task_capabilities_for_task(task)

        assert tools == []
        assert skills == [
            TaskSkillInfoApi(name="escalation", description="When to escalate.")
        ]

    async def test_unresolvable_tool_is_skipped(
        self,
        project_and_task,
        agent_run_config_properties,
        set_default_run_config,
        caplog,
    ):
        """A broken tool reference must not take down spec building; the rest
        of the surface is still reported."""
        _, task = project_and_task
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(
                    tools=["mcp::local::gone::vanished", "kiln_tool::add_numbers"]
                )
            ),
        )

        with caplog.at_level(logging.WARNING):
            tools, _ = await task_capabilities_for_task(task)

        assert [tool.name for tool in tools or []] == ["add"]
        assert "mcp::local::gone::vanished" in caplog.text

    async def test_skills_are_sorted_by_name(
        self,
        project_and_task,
        save_skill,
        agent_run_config_properties,
        set_default_run_config,
    ):
        """The skill loader returns an unordered map; sorting keeps the same
        task producing the same payload every call."""
        project, task = project_and_task
        zebra = save_skill(project, "zebra", "Last alphabetically.")
        alpha = save_skill(project, "alpha", "First alphabetically.")
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(
                    tools=[
                        f"kiln_tool::skill::{zebra.id}",
                        f"kiln_tool::skill::{alpha.id}",
                    ]
                )
            ),
        )

        _, skills = await task_capabilities_for_task(task)

        assert [skill.name for skill in skills or []] == ["alpha", "zebra"]


class TestTaskCapabilitiesForANamedRunConfig:
    """A caller can name the run config it is asking about — the one an eval is
    written against — instead of taking whatever the task defaults to."""

    @pytest.fixture
    def task_with_a_second_run_config(
        self,
        project_and_task,
        give_task_one_tool_and_skill,
        agent_run_config_properties,
    ):
        """A task whose default gives it `add` plus a skill, and a second saved
        config that gives it `multiply` and nothing else."""
        project, task = project_and_task
        give_task_one_tool_and_skill(project, task)
        other = TaskRunConfig(
            name="other",
            run_config_properties=agent_run_config_properties(
                tools_config=ToolsRunConfig(tools=["kiln_tool::multiply_numbers"])
            ),
            parent=task,
        )
        other.save_to_file()
        return task, other

    async def test_reads_the_named_config_rather_than_the_default(
        self, task_with_a_second_run_config
    ):
        task, other = task_with_a_second_run_config

        tools, skills = await task_capabilities_for_task(task, other.id)

        assert [tool.name for tool in tools or []] == ["multiply"]
        # The skill belongs to the default config, not this one.
        assert skills == []

    async def test_no_id_still_reads_the_default(self, task_with_a_second_run_config):
        """The second config must not change what an un-named call reports."""
        task, _ = task_with_a_second_run_config

        tools, skills = await task_capabilities_for_task(task)

        assert [tool.name for tool in tools or []] == ["add"]
        assert [skill.name for skill in skills or []] == ["refund-policy"]

    async def test_a_named_config_is_read_even_without_a_default(
        self, project_and_task, agent_run_config_properties
    ):
        """Naming a config is a complete answer on its own — a task with no
        default is still fully described."""
        _, task = project_and_task
        run_config = TaskRunConfig(
            name="only",
            run_config_properties=agent_run_config_properties(
                tools_config=ToolsRunConfig(tools=["kiln_tool::add_numbers"])
            ),
            parent=task,
        )
        run_config.save_to_file()

        tools, _ = await task_capabilities_for_task(task, run_config.id)

        assert [tool.name for tool in tools or []] == ["add"]

    @pytest.mark.parametrize(
        "run_config_id", ["does-not-exist", ""], ids=["unknown_id", "empty_id"]
    )
    async def test_an_unresolvable_named_config_is_404(
        self, task_with_a_second_run_config, run_config_id
    ):
        """Fail loud: falling back to the default would describe a config the
        caller never asked about, with nothing on the wire to say so. An empty
        string is a supplied id too."""
        task, _ = task_with_a_second_run_config

        with pytest.raises(HTTPException) as exc:
            await task_capabilities_for_task(task, run_config_id)

        assert exc.value.status_code == 404


class TestTaskCapabilitiesRunContext:
    """Collection runs inside one MCP session scope, so every tool on a server
    resolves through one shared session instead of connecting per tool.

    The scope's own semantics are covered in test_mcp_session_manager.py; these
    assert the adoption — that collection is actually wrapped in one.
    """

    # The scope tears down through its own module, so that is where the
    # manager is replaced. The tool resolves sessions through its own binding,
    # patched separately where a test needs to serve one.
    SCOPE_MANAGER_PATCH: ClassVar[str] = (
        "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
    )
    TOOL_MANAGER_PATCH: ClassVar[str] = (
        "kiln_ai.tools.mcp_server_tool.MCPSessionManager"
    )
    RUN_ID_PATCH: ClassVar[str] = (
        "kiln_ai.tools.mcp_session_manager.generate_agent_run_id"
    )

    @pytest.fixture(autouse=True)
    def clear_context(self):
        """No ambient run id, or collection would join a scope it should be
        opening for itself."""
        clear_agent_run_id()
        yield
        clear_agent_run_id()

    @pytest.fixture
    def task_with_two_mcp_tools(
        self, project_and_task, agent_run_config_properties, set_default_run_config
    ):
        """A task whose default run config lists two tools from ONE MCP
        server — the case that used to cost two connections."""
        project, task = project_and_task
        server = ExternalToolServer(
            name="test_server",
            type=ToolServerType.remote_mcp,
            properties={"server_url": "https://example.com", "is_archived": False},
            parent=project,
        )
        server.save_to_file()
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(
                    tools=[
                        f"mcp::remote::{server.id}::alpha",
                        f"mcp::remote::{server.id}::beta",
                    ]
                )
            ),
        )
        return task, server

    @pytest.fixture
    def mcp_session_serving_alpha_and_beta(self):
        """A warm session that answers list_tools with both of the server's
        tools, the way one shared connection serves the whole collection."""
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    MCPTool(
                        name="alpha",
                        description="Alpha tool",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    MCPTool(
                        name="beta",
                        description="Beta tool",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )
        )
        return session

    async def test_same_server_tools_share_one_session(
        self, task_with_two_mcp_tools, mcp_session_serving_alpha_and_beta
    ):
        """Both tools resolve through get_or_create_session under the same
        (server, run id) scope, which the session manager serves from one
        connection, and that same scope is the one torn down afterwards. The
        per-call ephemeral client (a fresh connect + teardown per tool) is
        never reached."""
        task, server = task_with_two_mcp_tools
        cleanup_mock = AsyncMock()

        with (
            patch(self.TOOL_MANAGER_PATCH) as mock_manager_cls,
            patch(self.SCOPE_MANAGER_PATCH) as mock_scope_manager_cls,
        ):
            shared = mock_manager_cls.shared.return_value
            shared.get_or_create_session = AsyncMock(
                return_value=mcp_session_serving_alpha_and_beta
            )
            mock_scope_manager_cls.shared.return_value.cleanup_session = cleanup_mock
            tools, _ = await task_capabilities_for_task(task)

        assert tools == [
            TaskToolInfoApi(name="alpha", description="Alpha tool"),
            TaskToolInfoApi(name="beta", description="Beta tool"),
        ]
        session_calls = shared.get_or_create_session.call_args_list
        assert session_calls
        # Asserted as a set of scopes rather than a call count: how many times
        # a warm session is asked for is an implementation detail, but every
        # ask landing on ONE (server, run id) pair is the reuse itself.
        scopes = {(call.args[0].id, call.args[1]) for call in session_calls}
        assert len(scopes) == 1
        scoped_server_id, scoped_run_id = next(iter(scopes))
        assert scoped_server_id == server.id
        shared.mcp_client.assert_not_called()
        # The scope torn down must be the one the sessions live under, or they
        # leak for the life of the process.
        cleanup_mock.assert_called_once_with(scoped_run_id)
        assert get_agent_run_id() is None

    async def test_session_is_cleaned_up_when_collection_fails(
        self, project_and_task, give_task_one_tool_and_skill, caplog
    ):
        """The scope still has to close on the failure path: the collection
        may already have opened sessions before it failed, and the manager has
        no reaper to catch them."""
        project, task = project_and_task
        give_task_one_tool_and_skill(project, task)
        cleanup_mock = AsyncMock()

        with (
            patch(self.SCOPE_MANAGER_PATCH) as mock_scope_manager_cls,
            patch(self.RUN_ID_PATCH, return_value="run_collection"),
            patch.object(
                type(task), "run_configs", side_effect=ValueError("corrupt file")
            ),
            caplog.at_level(logging.WARNING),
        ):
            mock_scope_manager_cls.shared.return_value.cleanup_session = cleanup_mock
            assert await task_capabilities_for_task(task) == (None, None)

        # The degrade came from the failing read, not from an earlier return
        # that would never have opened a scope at all.
        assert "corrupt file" in caplog.text
        cleanup_mock.assert_called_once_with("run_collection")
        assert get_agent_run_id() is None

    async def test_a_callers_run_context_is_joined_not_torn_down(
        self, task_with_two_mcp_tools, mcp_session_serving_alpha_and_beta
    ):
        """Inside an existing run context the tools resolve under the CALLER's
        scope, and the sessions are the caller's to close: tearing them down
        here would drop connections it still needs.

        Guards against collection going back to minting a run id of its own
        unconditionally, which is what the scope replaced.
        """
        task, server = task_with_two_mcp_tools
        cleanup_mock = AsyncMock()
        set_agent_run_id("caller_run_id")

        with (
            patch(self.TOOL_MANAGER_PATCH) as mock_manager_cls,
            patch(self.SCOPE_MANAGER_PATCH) as mock_scope_manager_cls,
        ):
            shared = mock_manager_cls.shared.return_value
            shared.get_or_create_session = AsyncMock(
                return_value=mcp_session_serving_alpha_and_beta
            )
            mock_scope_manager_cls.shared.return_value.cleanup_session = cleanup_mock
            await task_capabilities_for_task(task)

        scopes = {
            (call.args[0].id, call.args[1])
            for call in shared.get_or_create_session.call_args_list
        }
        assert scopes == {(server.id, "caller_run_id")}
        cleanup_mock.assert_not_called()
        assert get_agent_run_id() == "caller_run_id"


class TestTaskInfoPayload:
    """The single owner of capability-key omission on the wire."""

    BASE: ClassVar[dict] = {
        "task_prompt": "p",
        "task_input_schema": "in",
        "task_output_schema": "out",
    }

    def test_uncollected_capabilities_are_omitted_not_nulled(self):
        """None means not collected, which the contract reads as an absent
        key — sending an explicit null would change a payload that must stay
        exactly as it was before capabilities existed."""
        assert task_info_payload(TaskInfoApi(**self.BASE)) == self.BASE

    def test_empty_capabilities_are_sent(self):
        """[] says the task genuinely has none, which is worth telling the
        model, so it must survive onto the wire."""
        payload = task_info_payload(
            TaskInfoApi(**self.BASE, task_tools=[], task_skills=[])
        )
        assert payload == {**self.BASE, "task_tools": [], "task_skills": []}

    def test_each_side_is_omitted_independently(self):
        payload = task_info_payload(
            TaskInfoApi(
                **self.BASE,
                task_tools=[TaskToolInfoApi(name="add", description="Adds.")],
            )
        )
        assert payload == {
            **self.BASE,
            "task_tools": [{"name": "add", "description": "Adds."}],
        }
