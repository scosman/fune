"""Utility functions for creating specs with Kiln Copilot.

This module contains helper functions and constants for creating specs,
evals, eval configs, and task runs as part of the copilot-assisted
spec creation workflow.
"""

import logging
import random
import time
from typing import Any, TypeVar

from fastapi import HTTPException
from kiln_ai.adapters.adapter_registry import load_skills_for_task
from kiln_ai.datamodel import ClaimReview, Feedback, FeedbackSource, Task, TaskRun
from kiln_ai.datamodel.datamodel_enums import TaskOutputRatingType
from kiln_ai.datamodel.eval import (
    EvalInput,
    MultiTurnDriveConfig,
    MultiTurnSyntheticEvalInputData,
    SingleTurnEvalInputData,
    UserMessage,
)
from kiln_ai.datamodel.run_config import as_kiln_agent_run_config
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_output import (
    DataSource,
    DataSourceType,
    RequirementRating,
    TaskOutput,
    TaskOutputRating,
)
from kiln_ai.datamodel.tool_id import SKILL_TOOL_ID_PREFIX
from kiln_ai.synthetic_user.parser import (
    SyntheticUserInfoParseError,
    parse_synthetic_user_info,
)
from kiln_ai.tools.mcp_session_manager import mcp_session_scope
from kiln_ai.tools.tool_registry import tool_from_id
from kiln_ai.utils.config import Config

from app.desktop.studio_server.api_client.kiln_ai_server_client.api.copilot import (
    generate_batch_v1_copilot_generate_batch_post,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    GenerateBatchInput,
    GenerateBatchOutput,
)
from app.desktop.studio_server.api_client.kiln_server_client import (
    get_authenticated_client,
)
from app.desktop.studio_server.api_models.copilot_models import (
    ClaimReviewApi,
    DrivenSyntheticCaseApi,
    ReviewedChainApi,
    ReviewedExample,
    SampleApi,
    SyntheticDataGenerationSessionConfigApi,
    TaskInfoApi,
    TaskSkillInfoApi,
    TaskToolInfoApi,
)
from app.desktop.studio_server.utils.response_utils import unwrap_response

logger = logging.getLogger(__name__)

# Tag scheme the multi-turn synthetic-user runner stamps on each chain's leaf
# TaskRun — see kiln_ai.synthetic_user.runner. Kept in sync manually; if the
# runner ever changes its tag scheme these constants move too.
_TAG_PREFIX_SU_BATCH = "synthetic_user_batch:"
_TAG_SU_CASE = "synthetic_user_case"

# Tag scheme the single-turn pipeline stamps on each run it drives (the
# one-turn sibling of the runner scheme above): a marker tag for all
# wizard-driven single-turn runs plus a batch tag grouping one drive.
# Discovery is tag-based — save and delete-on-redrive both find a batch's
# runs through these.
_TAG_PREFIX_SINGLE_TURN_DRIVE_BATCH = "single_turn_drive_batch:"
_TAG_SINGLE_TURN_DRIVE = "single_turn_drive"

# Constants for copilot spec creation
KILN_COPILOT_MODEL_NAME = "kiln-copilot"
KILN_COPILOT_MODEL_PROVIDER = "kiln"
KILN_ADAPTER_NAME = "kiln-adapter"

# Single-turn synthetic generation sizes: how many examples the copilot API is
# asked to produce for the review + eval datasets. Owned here; the review UI
# advertises the resulting dataset size to the user off these.
NUM_SAMPLES_PER_TOPIC = 20
NUM_TOPICS = 15

# Dataset split — the 50/25/25 spec (train / eval / golden). Golden is the
# human-rated answer key, filled from RATED items only (never padded with
# unrated ones). On both arms the eval slice is EvalInput items — inputs the
# runner executes fresh per run config — so golden, train and val are the
# slices stored as TaskRuns. Both wizard arms split their batch runs the same
# way: golden is capped at GOLDEN_TARGET_FRACTION of the batch
# (select_golden_runs) and the remainder is dealt train:val
# (deal_pool_train_val). The legacy v1 manual flow's single-turn save instead
# takes its reviewed examples as golden (structurally small, no cap needed)
# and splits the generated pool train:eval at 2:1 (the 50:25), minting no val
# items at all. If fewer than the target fraction are rated the answer key is
# simply smaller (warned). One owner so the golden fraction can't drift
# between the splitters.
TRAIN_SPLIT_WEIGHT = 2
EVAL_SPLIT_WEIGHT = 1
GOLDEN_SPLIT_WEIGHT = 1
GOLDEN_TARGET_FRACTION = 0.25

# The non-golden pool's train:val deal, from the agreed
# train/val/test/golden = 40/25/25/10 scheme. Only the train:val ratio of that
# scheme lives here: the test slice is EvalInput items minted separately and
# golden is carved by select_golden_runs, so neither is in this pool to deal.
# Kept apart from the *_SPLIT_WEIGHT constants above, which do the golden/eval
# math and must not move when this ratio does. The same two weights drive the
# dataset-generation allocator in
# app/web_ui/src/lib/utils/eval_generation_splits.ts (TRAIN_SPLIT_WEIGHT /
# VAL_SPLIT_WEIGHT there); the two must move together or generated data and
# wizard-saved data land in the splits at different ratios.
#
# Known limitation of this dealing: val runs share their inputs with the test
# slice (the same driven cases feed both), which is honest for judge
# iteration but leaks eval inputs into any optimizer loop that trains against
# val. Fixing that requires partitioning the input pool before the drive, a
# design change rather than a ratio change.
TRAIN_DEAL_WEIGHT = 40
VAL_DEAL_WEIGHT = 25


def spec_rating_key(spec_name: str) -> str:
    """The requirement_ratings key a spec's golden verdicts are stored under."""
    return f"named::{spec_name}"


def golden_requirement_rating(user_says_meets_spec: bool) -> RequirementRating:
    """The human's pass/fail verdict as the golden requirement rating.

    One constructor for both answer-key writers (single-turn golden runs and
    multi-turn chain leaves) so the rating shape can't drift between them.
    """
    return RequirementRating(
        type=TaskOutputRatingType.pass_fail,
        value=1.0 if user_says_meets_spec else 0.0,
    )


def get_copilot_api_key() -> str:
    """Get the Kiln Copilot API key from config, raising an error if not set."""
    api_key = Config.shared().kiln_copilot_api_key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Kiln Copilot API key not configured. Please connect your API key in settings.",
        )
    return api_key


async def task_capabilities_for_task(
    task: Task,
    run_config_id: str | None = None,
) -> tuple[list[TaskToolInfoApi] | None, list[TaskSkillInfoApi] | None]:
    """The tools and skills one of the task's run configs gives the model.

    `run_config_id` names the config the caller is asking about — the one an
    eval is being written against, say. Without it the task's DEFAULT run
    config is read. Exactly one config is read either way: unioning across
    configs would describe a capability surface no single run of the task
    actually has.

    Names and descriptions only: enough for the copilot prompts to reason about
    what the task can do, without shipping tool parameter schemas or skill
    bodies.

    Returns (None, None) when the capabilities could not be collected (no
    resolvable default run config, or the collection itself failed). Callers
    must keep that distinct from ([], []), which means the task genuinely has
    none.
    """
    started = time.monotonic()
    try:
        # Every tool resolved below shares one session per MCP server instead
        # of dialing the server again per tool, and the scope closes those
        # sessions on the way out.
        async with mcp_session_scope():
            tools, skills = await _collect_task_capabilities(task, run_config_id)
    except HTTPException:
        # A named run config that does not exist is the caller's mistake, not
        # an unreadable capability surface. Degrading it to "uncollected"
        # below would build the prompt against a config the caller never
        # asked for, and say nothing about it.
        raise
    except Exception:
        # Collection reads run configs and skills off disk, so one corrupt or
        # forward-versioned file would otherwise fail a whole spec-building
        # request. Falling back to uncollected keeps the caller working with
        # the prompt it got before capabilities existed.
        logger.warning(
            "Could not collect capabilities for task %s; continuing without them",
            task.id,
            exc_info=True,
        )
        return None, None

    # Resolving a tool can dial its MCP server, so these callers now make
    # network calls they never used to. Logged rather than capped: the cost
    # should be visible before anyone decides what to do about it.
    logger.info(
        "Collected capabilities for task %s in %.0f ms: %s tools, %s skills",
        task.id,
        (time.monotonic() - started) * 1000,
        "uncollected" if tools is None else len(tools),
        "uncollected" if skills is None else len(skills),
    )
    return tools, skills


def _capability_run_config(
    task: Task, run_config_id: str | None
) -> TaskRunConfig | None:
    """The run config whose capabilities answer this request: the one the
    caller named, or the task's default.

    None means the default was asked for and none is resolvable, which the
    caller reports as an uncollected surface. A named config that is not on
    the task raises instead — see the caller.
    """
    if run_config_id is not None:
        # Presence, not truthiness: an empty id is a real (bad) id and must
        # not quietly fall back to the default config, whose tools and skills
        # are not the ones the caller asked about.
        run_config = next(
            (
                candidate
                for candidate in task.run_configs(readonly=True)
                if candidate.id == run_config_id
            ),
            None,
        )
        if run_config is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task run config not found. ID: {run_config_id}",
            )
        return run_config

    if not task.default_run_config_id:
        return None
    return next(
        (
            candidate
            for candidate in task.run_configs(readonly=True)
            if candidate.id == task.default_run_config_id
        ),
        None,
    )


async def _collect_task_capabilities(
    task: Task,
    run_config_id: str | None,
) -> tuple[list[TaskToolInfoApi] | None, list[TaskSkillInfoApi] | None]:
    """Read one run config's capability surface. See the caller for the
    None vs [] contract; failures propagate to it."""
    run_config = _capability_run_config(task, run_config_id)
    if run_config is None:
        return None, None

    properties = run_config.run_config_properties
    if properties.type != "kiln_agent":
        # Other config types (e.g. MCP) carry no tools_config and load no
        # skills, so their capability surface is genuinely empty, not unknown.
        return [], []

    tools_config = as_kiln_agent_run_config(properties).tools_config
    tool_ids = tools_config.tools if tools_config is not None else None

    tools: list[TaskToolInfoApi] = []
    for tool_id in tool_ids or []:
        # Skills ride in the same tools list but are resolved by the adapter,
        # and tool_from_id raises on them. They are collected below instead.
        if tool_id.startswith(SKILL_TOOL_ID_PREFIX):
            continue
        try:
            tool = tool_from_id(tool_id, task)
            tools.append(
                TaskToolInfoApi(
                    name=await tool.name(),
                    description=await tool.description(),
                )
            )
        except Exception:
            # A tool reference that no longer resolves (a removed MCP server, a
            # deleted code tool) must not take down spec building; the rest of
            # the surface is still worth describing.
            logger.warning(
                "Skipping tool %s for task %s: could not resolve it",
                tool_id,
                task.id,
                exc_info=True,
            )

    # Sorted by name so the same task always produces the same payload — the
    # skill loader returns an unordered map.
    skills = [
        TaskSkillInfoApi(name=skill.name, description=skill.description)
        for skill in sorted(
            load_skills_for_task(task, properties).values(), key=lambda s: s.name
        )
    ]
    return tools, skills


def capability_payload_fields(
    task_tools: list[TaskToolInfoApi] | None,
    task_skills: list[TaskSkillInfoApi] | None,
) -> dict[str, Any]:
    """The capability keys to merge into an outgoing copilot payload.

    A None side is omitted entirely rather than sent as null: an absent key is
    how the wire contract says "not collected", and omitting keeps the payload
    identical to what a caller without capabilities has always sent.
    """
    fields: dict[str, Any] = {}
    if task_tools is not None:
        fields["task_tools"] = [tool.model_dump() for tool in task_tools]
    if task_skills is not None:
        fields["task_skills"] = [skill.model_dump() for skill in task_skills]
    return fields


def task_info_payload(task_info: TaskInfoApi) -> dict[str, Any]:
    """target_task_info as the wire wants it, for every copilot call.

    One owner for the capability-key omission so no call site sends an explicit
    null where the contract expects the key to be absent.
    """
    payload = task_info.model_dump(exclude={"task_tools", "task_skills"})
    payload.update(
        capability_payload_fields(task_info.task_tools, task_info.task_skills)
    )
    return payload


async def generate_copilot_examples(
    api_key: str,
    target_task_info: TaskInfoApi,
    sdg_session_config: SyntheticDataGenerationSessionConfigApi,
    spec_definition: str,
) -> list[SampleApi]:
    """Generate examples via the Kiln Copilot API.

    Calls the copilot generate_batch endpoint and returns a flat list of SampleApi objects.
    Raises HTTPException on API errors.

    Args:
        api_key: The Kiln Copilot API key
        target_task_info: Task info for the target task
        sdg_session_config: Session config for synthetic data generation
        spec_definition: The rendered spec definition
    """
    client = get_authenticated_client(api_key)

    generate_input = GenerateBatchInput.from_dict(
        {
            "target_task_info": task_info_payload(target_task_info),
            "sdg_session_config": sdg_session_config.model_dump(),
            "target_specification": spec_definition,
            "num_samples_per_topic": NUM_SAMPLES_PER_TOPIC,
            "num_topics": NUM_TOPICS,
        }
    )

    detailed_result = (
        await generate_batch_v1_copilot_generate_batch_post.asyncio_detailed(
            client=client,
            body=generate_input,
        )
    )
    result = unwrap_response(
        detailed_result,
        none_detail="Failed to generate synthetic data for spec. Please try again.",
    )

    if not isinstance(result, GenerateBatchOutput):
        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    # Convert result to flat list of SampleApi
    examples: list[SampleApi] = []
    data_dict = result.to_dict().get("data_by_topic", {})
    for topic_examples in data_dict.values():
        for ex in topic_examples:
            examples.append(
                SampleApi(
                    input=ex.get("input", ""),
                    output=ex.get("output", ""),
                )
            )

    return examples


T = TypeVar("T")


def split_pool_train_eval(pool: list[T], rng: random.Random) -> tuple[list[T], list[T]]:
    """Divide the non-golden pool into (train, eval) at 2:1 — the 50:25 of
    the split. Golden never comes from this pool: it is the human-reviewed
    examples, selected before this call.

    eval takes the smaller floor share so train is never starved on small
    pools. The pool is shuffled through the injected rng, so the assignment is
    random in production and deterministic under a seeded rng in tests. The
    input list is not mutated.
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)
    eval_count = (
        len(shuffled) * EVAL_SPLIT_WEIGHT // (TRAIN_SPLIT_WEIGHT + EVAL_SPLIT_WEIGHT)
    )
    return shuffled[eval_count:], shuffled[:eval_count]


def warn_if_golden_below_target(golden_count: int, total_count: int) -> None:
    """Warn when the human-rated golden set is under the 25% target.

    Golden is never padded to hit the target (an unrated golden calibrates
    nothing), so a small rated set just yields a smaller answer key — worth a
    warning because the 50/25/25 split can't hold once golden is short.
    """
    if total_count <= 0:
        return
    fraction = golden_count / total_count
    if fraction < GOLDEN_TARGET_FRACTION:
        logger.warning(
            "Golden (human-rated) set is %d of %d examples (%.0f%%), below the "
            "%.0f%% target — the answer key is smaller than the 50/25/25 split "
            "intends; it is not padded with unrated examples.",
            golden_count,
            total_count,
            fraction * 100,
            GOLDEN_TARGET_FRACTION * 100,
        )


def create_task_run_from_sample(
    sample: SampleApi, tag: str, extra_tags: list[str] | None = None
) -> TaskRun:
    """Create a TaskRun from a SampleApi (without parent set)."""
    data_source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "adapter_name": KILN_ADAPTER_NAME,
            "model_name": KILN_COPILOT_MODEL_NAME,
            "model_provider": KILN_COPILOT_MODEL_PROVIDER,
        },
    )

    tags = [tag]
    if extra_tags:
        tags.extend(extra_tags)

    # Access input using model_dump since SampleApi uses alias
    sample_dict = sample.model_dump(by_alias=True)
    return TaskRun(
        input=sample_dict["input"],
        input_source=data_source,
        output=TaskOutput(
            output=sample.output,
            source=data_source,
        ),
        tags=tags,
    )


def create_task_run_from_reviewed(
    example: ReviewedExample,
    tag: str,
    spec_name: str,
    extra_tags: list[str] | None = None,
) -> tuple[TaskRun, str | None]:
    """Create a TaskRun from a reviewed example with rating (without parent set).

    Returns a (TaskRun, feedback_text) tuple. The caller should create Feedback
    and ClaimReview children on the TaskRun after saving it (see
    SingleTurnDataset.save_pending_children).
    """
    data_source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "adapter_name": KILN_ADAPTER_NAME,
            "model_name": KILN_COPILOT_MODEL_NAME,
            "model_provider": KILN_COPILOT_MODEL_PROVIDER,
        },
    )

    tags = [tag]
    if extra_tags:
        tags.extend(extra_tags)

    task_run = TaskRun(
        input=example.input,
        input_source=data_source,
        output=TaskOutput(
            output=example.output,
            source=data_source,
            rating=TaskOutputRating(
                type=TaskOutputRatingType.five_star,
                value=None,  # Actual rating is in requirement_ratings
                requirement_ratings={
                    spec_rating_key(spec_name): golden_requirement_rating(
                        example.user_says_meets_spec
                    )
                },
            ),
        ),
        tags=tags,
    )
    feedback_text = example.feedback if example.feedback else None
    return task_run, feedback_text


class SingleTurnDataset:
    """The dataset one single-turn spec save creates: the golden and train
    TaskRuns (with pending review children — feedback + claim reviews — to
    attach after saving) plus the eval slice as EvalInput items.

    The two stores differ because the slices are used differently: golden and
    train are finished input/output pairs the judge is calibrated on and the
    user can fine-tune from, while the eval slice is inputs only — the runner
    generates the output fresh per run config at eval time.
    """

    def __init__(self) -> None:
        self.task_runs: list[TaskRun] = []
        self.eval_inputs: list[EvalInput] = []
        self._pending_feedback: dict[str, str] = {}
        self._pending_claim_reviews: dict[str, ClaimReviewApi] = {}

    def add_run(
        self,
        task_run: TaskRun,
        feedback_text: str | None = None,
        claim_review: ClaimReviewApi | None = None,
    ) -> None:
        self.task_runs.append(task_run)
        if feedback_text and task_run.id:
            self._pending_feedback[task_run.id] = feedback_text
        if claim_review and task_run.id:
            self._pending_claim_reviews[task_run.id] = claim_review

    def save_pending_children(self, task_run: TaskRun) -> None:
        """Create Feedback / ClaimReview children for a saved TaskRun."""
        if not task_run.id:
            return
        feedback_text = self._pending_feedback.get(task_run.id)
        if feedback_text:
            fb = Feedback(
                feedback=feedback_text,
                source=FeedbackSource.spec_feedback,
                parent=task_run,
            )
            fb.save_to_file()
        claim_review = self._pending_claim_reviews.get(task_run.id)
        if claim_review:
            save_claim_review(task_run, claim_review)


def save_claim_review(task_run: TaskRun, claim_review: ClaimReviewApi) -> ClaimReview:
    """Persist a reviewer's per-claim grades as a ClaimReview child of the run.

    This is the durable half of the answer key: the golden rating records the
    human's verdict, the ClaimReview records WHY (per-claim agree/disagree +
    whys) in the shape judge-prompt refinement consumes.
    """
    # model_dump instead of a field-by-field copy: the API model mirrors the
    # datamodel, so a new field flows through without a silent drop here.
    review = ClaimReview(**claim_review.model_dump(), parent=task_run)
    review.save_to_file()
    return review


def create_single_turn_dataset(
    all_examples: list[SampleApi],
    reviewed_examples: list[ReviewedExample],
    eval_tag: str,
    train_tag: str,
    golden_tag: str,
    spec_name: str,
    rng: random.Random | None = None,
) -> SingleTurnDataset:
    """Build the golden, train, and eval slices of a single-turn save (disjoint).

    - Golden: the human-rated reviewed examples ONLY (the answer key), as
      TaskRuns. Never padded with unrated machine examples — an unrated golden
      calibrates nothing.
    - Train + eval: the unrated machine pool, split 2:1 (the 50:25 of the
      split). Train is stored as TaskRuns; the eval slice is EvalInput items
      carrying the generated input only.

    The three tag sets never overlap. `rng` is injected for deterministic
    tests; None uses a fresh system-seeded Random. `all_examples` is not
    mutated. Returns a SingleTurnDataset with no parents set — the caller
    parents every model, and calls save_pending_children after saving each run.
    """
    rng = rng or random.Random()
    result = SingleTurnDataset()

    # One session tag stamps every item in this batch — runs and eval inputs
    # alike, so a saved spec's whole dataset traces back to one generation.
    session_tag = f"synthetic_session_{rng.randint(0, 999999999999)}"
    extra_tags = [session_tag]

    # Golden = human-rated only.
    for reviewed in reviewed_examples:
        task_run, feedback_text = create_task_run_from_reviewed(
            reviewed, golden_tag, spec_name, extra_tags
        )
        result.add_run(task_run, feedback_text, reviewed.claim_review)

    # The unrated machine pool fills eval + train (disjoint from golden). The
    # single-turn golden set is the reviewed examples only (a small human-rated
    # pool from a separate source), so it is structurally well under the 25%
    # cap — no cap needed here, unlike the multi-turn all-rated case.
    train_examples, eval_examples = split_pool_train_eval(all_examples, rng)
    result.eval_inputs = build_single_turn_eval_inputs(
        # model_dump for the input read: SampleApi keeps it behind an alias.
        [example.model_dump(by_alias=True)["input"] for example in eval_examples],
        eval_tag,
        extra_tags,
    )
    for example in train_examples:
        result.add_run(create_task_run_from_sample(example, train_tag, extra_tags))

    warn_if_golden_below_target(
        len(reviewed_examples), len(reviewed_examples) + len(all_examples)
    )
    return result


def build_single_turn_eval_inputs(
    inputs: list[str],
    eval_tag: str,
    extra_tags: list[str],
) -> list[EvalInput]:
    """Mint one EvalInput per input string — the single-turn eval slice.

    Each carries a generated task INPUT only (structured-task inputs as JSON
    strings), tagged with the eval-slice tag plus provenance (the drive
    batch, or the legacy flow's generation session). No output on purpose:
    the runner produces a fresh output per run config at eval time and
    judges that, so a stored output would only be a misleading artifact of
    the machine that wrote the input.

    Models are built and validated here, unsaved — persistence happens in
    persist_eval_slice inside the save unit-of-work (mirrors the multi-turn
    producer).
    """
    return [
        EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text=input_text)),
            tags=[eval_tag, *extra_tags],
        )
        for input_text in inputs
    ]


def build_single_turn_batch_eval_inputs(
    inputs: list[str],
    batch_tag: str,
    task: Task,
    eval_tag: str,
) -> list[EvalInput]:
    """The wizard single-turn save's eval slice: one EvalInput per generated
    input the pipeline ran, tagged with the eval-slice tag plus the drive
    batch it came from — the single-turn sibling of
    build_multi_turn_eval_inputs, with the same build-unsaved contract
    (persistence happens in persist_eval_slice inside the save
    unit-of-work)."""
    eval_inputs = build_single_turn_eval_inputs(
        inputs, eval_tag, [f"{_TAG_PREFIX_SINGLE_TURN_DRIVE_BATCH}{batch_tag}"]
    )
    for eval_input in eval_inputs:
        eval_input.parent = task
    return eval_inputs


def find_multi_turn_chain_leaves(task: Task, batch_tag: str) -> list[TaskRun]:
    """Return the leaf TaskRuns of all chains tagged with the given batch_tag.

    The multi-turn runner tags only the leaf of each chain with
    "synthetic_user_batch:{batch_tag}". Walking parent_task_run_id from the
    leaf reconstructs the full conversation if a caller needs it; for eval
    purposes the leaf alone is enough because its `.trace` field already
    holds the cumulative OpenAI-format conversation.
    """
    target_tag = f"{_TAG_PREFIX_SU_BATCH}{batch_tag}"
    return [run for run in task.runs() if target_tag in (run.tags or [])]


def delete_multi_turn_batch_chains(task: Task, batch_tag: str) -> int:
    """Delete every chain TaskRun of an abandoned synthetic-user batch.

    Re-driving a batch mints a new batch_tag, which would orphan the previous
    batch's chains on disk forever — the caller passes the superseded tag and
    this removes those chains (every run from leaf to root) once the replacing
    drive has produced results, so a failed re-drive never destroys the only
    batch on disk. Returns the number of TaskRuns deleted.

    Safety: a chain is only deleted when its leaf carries EXACTLY the
    runner's own tags, no rating, and no descendants. Any extra tag, a
    rating, or a child run means some other flow (an eval save, a manual
    rating, a continued conversation) claimed the chain — it is no longer an
    abandoned drive artifact, so it is left alone. The exact-set match fails
    CLOSED: if the runner's tag scheme ever grows, batches get skipped
    (orphaned) rather than risking deletion of claimed chains.
    """
    # include_intermediate_runs: the ancestor walk needs the complete on-disk
    # set, not the default leaves-only view. One corpus load serves the leaf
    # scan, the descendant check, and the ancestor lookups.
    all_runs = task.runs(include_intermediate_runs=True)
    runs_by_id = {str(run.id): run for run in all_runs}
    parent_ids = {
        str(run.parent_task_run_id)
        for run in all_runs
        if run.parent_task_run_id is not None
    }
    children_by_parent: dict[str, set[str]] = {}
    for run in all_runs:
        if run.parent_task_run_id is not None:
            children_by_parent.setdefault(str(run.parent_task_run_id), set()).add(
                str(run.id)
            )
    target_tag = f"{_TAG_PREFIX_SU_BATCH}{batch_tag}"
    runner_tags = {_TAG_SU_CASE, target_tag}
    deleted = 0
    for leaf in (run for run in all_runs if target_tag in (run.tags or [])):
        if (
            set(leaf.tags or []) != runner_tags
            or leaf.output.rating is not None
            or str(leaf.id) in parent_ids
        ):
            logger.info(
                "Skipping delete of chain leaf %s: claimed by another flow",
                leaf.id,
            )
            continue
        chain: list[TaskRun] = [leaf]
        current = leaf
        while current.parent_task_run_id is not None:
            parent = runs_by_id.get(str(current.parent_task_run_id))
            if parent is None:
                break
            chain.append(parent)
            current = parent
        # A mid-chain turn can parent runs OUTSIDE this chain (a conversation
        # continued from an earlier turn). Deleting it would dangle that
        # fork's parent_task_run_id, so the whole chain is left alone.
        chain_ids = {str(run.id) for run in chain}
        if any(
            not children_by_parent.get(str(run.id), set()) <= chain_ids for run in chain
        ):
            logger.info(
                "Skipping delete of chain leaf %s: a conversation outside "
                "the batch forks from this chain",
                leaf.id,
            )
            continue
        for run in chain:
            run.delete()
            deleted += 1
    return deleted


def single_turn_drive_tags(batch_tag: str) -> list[str]:
    """The single-turn pipeline's discovery tags for one batch — the one
    producer of the scheme, shared by the adapter's save-time default_tags
    and the explicit tagger so the two paths can't drift."""
    return sorted(
        [_TAG_SINGLE_TURN_DRIVE, f"{_TAG_PREFIX_SINGLE_TURN_DRIVE_BATCH}{batch_tag}"]
    )


def tag_single_turn_drive_run(run: TaskRun, batch_tag: str) -> None:
    """Ensure a driven run carries the pipeline's discovery tags and persist.

    Normally a no-op belt-and-braces pass (the adapter's default_tags land
    the same tags in the run's own save); it exists so a run persisted by an
    adapter without them can never slip through untagged. Tags are
    deduplicated (treated as a set then sorted) so re-tagging is idempotent.
    A save_to_file exception surfaces to the caller (which converts it to a
    case failure) — an untagged run is invisible to save and cleanup, so
    silence here would strand it.
    """
    tags = set(run.tags or []) | set(single_turn_drive_tags(batch_tag))
    if sorted(tags) == (run.tags or []):
        return
    run.tags = sorted(tags)
    run.save_to_file()


def find_single_turn_batch_runs(task: Task, batch_tag: str) -> list[TaskRun]:
    """Return the runs of one single-turn pipeline batch, by its batch tag."""
    target_tag = f"{_TAG_PREFIX_SINGLE_TURN_DRIVE_BATCH}{batch_tag}"
    return [run for run in task.runs() if target_tag in (run.tags or [])]


def delete_single_turn_batch_runs(task: Task, batch_tag: str) -> int:
    """Delete every run of an abandoned single-turn pipeline batch.

    Re-running a batch mints a new batch_tag, which would orphan the previous
    batch's runs on disk forever — the caller passes the superseded tag once
    the replacing run has produced results, so a failed re-run never destroys
    the only batch on disk. Returns the number of TaskRuns deleted.

    Safety mirrors delete_multi_turn_batch_chains: a run is only deleted when
    it carries EXACTLY the pipeline's own tags and no rating. Any extra tag
    or a rating means another flow (an eval save, a manual rating) claimed
    it — no longer an abandoned drive artifact, so it is left alone. The
    exact-set match fails CLOSED: if the tag scheme ever grows, batches get
    skipped (orphaned) rather than risking deletion of claimed runs. No
    descendant check is needed — single-turn tasks reject chained runs at the
    datamodel level.
    """
    target_tag = f"{_TAG_PREFIX_SINGLE_TURN_DRIVE_BATCH}{batch_tag}"
    pipeline_tags = {_TAG_SINGLE_TURN_DRIVE, target_tag}
    deleted = 0
    for run in find_single_turn_batch_runs(task, batch_tag):
        if set(run.tags or []) != pipeline_tags or run.output.rating is not None:
            logger.info(
                "Skipping delete of single-turn run %s: claimed by another flow",
                run.id,
            )
            continue
        run.delete()
        deleted += 1
    return deleted


def split_and_tag_batch_runs(
    leaves: list[TaskRun],
    reviewed_leaf_ids: set[str],
    train_tag: str,
    golden_tag: str,
    val_tag: str,
    rng: random.Random | None = None,
    tagged_out: list[tuple[TaskRun, set[str]]] | None = None,
) -> None:
    """Assign each batch run to exactly ONE split (golden XOR train XOR val).

    Both arms' save writer: `leaves` are the multi-turn chain leaves or the
    single-turn pipeline's batch-tagged runs. Golden = the human-rated runs
    (the answer key), capped at the target fraction; everything left over is
    dealt train:val. The runs carry no eval slice — the eval set is EvalInput
    items minted separately (from the driven cases or the generated inputs)
    and re-run fresh at eval time, so reusing a golden run's input there is
    not circular: golden validates the judge on the STORED result while the
    eval set scores NEW ones.

    `rng` is injected for deterministic tests. If `tagged_out` is provided,
    each run actually mutated is appended as `(run, {tag_added})` so the
    caller can reverse the mutation on failure via
    `untag_batch_runs_for_eval` without disturbing pre-existing tags.
    Mutates each run in place and persists via save_to_file.
    """
    rng = rng or random.Random()
    golden, pool = select_golden_runs(leaves, reviewed_leaf_ids, rng)
    train, val = deal_pool_train_val(pool, rng)

    tag_batch_runs(golden, golden_tag, tagged_out)
    tag_batch_runs(train, train_tag, tagged_out)
    tag_batch_runs(val, val_tag, tagged_out)

    warn_if_golden_below_target(len(golden), len(leaves))


def deal_pool_train_val(pool: list[T], rng: random.Random) -> tuple[list[T], list[T]]:
    """Deal the non-golden pool into (train, val) at TRAIN:VAL weights.

    The pool must be re-shuffled here even though select_golden_runs shuffles:
    it shuffles only the RATED runs and returns rated-leftovers ahead of the
    unrated ones in disk order, so dealing that order by prefix would send
    every over-cap rated run to the same bucket every time. Shuffling through
    the injected rng keeps the deal random in production and reproducible
    under a seeded rng. The input list is not mutated.

    Sizes are apportioned by largest remainder so no run is dropped: both
    shares are floored, and the at-most-one leftover seat goes to the larger
    fractional remainder. The two remainders always sum to 0 or to
    TRAIN + VAL, because the exact shares sum to the pool size; a leftover
    seat exists exactly in the second case, where both are nonzero and sum to
    an odd 65. So whenever there is a seat to award the remainders cannot be
    equal, and the deal has no arbitrary tie-break to get wrong.
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)
    size = len(shuffled)
    total_weight = TRAIN_DEAL_WEIGHT + VAL_DEAL_WEIGHT
    train_count = size * TRAIN_DEAL_WEIGHT // total_weight
    val_count = size * VAL_DEAL_WEIGHT // total_weight
    # The two floors leave at most one seat unassigned; largest remainder
    # gives it to whichever bucket was rounded down harder. Val takes the
    # rest of the pool, so only train's count has to move.
    if train_count + val_count < size and (
        size * TRAIN_DEAL_WEIGHT % total_weight > size * VAL_DEAL_WEIGHT % total_weight
    ):
        train_count += 1
    return shuffled[:train_count], shuffled[train_count:]


def select_golden_runs(
    leaves: list[TaskRun],
    reviewed_leaf_ids: set[str],
    rng: random.Random,
) -> tuple[list[TaskRun], list[TaskRun]]:
    """Carve the golden answer-key slice off the batch runs.

    Golden is up to GOLDEN_TARGET_FRACTION of the runs, drawn from RATED
    runs only (the answer key is human-rated by definition). Under the
    pooled stratified review both arms rate ~25% of the batch, so golden is
    normally every reviewed run; a reviewer who grades extra runs beyond the
    cap sends the extras back into the pool with their ratings kept, where
    they can land in either dealt slice. Returns (golden, remaining):
    remaining holds the rated runs beyond the cap plus the unrated runs — the
    pool that deal_pool_train_val splits train:val. Only the golden slice is
    the answer key the judge is calibrated against.
    """
    golden_target = (
        len(leaves)
        * GOLDEN_SPLIT_WEIGHT
        // (TRAIN_SPLIT_WEIGHT + EVAL_SPLIT_WEIGHT + GOLDEN_SPLIT_WEIGHT)
    )
    rated = [leaf for leaf in leaves if leaf.id in reviewed_leaf_ids]
    unrated = [leaf for leaf in leaves if leaf.id not in reviewed_leaf_ids]
    rng.shuffle(rated)
    golden = rated[:golden_target]
    remaining = rated[golden_target:] + unrated
    return golden, remaining


def tag_batch_runs(
    leaves: list[TaskRun],
    tag: str,
    tagged_out: list[tuple[TaskRun, set[str]]] | None = None,
) -> None:
    """Add one split tag to each run, recording the addition for rollback."""
    for leaf in leaves:
        current = set(leaf.tags or [])
        if tag in current:
            continue
        leaf.tags = sorted(current | {tag})
        leaf.save_to_file()
        if tagged_out is not None:
            tagged_out.append((leaf, {tag}))


def build_multi_turn_eval_inputs(
    cases: list[DrivenSyntheticCaseApi],
    batch_tag: str,
    task: Task,
    eval_tag: str,
    drive_config: MultiTurnDriveConfig,
) -> list[EvalInput]:
    """Mint one EvalInput per driven case — the multi-turn eval slice.

    Each carries the case's seed message, the parsed synthetic-user persona
    (the structured submodel; the XML blob never persists), and the drive
    settings the batch's conversations ran with — stamped per item so every
    item is a self-contained replication recipe for eval-time re-drives.
    Tagged with the eval-slice tag and its provenance: the synthetic-user
    batch the case was driven in and, when known, the batch-plan scenario it
    came from.

    Models are built and validated here, unsaved — persistence happens in
    persist_eval_slice inside the save unit-of-work. Raises
    HTTPException(422) when a case's persona blob doesn't parse, so a
    malformed request fails before anything is written.
    """
    eval_inputs: list[EvalInput] = []
    for position, case in enumerate(cases):
        try:
            info = parse_synthetic_user_info(case.synthetic_user_info)
        except SyntheticUserInfoParseError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Case {position}: invalid synthetic_user_info: {e}",
            )
        tags = [eval_tag, f"{_TAG_PREFIX_SU_BATCH}{batch_tag}"]
        if case.scenario_index is not None:
            tags.append(f"scenario:{case.scenario_index}")
        eval_inputs.append(
            EvalInput(
                parent=task,
                data=MultiTurnSyntheticEvalInputData(
                    first_message=UserMessage(text=case.seed_prompt),
                    synthetic_user_info=info,
                    drive_config=drive_config,
                ),
                tags=tags,
            )
        )
    return eval_inputs


def persist_eval_slice(
    eval_inputs: list[EvalInput],
    saved_out: list,
) -> None:
    """Materialize an eval slice by persisting its EvalInput items.

    Shared by both arms: the items differ (a driven case's seed + persona vs a
    generated single-turn input) but the persistence and rollback contract is
    the same. Each item is appended to `saved_out` the moment it hits disk so
    a failed save rolls it back with the other created models.
    """
    for eval_input in eval_inputs:
        eval_input.save_to_file()
        saved_out.append(eval_input)


def untag_batch_runs_for_eval(
    tagged_leaves: list[tuple[TaskRun, set[str]]],
) -> None:
    """Reverse the tagging done by split_and_tag_batch_runs.

    Removes only the tags that THIS save added (passed in via `tagged_out`),
    so pre-existing tags on the run are preserved. Best-effort: a per-run
    save failure is logged and the loop continues — the original save error
    that triggered cleanup is the one the user needs to see.
    """
    for leaf, added_tags in tagged_leaves:
        try:
            leaf.tags = sorted(set(leaf.tags or []) - added_tags)
            leaf.save_to_file()
        except Exception:
            logger.exception(f"Failed to untag leaf {leaf.id} during cleanup")


def rate_reviewed_batch_runs(
    leaves: list[TaskRun],
    reviewed_chains: list[ReviewedChainApi],
    spec_name: str,
    rated_out: list[
        tuple[TaskRun, TaskOutputRating | None, list[Feedback | ClaimReview]]
    ]
    | None = None,
) -> None:
    """Write the human's review verdicts onto the batch runs, both arms.

    Each reviewed run (a chain leaf on multi-turn, the run itself on
    single-turn) gets a golden RequirementRating (pass_fail under
    `named::{spec_name}`), plus a Feedback for the disagree-why text and a
    ClaimReview child carrying the per-claim grades — ONE answer-key shape
    across the arms.

    If `rated_out` is provided, each mutated run is appended as
    `(run, rating_before_this_call, children_added)` so a failed save can
    be reversed via `unrate_reviewed_batch_runs`.

    Raises HTTPException(404) when a review references a run id not in
    `leaves` — the review must describe the batch being saved.
    """
    leaves_by_id = {leaf.id: leaf for leaf in leaves if leaf.id}
    rating_key = spec_rating_key(spec_name)

    for reviewed in reviewed_chains:
        leaf = leaves_by_id.get(reviewed.leaf_run_id)
        if leaf is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Reviewed chain leaf '{reviewed.leaf_run_id}' is not part "
                    "of this batch."
                ),
            )

        prior_rating = (
            leaf.output.rating.model_copy(deep=True) if leaf.output.rating else None
        )
        rating = leaf.output.rating or TaskOutputRating(
            type=TaskOutputRatingType.five_star,
            value=None,  # Actual rating is in requirement_ratings
        )
        rating.requirement_ratings[rating_key] = golden_requirement_rating(
            reviewed.user_says_meets_spec
        )
        leaf.output.rating = rating
        leaf.save_to_file()

        # Record for rollback the moment the leaf is mutated on disk;
        # added_children is filled in place below, so a failure while saving
        # a child still rolls back everything already persisted.
        added_children: list[Feedback | ClaimReview] = []
        if rated_out is not None:
            rated_out.append((leaf, prior_rating, added_children))

        if reviewed.feedback:
            fb = Feedback(
                feedback=reviewed.feedback,
                source=FeedbackSource.spec_feedback,
                parent=leaf,
            )
            fb.save_to_file()
            added_children.append(fb)
        if reviewed.claim_review:
            added_children.append(save_claim_review(leaf, reviewed.claim_review))


def unrate_reviewed_batch_runs(
    rated_leaves: list[
        tuple[TaskRun, TaskOutputRating | None, list[Feedback | ClaimReview]]
    ],
) -> None:
    """Reverse the mutations done by rate_reviewed_batch_runs.

    Restores each run's prior rating and deletes the Feedback/ClaimReview
    children this save added. Best-effort like the untag path: per-run
    failures are logged and the loop continues so the original error stays
    visible.
    """
    for leaf, prior_rating, added_children in rated_leaves:
        try:
            leaf.output.rating = prior_rating
            leaf.save_to_file()
            for child in added_children:
                child.delete()
        except Exception:
            logger.exception(f"Failed to unrate leaf {leaf.id} during cleanup")
