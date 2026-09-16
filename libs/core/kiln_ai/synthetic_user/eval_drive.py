"""Eval-time re-drive of one multi-turn synthetic case.

The eval runner calls this to regenerate a conversation per run config: the
agent under test comes from the run config being evaluated; the synthetic
user (customer) comes from the item's drive config, held constant so a
comparison varies only the agent.

Nothing here touches disk — the drive runs with allow_saving=False, and the
eval runner persists the finished conversation itself as one standalone
TaskRun, after stamping it as an eval trace. Conversation continuity rides
`prior_trace`, so no parent_task_run chaining (which requires persisted
parents) is involved.
"""

from kiln_ai.adapters.adapter_registry import adapter_for_task
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig, SkillsDict
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_output import DataSource, DataSourceType
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.synthetic_user.drive_loop import DriveCaseResult, drive_case
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.models import (
    SyntheticUserDriverConfig,
    SyntheticUserInfo,
)
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam

# Identifies this code path in `input_source.properties.adapter_name` — the
# runs are in-memory only, but anything inspecting one should still see who
# authored the user side.
_EVAL_DRIVER_ADAPTER_NAME = "kiln_synthetic_user_eval_driver"


async def drive_case_for_eval(
    *,
    seed_prompt: str,
    synthetic_user_info: SyntheticUserInfo,
    target_task: Task,
    target_run_config: KilnAgentRunConfigProperties,
    su_driver_config: SyntheticUserDriverConfig,
    turns: int,
    skills: SkillsDict,
) -> DriveCaseResult:
    """Drive one case in memory and return the full DriveCaseResult (never saved).

    The result's leaf (`chain[-1]`) has `.trace` holding the full cumulative
    conversation and its id is None (nothing touches disk). The result also
    carries `su_usage` — the synthetic user model's spend, which surfaces
    nowhere else since SU turns aren't persisted. `skills` must be preloaded
    by the caller — the adapter raises on skill tools with no injected dict.
    """
    su_driver = SyntheticUserDriver(synthetic_user_info, su_driver_config)
    adapter = adapter_for_task(
        target_task,
        target_run_config,
        base_adapter_config=AdapterConfig(allow_saving=False, skills=skills),
    )
    input_source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": su_driver_config.model_name,
            "model_provider": su_driver_config.model_provider_name.value,
            "adapter_name": _EVAL_DRIVER_ADAPTER_NAME,
        },
    )

    async def _invoker(
        *,
        input: str,
        prior_trace: list[ChatCompletionMessageParam] | None,
        parent_task_run: TaskRun | None,
    ) -> TaskRun:
        # Chaining is deliberately dropped: parent runs are unsaved (id None)
        # and the conversation already continues through prior_trace.
        _ = parent_task_run
        return await adapter.invoke(
            input=input,
            input_source=input_source,
            prior_trace=prior_trace,
        )

    return await drive_case(
        seed_prompt=seed_prompt,
        target_invoker=_invoker,
        su_driver=su_driver,
        turns=turns,
    )
