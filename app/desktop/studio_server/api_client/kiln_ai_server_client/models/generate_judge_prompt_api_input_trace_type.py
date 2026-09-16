from enum import Enum


class GenerateJudgePromptApiInputTraceType(str, Enum):
    MULTI_TURN = "multi_turn"
    SINGLE_TURN = "single_turn"

    def __str__(self) -> str:
        return str(self.value)
