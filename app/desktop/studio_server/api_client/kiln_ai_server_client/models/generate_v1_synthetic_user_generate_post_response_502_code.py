from enum import Enum


class GenerateV1SyntheticUserGeneratePostResponse502Code(str, Enum):
    LLM_UNAVAILABLE = "llm_unavailable"
    UPSTREAM_INVALID_OUTPUT = "upstream_invalid_output"

    def __str__(self) -> str:
        return str(self.value)
