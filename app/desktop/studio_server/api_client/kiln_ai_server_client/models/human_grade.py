from enum import Enum


class HumanGrade(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"

    def __str__(self) -> str:
        return str(self.value)
