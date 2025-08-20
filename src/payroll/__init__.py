from dataclasses import dataclass
from enum import Enum


class Architecture(str, Enum):
    LLM = "LLM"
    FOUNDATION = "Foundation"


@dataclass
class ModelInfo:
    name: str
    is_multivariate: bool
    context_length: int
    supports_zero_shot: bool
    supports_finetuning: bool
    architecture: Architecture

    @property
    def type(self) -> str:
        return "multivariate" if self.is_multivariate else "univariate"

    def __str__(self) -> str:
        return (
            f"ModelInfo(\n"
            f"  Name: {self.name}\n"
            f"  Type: {self.type}\n"
            f"  Context Length: {self.context_length}\n"
            f"  Zero-Shot: {'Yes' if self.supports_zero_shot else 'No'}\n"
            f"  Fine-tuning: {'Yes' if self.supports_finetuning else 'No'}\n"
            f"  Architecture: {self.architecture.value}\n"
            f")"
        )


models: list[ModelInfo] = [
    ModelInfo(
        name="lag-llama",
        is_multivariate=False,
        context_length=-1,
        supports_zero_shot=True,
        supports_finetuning=True,
        architecture=Architecture.FOUNDATION,
    ),
    ModelInfo(
        name="TimeLLM",
        is_multivariate=False,
        context_length=-1,
        supports_zero_shot=True,
        supports_finetuning=True,
        architecture=Architecture.LLM,
    ),
]
