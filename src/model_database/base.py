from abc import ABC, abstractmethod

from src.model_database import ModelInfo


class ModelChoice(ABC):
    """
    Absract class for generic time series dataset
    """

    #################### Creator operation ####################
    def __init__(self, model: ModelInfo):
        """
        Narrow specs for now
        Input: a csv url
        """
        self.model = model  # store as private

    #################### Observer operations ####################
    @abstractmethod
    def zero_shot(self, data, context_length, prediciton_length) -> set[str]:
        """Returns the variates in the dataset"""

    #################### Mutator operations ####################
    @abstractmethod
    def finetune(self) -> int:
        """Returns the number of samples of the time series data, assuming that each person + variable combination is the same"""

    #################### Representation ####################
    def __str__(self) -> str:
        """Pretty-print all observer information about the dataset."""
        return f"{self.model}:\n"

    def __repr__(self) -> str:
        return f"ModelChoice({self.model})"
