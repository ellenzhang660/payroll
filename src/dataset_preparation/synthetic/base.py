from abc import ABC, abstractmethod

import pandas as pd

from src.dataset.base import TimeSeriesData
from src.dataset_preparation.base import GenerateFakeData

class SyntheticGenerator(GenerateFakeData):
    """
    Absract class for generating fake data
    """

    #################### Creator operation ####################
    def __init__(self, dataset: TimeSeriesData):
        super().__init__(dataset=dataset)

    #################### Observer operations ####################
    def generate_fake_data(self) -> dict[str, list[pd.Series]]:
        

    #################### Representation ####################
    def __str__(self) -> str:
        """Pretty-print all observer information about the dataset."""
        return ""

    def __repr__(self) -> str:
        return f"GenerateFakeData({self.base})"
