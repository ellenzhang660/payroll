from abc import ABC, abstractmethod
import pandas as pd
from src.dataset.base import TimeSeriesData

class GenerateFakeData(ABC):
    """
    Absract class for generating fake data
    """
    #################### Creator operation ####################
    def __init__(self, dataset: TimeSeriesData):
        """
        Narrow specs for now
        Input: a csv url
        """
        self.base = dataset  # store as private

    #################### Observer operations ####################
    @abstractmethod
    def generate_fake_data(self, time_series: pd.Series) -> dict[str, list[pd.Series]]:
        """
        Given a timeseries long pandas series, generates various augmentations/synthetic
        Returns
        -------
            dicitonary mapping augmentation description to one or more augmented series
        """

    #################### Representation ####################
    def __str__(self) -> str:
        """Pretty-print all observer information about the dataset."""
        return ""

    def __repr__(self) -> str:
        return f"GenerateFakeData({self.base})"
