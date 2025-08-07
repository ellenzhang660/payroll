from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from gluonts.dataset.pandas import PandasDataset


@dataclass
class ClassAttributes:
    prediction_length: int
    context_length: int
    target_column: str
    freq: str


@dataclass
class FinetuneDataset(ClassAttributes):
    dataset: PandasDataset


class BasePreprocessingInterface(ABC):
    def __init__(self):
        self.attributes = self._init_attributes()
        self._log_preprocess()
        self.df = self.preprocess()

    @abstractmethod
    def _init_attributes(self) -> ClassAttributes:
        """Initializes class attributes"""
        pass

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Processes the dataframe (e.g. set datatime as idx, clean or remove rows, etc)"""
        pass

    @abstractmethod
    def _log_preprocess(self):
        """Internal method to log preprocessing steps"""
        pass

    @abstractmethod
    def __getitem__(self, key: Literal["train", "val", "test"]) -> FinetuneDataset:
        """Access specific split of the dataset"""
        pass
