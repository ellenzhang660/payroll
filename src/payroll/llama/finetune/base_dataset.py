from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from gluonts.dataset.pandas import PandasDataset


@dataclass
class ClassAttributes:
    prediction_length: int
    context_length: int
    available_target_columns: list[str]  # as LagLlama is univariate, can only finetune on one column at a time
    freq: str


@dataclass
class FinetuneDataset(ClassAttributes):
    train_dataset: PandasDataset
    val_dataset: PandasDataset


class BaseFinetuningDataset(ABC):
    def __init__(self):
        self.attributes = self._init_attributes()
        self._log_preprocess()
        self.df = self.preprocess()

    @abstractmethod
    def _init_attributes(self) -> ClassAttributes:
        """Initializes class attributes"""
        pass

    @property
    def available_target_columns(self) -> list[str]:
        return self.attributes.available_target_columns

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Processes the dataframe (e.g. set datatime as idx, clean or remove rows, etc)"""
        pass

    @abstractmethod
    def _log_preprocess(self):
        """Internal method to log preprocessing steps"""
        pass

    @abstractmethod
    def __getitem__(self, target_column: str) -> FinetuneDataset:
        """Dataset for a specific target_column of the dataset"""
        pass
