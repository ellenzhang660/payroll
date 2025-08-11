from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from gluonts.dataset.pandas import PandasDataset  # type: ignore


@dataclass(frozen=True)
class ClassAttributes:
    prediction_length: int
    context_length: int
    available_target_columns: Tuple[str, ...]
    freq: str


@dataclass(frozen=True)
class FinetuneDataset(ClassAttributes):
    train_dataset: PandasDataset
    val_dataset: PandasDataset


class BaseFinetuningDataset(ABC):
    """
    Base class for lagllama finetune dataset
    Subclasses implement deails
    """

    def __init__(self):
        self.attributes = self._init_attributes()
        self._log_preprocess()
        self.df = self.preprocess()

    @abstractmethod
    def _init_attributes(self) -> ClassAttributes:
        """Initializes class attributes"""
        pass

    @property
    def available_target_columns(self) -> Tuple[str, ...]:
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
