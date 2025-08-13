from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from gluonts.dataset.pandas import PandasDataset  # type: ignore
from torch.utils.data import Dataset

from src.payroll.llama.finetune.base_dataset import ClassAttributes


@dataclass(frozen=True)
class TestDatasetAttributes(ClassAttributes):
    id_var: str


class BaseTestDataset(Dataset):  # type: ignore
    """
    Base class for lagllama finetune dataset
    Subclasses implement deails
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.attributes = self._init_attributes()
        self.unique_ids = self.df[self.attributes.id_var].unique().tolist()  # type: ignore
        assert (
            target_column in self.attributes.available_target_columns
        ), f"target column {target_column} not found, pleae choose from {self.attributes.available_target_columns}"
        self._log_preprocess()
        self.df = self.preprocess()

    @abstractmethod
    def _init_attributes(self) -> TestDatasetAttributes:
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
    def _reformat(self, df: pd.DataFrame) -> PandasDataset:
        """Internal method to reformat df into a PandasDataset"""
        pass

    def __len__(self):
        return len(self.unique_ids)

    def get_id(self, idx: int):
        return self.unique_ids[idx]

    def __getitem__(self, idx: int) -> PandasDataset:
        id = self.unique_ids[idx]
        df_person = self.df[self.df[self.attributes.id_var] == id].copy()  # type: ignore
        dataset = self._reformat(df=df_person)  # type: ignore
        return dataset
