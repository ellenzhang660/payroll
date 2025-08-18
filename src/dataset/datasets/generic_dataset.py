from pathlib import Path

import pandas as pd

from src.dataset.base import TimeSeriesData


class GenericDataset(TimeSeriesData):
    """
    Absract class for generic time series dataset
    """

    home = Path(__file__).parent
    url = f"{home}/database/ts_long.csv"

    #################### Creator operation ####################
    def __init__(self):
        super().__init__(url=self.url)
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self._clean()
        self.id_var = "item_id"
        self.unique_samples = self.df[self.id_var].unique().tolist()  # type: ignore

    def _clean(self):
        # Set numerical columns as float32
        for col in self.df.columns:
            # Check if column is not of string type
            if self.df[col].dtype != "object" and pd.api.types.is_string_dtype(self.df[col]) == False:
                self.df[col] = self.df[col].astype("float32")

    #################### Observer operations ####################
    def how_many_unique_samples(self) -> int:
        return len(self.unique_samples)

    def what_variates(self) -> set[str]:
        return set(["target"])

    def length_of_time_series(self) -> int:
        return self.df.groupby(self.id_var).size()

    def frequency_of_time_series(self) -> str:
        return "H"

    def how_much_memory_in_MB(self) -> float:
        # Get memory usage in bytes (sum all columns)
        memory_bytes = self.df.memory_usage(deep=True).sum()

        # Convert to megabytes
        memory_mb = memory_bytes / (1024**2)
        return memory_mb
    
    def __getitem__(self, idx: int) -> dict[str, pd.Series]:
        id = self.unique_samples[idx]
        df_person = self.df[self.df[self.id_var] == id].copy()  # type: ignore
        variates: dict[str, pd.Series] = {}
        for variable in self.what_variates():
            df_person_variable = df_person["target"]
            variates[variable] = df_person_variable
        return variates
