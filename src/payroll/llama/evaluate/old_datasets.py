from typing import Literal

import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from torch.utils.data import Dataset

from src.dataset.austria_payroll.dataset import PayrollDataset


def init_dataset(dataset: Literal["payroll", "generic", "retail", "weather"]) -> Dataset:  # type: ignore
    if dataset == "payroll":
        return LlamaDataset(target_column="Gross total")
        # return LlamaDataset(target_column="** Payout **")
        # return LlamaDataset(target_column="Gross pay", mode = mode)
    elif dataset == "generic":
        return GenericDataset()
    elif dataset == "retail":
        return RetailDataset()
    elif dataset == "weather":
        return WeatherDataset()
    else:
        raise ValueError


class LlamaDataset(PayrollDataset):
    def __init__(self, target_column: str):
        self.target_column = target_column
        super().__init__()
        self.prediction_length = 6
        self.context_length = 32
        self._assert_output()
        assert self.target_column in self.available_keys

    def _assert_output(self):
        result = self.__getitem__(0)
        print(result)

    def reformat(self, df: pd.DataFrame) -> PandasDataset:
        """
        Input: multivariate dataframe
        rows = Different time series
        time cols are the relevant ones we want to extract

        Returns a PandasDataset compatible with llama
        """
        # Melt the time columns to long format
        df_long = df.melt(
            id_vars=["ID", "Description"], value_vars=self.time_cols, var_name="Month", value_name="Amount"
        )

        # Pivot to desired format: months as index, descriptions as columns
        df_pivoted = df_long.pivot_table(
            index=["ID", "Month"],  # include ID in the index too
            columns="Description",
            values="Amount",
            aggfunc="first",
        ).fillna(0.0)

        # Reset 'ID' from index to a column but keep 'Month' as the index
        df_pivoted = df_pivoted.reset_index()

        # Optional: Convert Month to datetime
        df_pivoted["Month"] = pd.to_datetime(df_pivoted["Month"], format="%B_%Y")
        df_pivoted = df_pivoted.sort_values("Month")

        # Now, df should haev month as the index, and columns for each of the 42? payroll descriptions, and an ID column

        dataset = PandasDataset.from_long_dataframe(
            df_pivoted, target=self.target_column, timestamp="Month", item_id="ID", freq=self.freq
        )

        return dataset

    def __getitem__(self, idx: int) -> PandasDataset:
        results = super().__getitem__(idx)
        dataset = self.reformat(df=results["df"])

        return dataset


class GenericDataset(Dataset):
    def __init__(self):
        url = (
            "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
            "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
        )
        self.df = pd.read_csv(url, index_col=0, parse_dates=True)

        # Set numerical columns as float32
        for col in self.df.columns:
            # Check if column is not of string type
            if self.df[col].dtype != "object" and pd.api.types.is_string_dtype(self.df[col]) == False:
                self.df[col] = self.df[col].astype("float32")

        self.unique_ids = self.df["item_id"].unique().tolist()
        self.id_var = "item_id"
        self.prediction_length = 24
        self.context_length = 32
        self._assert_output()

    def _assert_output(self):
        result = self.__getitem__(0)
        print(result)

    def __len__(self):
        return len(self.unique_ids)

    def reformat(self, df: pd.DataFrame) -> PandasDataset:
        # Create the Pandas
        dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
        return dataset

    def __getitem__(self, idx: int) -> PandasDataset:
        id = self.unique_ids[idx]
        df_person = self.df[self.df[self.id_var] == id].copy()
        dataset = self.reformat(df=df_person)
        return dataset


class RetailDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(
            "https://gist.githubusercontent.com/dannymorris/ac176586e0236bd9278e9c81e06851a8/raw/54fd7c7520702d3dd7d4bd59c9dfbed5385af438/aus_retail.csv"
        )
        self.df = self.df.set_index("Month")

        self.unique_ids = self.df["item_id"].unique().tolist()
        self.id_var = "item_id"
        self.prediction_length = 12
        self.context_length = 32
        self.freq = "1ME"
        self._assert_output()

    def _assert_output(self):
        result = self.__getitem__(0)
        print(result)

    def __len__(self):
        return len(self.unique_ids)

    def reformat(self, df: pd.DataFrame) -> ListDataset:
        # Create the Pandas
        test_data = [{"start": df.index[0], "target": df[i].values} for i in df.columns]
        dataset = ListDataset(data_iter=test_data, freq=self.freq)
        return dataset

    def __getitem__(self, idx: int):
        id = self.unique_ids[idx]
        df_person = self.df[self.df[self.id_var] == id].copy()
        dataset = self.reformat(df=df_person)
        return dataset


class WeatherDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prediction_length = 7
        self.context_length = 21

        self.df = pd.read_csv(
            "https://raw.githubusercontent.com/joshuajnoble/Lag-Llama-Tutorial/main/daily-min-temperatures.csv"
        )
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.set_index("Date")
        self.df = self.df.resample("D").sum().interpolate("linear")
        for col in self.df.columns:
            # Check if column is not of string type
            if self.df[col].dtype != "object" and pd.api.types.is_string_dtype(self.df[col]) == False:
                self.df[col] = self.df[col].astype("float32")

        self.freq = "1d"
        self.target = "Temp"
        train_end = round(len(self.df) * 0.7)

        valid_end = round(len(self.df) * 0.9)

        train = PandasDataset(self.df[:train_end], freq="1d", target="Temp")

        valid = PandasDataset(self.df[train_end:valid_end], freq="1d", target="Temp")

        test = PandasDataset(self.df[valid_end:], freq="1d", target="Temp")
        self._assert_output()

    def _assert_output(self):
        result = self.__getitem__(0)
        print(result)

    def __getitem__(self, idx: int):
        id = self.unique_ids[idx]
        df_person = self.df[self.df[self.id_var] == id].copy()
        dataset = self.reformat(df=df_person)
        return dataset
