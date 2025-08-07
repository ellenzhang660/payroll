import json
from pathlib import Path
from typing import Literal

import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from src.dataset.austria_payroll.data_convert import german_to_english
from src.payroll.llama.finetune.base_dataset import BasePreprocessingInterface, ClassAttributes, FinetuneDataset
from src.utils import logger


# syntehtic data generation
# generate hundreds and thousands of data sample
# but only evaluate on the 14 actual people
# forecast, fraud detection, anamoly etc.


class PayrollDataset(BasePreprocessingInterface):
    """
    Base dataset for payroll finetuning

    __getitem__(mode):
        returns the Pandas dataset with relevant slicing (for train, everything but last window)
    """

    home = Path(__file__).parents[3]
    url = f"{home}/dataset/austria_payroll/Lohnkonto2022-2025_english.csv"
    keys = f"{home}/dataset/austria_payroll/descriptions_count.json"

    def __init__(self, target_column: str):
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self.time_cols = self.df.columns[
            self.df.columns.get_loc("January_2022") : self.df.columns.get_loc("Total Amount")
        ]
        self.id_var = "ID"
        self.target_column = target_column
        with open(f"{self.keys}", "r", encoding="utf-8") as f:
            data = json.load(f)
            keys_with_1 = [k for k, v in data.items() if v[0] == 1.0]
            self.available_keys = list(map(german_to_english.get, keys_with_1))
        assert self.target_column in self.available_keys
        super().__init__()

    def _init_attributes(self) -> ClassAttributes:
        return ClassAttributes(prediction_length=6, context_length=32, target_column=self.target_column, freq="M")

    def _log_preprocess(self):
        logger.info(f"Available keys: {self.available_keys}")
        logger.info("We filter out the series with all zero values")

    def preprocess(self) -> pd.DataFrame:
        self.df[self.time_cols] = (
            self.df[self.time_cols]
            .apply(lambda col: col.map(lambda x: str(x).replace(",", ".").replace("–", "0") if pd.notnull(x) else "0"))
            .fillna("0")
            .astype("float32")
        )
        self.df = self.df[self.df[self.time_cols].any(axis=1)].copy()  # filter out zero rows
        # Melt the time columns to long format
        df_long = self.df.melt(
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
        self.df = df_pivoted.sort_values("Month")
        return self.df

    def __getitem__(self, key: Literal["train", "val", "test"]) -> FinetuneDataset:
        all_months = sorted(self.df["Month"].unique())

        if key == "train":
            # Exclude the last month
            last_month = all_months[-1]
            df = self.df[
                self.df["Month"] < last_month
            ].copy()  # exclude the single last window of the training script and use it as a validation set
        elif key == "val":
            val_months = all_months[-self.attributes.context_length :]
            df = self.df[self.df["Month"].isin(val_months)].copy()
        else:
            df = self.df.copy()

        dataset = PandasDataset.from_long_dataframe(
            df, target=self.target_column, timestamp="Month", item_id=self.id_var, freq=self.attributes.freq
        )

        return FinetuneDataset(dataset=dataset, **vars(self.attributes))


class WeatherDataset(BasePreprocessingInterface):
    def __init__(self):
        self.df = pd.read_csv(
            "https://raw.githubusercontent.com/joshuajnoble/Lag-Llama-Tutorial/main/daily-min-temperatures.csv"
        )
        super().__init__()

    def _init_attributes(self) -> ClassAttributes:
        return ClassAttributes(prediction_length=8, context_length=32, target_column="Temp", freq="1d")

    def _log_preprocess(self):
        pass

    def preprocess(self) -> pd.DataFrame:
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.resample("D").sum().interpolate("linear")

        for col in df.columns:
            # Check if column is not of string type
            if df[col].dtype != "object" and pd.api.types.is_string_dtype(df[col]) == False:
                df[col] = df[col].astype("float32")

        self.df = df
        return self.df

    def __getitem__(self, key: Literal["train", "val", "test"]) -> FinetuneDataset:
        train_end = round(len(self.df) * 0.7)
        valid_end = round(len(self.df) * 0.9)

        if key == "train":
            df = self.df[:train_end].copy()
        elif key == "val":
            df = self.df[train_end:valid_end].copy()
        else:
            df = self.df.copy()

        dataset = PandasDataset(df, freq=self.attributes.freq, target=self.attributes.target_column)

        return FinetuneDataset(dataset=dataset, **vars(self.attributes))
