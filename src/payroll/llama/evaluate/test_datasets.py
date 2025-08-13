import json
from pathlib import Path

import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from src.dataset.austria_payroll.data_convert import german_to_english
from src.payroll.llama.evaluate.base_dataset import BaseTestDataset, TestDatasetAttributes
from src.utils import logger


class PayrollDataset(BaseTestDataset):
    """
    Base dataset for payroll testing
    """

    home = Path(__file__).parents[3]
    url = f"{home}/dataset/austria_payroll/Lohnkonto2022-2025_english.csv"
    keys = f"{home}/dataset/austria_payroll/descriptions_count.json"

    def __init__(self, target_column: str):
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self.time_cols = self.df.columns[
            self.df.columns.get_loc("January_2022") : self.df.columns.get_loc("Total Amount")
        ]
        super().__init__(target_column=target_column)

    def _init_attributes(self) -> TestDatasetAttributes:
        with open(f"{self.keys}", "r", encoding="utf-8") as f:
            data = json.load(f)
            keys_with_1 = [k for k, v in data.items() if v[0] == 1.0]  # if == 1.0, all people have this column
            self.available_keys = tuple([german_to_english[k] for k in keys_with_1 if k in german_to_english])
        return TestDatasetAttributes(
            prediction_length=6, context_length=32, available_target_columns=self.available_keys, freq="M", id_var="ID"
        )

    def preprocess(self) -> pd.DataFrame:
        self.df[self.time_cols] = (
            self.df[self.time_cols]
            .apply(lambda col: col.map(lambda x: str(x).replace(",", ".").replace("–", "0") if pd.notnull(x) else "0"))
            .fillna("0")
            .astype("float32")
        )
        df = self.df[self.df[self.time_cols].any(axis=1)].copy()  # filter out zero rows
        return df

    def _log_preprocess(self):
        logger.info("We reformat numbers to float32 and filter out the series with all zero values")

    def _reformat(self, df: pd.DataFrame) -> PandasDataset:
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
            df_pivoted,
            target=self.target_column,
            timestamp="Month",
            item_id=self.attributes.id_var,
            freq=self.attributes.freq,
        )

        return dataset


class GenericDataset(BaseTestDataset):

    url = (
        "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
        "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
    )

    def __init__(self, target_column: str):
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        super().__init__(target_column=target_column)

    def _init_attributes(self) -> TestDatasetAttributes:
        return TestDatasetAttributes(
            prediction_length=24, context_length=32, available_target_columns=("target",), freq="", id_var="item_id"
        )

    def preprocess(self) -> pd.DataFrame:
        # Set numerical columns as float32
        for col in self.df.columns:
            # Check if column is not of string type
            if self.df[col].dtype != "object" and pd.api.types.is_string_dtype(self.df[col]) == False:
                self.df[col] = self.df[col].astype("float32")
        return self.df

    def _log_preprocess(self):
        pass

    def _reformat(self, df: pd.DataFrame) -> PandasDataset:
        dataset = PandasDataset.from_long_dataframe(
            df, target=self.target_column, item_id=self.attributes.id_var, freq=self.attributes.freq
        )
        return dataset
