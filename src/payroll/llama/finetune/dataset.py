import json
from pathlib import Path

import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from src.dataset.austria_payroll.data_convert import german_to_english
from src.payroll.llama.finetune.base_dataset import BaseFinetuningDataset, ClassAttributes, FinetuneDataset
from src.utils import logger


class PayrollDataset(BaseFinetuningDataset):
    """
    Base dataset for payroll finetuning
    """

    home = Path(__file__).parents[3]
    url = f"{home}/dataset/austria_payroll/Lohnkonto2022-2025_english.csv"
    keys = f"{home}/dataset/austria_payroll/descriptions_count.json"

    def __init__(self):
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self.time_cols = self.df.columns[
            self.df.columns.get_loc("January_2022") : self.df.columns.get_loc("Total Amount")
        ]
        self.id_var = "ID"
        with open(f"{self.keys}", "r", encoding="utf-8") as f:
            data = json.load(f)
            keys_with_1 = [k for k, v in data.items() if v[0] == 1.0]  # if == 1.0, all people have this column
            self.available_keys: list[str] = [german_to_english[k] for k in keys_with_1 if k in german_to_english]
        super().__init__()

    def _init_attributes(self) -> ClassAttributes:
        return ClassAttributes(
            prediction_length=6, context_length=32, available_target_columns=self.available_keys, freq="M"
        )

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

    def __getitem__(self, target_column: str) -> FinetuneDataset:
        assert target_column in self.attributes.available_target_columns
        all_months = sorted(self.df["Month"].unique())
        # exclude the single last window of the training script and use it as a validation set
        last_month = all_months[-1]
        train_df = self.df[self.df["Month"] < last_month].copy()
        val_months = all_months[-self.attributes.context_length :]
        val_df = self.df[self.df["Month"].isin(val_months)].copy()

        train_dataset = PandasDataset.from_long_dataframe(
            dataframe=train_df, target=target_column, timestamp="Month", item_id=self.id_var, freq=self.attributes.freq
        )
        val_dataset = PandasDataset.from_long_dataframe(
            dataframe=val_df, target=target_column, timestamp="Month", item_id=self.id_var, freq=self.attributes.freq
        )

        return FinetuneDataset(train_dataset=train_dataset, val_dataset=val_dataset, **vars(self.attributes))


class WeatherDataset(BaseFinetuningDataset):
    def __init__(self):
        self.df = pd.read_csv(
            "https://raw.githubusercontent.com/joshuajnoble/Lag-Llama-Tutorial/main/daily-min-temperatures.csv"
        )
        super().__init__()

    def _init_attributes(self) -> ClassAttributes:
        return ClassAttributes(prediction_length=8, context_length=32, available_target_columns=["Temp"], freq="1d")

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

    def __getitem__(self, target_column: str) -> FinetuneDataset:
        assert target_column in self.attributes.available_target_columns
        train_end = round(len(self.df) * 0.7)
        valid_end = round(len(self.df) * 0.9)

        train_df = self.df[:train_end].copy()
        val_df = self.df[train_end:valid_end].copy()

        train_dataset = PandasDataset(train_df, freq=self.attributes.freq, target=target_column)
        val_dataset = PandasDataset(val_df, freq=self.attributes.freq, target=target_column)

        return FinetuneDataset(train_dataset=train_dataset, val_dataset=val_dataset, **vars(self.attributes))
