import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio
from torch.utils.data import Dataset

from src.dataset.austria_payroll.data_convert import german_to_english
from src.utils import logger


pio.renderers.default = "browser"  # or "notebook" if you're using Jupyter


class PayrollDataset(Dataset):
    """
    Base Dataloader for payroll data

    __getitem__(idx):
        returns the target column of that person along with the pandas dataframe with zero rows removed, and a list of other relevant columns

    visualize_sample(idx):
        plots an interactive plot of the dataframe for a specific idx
    """

    home = Path(__file__).parent
    url = f"{home}/Lohnkonto2022-2025_english.csv"
    keys = f"{home}/descriptions_count.json"

    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self.unique_ids = self.df["ID"].unique().tolist()
        self.time_cols = self.df.columns[
            self.df.columns.get_loc("January_2022") : self.df.columns.get_loc("Total Amount")
        ]
        self.df[self.time_cols] = (
            self.df[self.time_cols]
            .apply(lambda col: col.map(lambda x: str(x).replace(",", ".").replace("–", "0") if pd.notnull(x) else "0"))
            .fillna("0")
            .astype("float32")
        )
        self.id_var = "ID"
        self.freq = "M"  # monthly
        with open(f"{self.keys}", "r", encoding="utf-8") as f:
            data = json.load(f)
            keys_with_1 = [k for k, v in data.items() if v[0] == 1.0]
            self.available_keys = list(map(german_to_english.get, keys_with_1))
        self._log_preprocess()

    @abstractmethod
    def _assert_output(self):
        """This method must be implemented by subclasses."""
        pass

    def __len__(self):
        return len(self.unique_ids)

    def _log_preprocess(self):
        logger.info(f"Available keys: {self.available_keys}")
        logger.info(
            "For each unique ID, we filter out the series with all zero values, and also find the duplicate descriptions (although we don't remove them)"
        )

    def preprocess(self, df_person: pd.DataFrame) -> pd.DataFrame:
        df_person = df_person[df_person[self.time_cols].any(axis=1)].copy()  # filter out zero rows
        # Count duplicates by period columns
        group_sizes = df_person.groupby(self.time_cols.tolist()).size()

        # Filter to only groups with more than 1 row (duplicates)
        duplicates = group_sizes[group_sizes > 1]

        for group_vals in duplicates.index:
            # Mask for rows in this group
            mask = (df_person[self.time_cols] == group_vals).all(axis=1)

            # Print only the Description column for these rows
            descriptions = df_person.loc[mask, "Description"]
            print("Redundant group: ", descriptions.to_list())
        return df_person

    def __getitem__(self, idx: int) -> dict[Any, Any]:
        results = {}
        id = self.unique_ids[idx]
        df_person = self.df[self.df[self.id_var] == id].copy()
        df_person = self.preprocess(df_person)
        results["ID"] = id
        results["df"] = df_person
        return results

    def visualize_sample(self, idx: int):
        result = self.__getitem__(idx)
        result["df"].to_csv(f"{self.home}/{result["ID"]}.csv", index=False)

        # Melt to long format: one row per (Description, PayrollPeriod)
        df_long = result["df"].melt(
            id_vars="Description", value_vars=self.time_cols, var_name="PayrollPeriod", value_name="Amount"
        )

        # Plot
        fig = px.line(
            df_long,
            x="PayrollPeriod",
            y="Amount",
            color="Description",
            markers=True,
            title=f"Payroll Components Over Time for {result["ID"]}",
        )

        fig.update_layout(
            xaxis_title="Payroll Period",
            yaxis_title="Amount",
            legend_title="Description",
            autosize=False,
            width=1000,
            height=600,
        )

        fig.show()  # interactive plot in browser or notebook


if __name__ == "__main__":
    dataset = PayrollDataset()
    print(dataset.__getitem__(0))
    dataset.visualize_sample(0)
