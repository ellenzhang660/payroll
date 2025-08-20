from pathlib import Path

import pandas as pd

from src.dataset.base import TimeSeriesData


class PayrollDataset(TimeSeriesData):
    """
    Absract class for generic time series dataset
    """

    home = Path(__file__).parents[2]
    payroll_url = f"{home}/database/Lohnkonto2022-2025_english.csv"

    #################### Creator operation ####################
    def __init__(self):
        super().__init__(url=self.payroll_url)
        self.df = pd.read_csv(self.payroll_url, index_col=0, parse_dates=True)
        self.time_cols = self.df.columns[
            self.df.columns.get_loc("January_2022") : self.df.columns.get_loc("Total Amount")
        ]
        self._clean()
        self.id_var = "ID"
        self.unique_samples = self.df[self.id_var].unique().tolist()  # type: ignore

    def _clean(self):
        self.df[self.time_cols] = (
            self.df[self.time_cols]
            .apply(lambda col: col.map(lambda x: str(x).replace(",", ".").replace("–", "0") if pd.notnull(x) else "0"))
            .fillna("0")
            .astype("float32")
        )
        self.df = self.df[self.df[self.time_cols].any(axis=1)].copy()  # filter out zero rows

    def _reformat(self, df_person):
        df_long = df_person.melt(
            id_vars=["ID", "Description"], value_vars=self.time_cols, var_name="Month", value_name="Amount"
        )

        # Pivot to desired format: months as index, descriptions as columns
        df_pivoted = df_long.pivot_table(
            index=["ID", "Month"],  # include ID in the index too
            columns="Description",
            values="Amount",
            aggfunc="first",
        )

        # Reset 'ID' from index to a column but keep 'Month' as the index
        df_pivoted = df_pivoted.reset_index()

        # Optional: Convert Month to datetime
        df_pivoted["Month"] = pd.to_datetime(df_pivoted["Month"], format="%B_%Y")
        return df_pivoted.sort_values("Month")

    #################### Observer operations ####################
    def how_many_unique_samples(self) -> int:
        return len(self.unique_samples)

    def what_variates(self) -> set[str]:
        return set(self.df["Description"].unique())

    def length_of_time_series(self) -> int:
        return len(self.time_cols)

    def frequency_of_time_series(self) -> str:
        return "M"

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
            df_person_variable = df_person.loc[df_person["Description"] == variable].copy()
            if not df_person_variable.empty:
                variates[variable] = self._reformat(df_person=df_person_variable)
        return variates
        
