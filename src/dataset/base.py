from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TimeSeriesData(ABC):
    """
    Absract class for generic time series dataset
    """

    #################### Creator operation ####################
    def __init__(self, url: str):
        """
        Narrow specs for now
        Input: a csv url
        """
        self._url = url  # store as private

    #################### Observer operations ####################
    @abstractmethod
    def how_many_unique_samples(self) -> int:
        """Returns number of samples in the dataset."""

    @abstractmethod
    def what_variates(self) -> set[str]:
        """Returns the variates in the dataset"""

    @abstractmethod
    def length_of_time_series(self) -> int:
        """Returns the number of samples of the time series data, assuming that each person + variable combination is the same"""

    @abstractmethod
    def frequency_of_time_series(self) -> str:
        """Returns the frequency of the time series data"""

    @abstractmethod
    def how_much_memory_in_MB(self) -> float:
        """Returns the size of the dataset in MB"""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, pd.Series]:
        """
        Reurns a dictionary mapping all variates for person idx to their time sereis data.
        If variate key does not exist for particular idx, value is empty pandas series
        Pandas series long format
            index = time stamp 
            column = value
        """

    #################### Representation ####################
    def __str__(self) -> str:
        """Pretty-print all observer information about the dataset."""
        return (
            f"Time Series Dataset Summary:\n"
            f"  Unique samples       : {self.how_many_unique_samples()}\n"
            f"  Variates             : {', '.join(self.what_variates())}\n"
            f"  Time series length   : {self.length_of_time_series()}\n"
            f"  Frequency            : {self.frequency_of_time_series()}\n"
            f"  Memory usage (MB)    : {self.how_much_memory_in_MB():.2f}"
        )

    def __repr__(self) -> str:
        return f"TimeSeriesDaa({self._url})"


def visualize(series: pd.Series):
    # ✅ Create figure/axes explicitly
    fig, ax = plt.subplots(figsize=(14, 6)) 

    ax.plot(series.index, series.values, label="Target", linewidth=2)

    ax.set_xlabel("Time") #type: ignore
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    return fig, ax
