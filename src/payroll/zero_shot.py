# To filter warnings for readability
import warnings

import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from llama import get_lag_llama_predictions


warnings.simplefilter("ignore", UserWarning)
from itertools import islice

import matplotlib.dates as mdates

# For this dataset
from gluonts.dataset.common import ListDataset
from matplotlib import pyplot as plt


def load_dataset_ts():
    url = (
        "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
        "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
    )
    df = pd.read_csv(url, index_col=0, parse_dates=True)

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != "object" and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype("float32")

    # Create the Pandas
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

    backtest_dataset = dataset
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100  # number of samples sampled from the probability distribution for each timestep
    # device = torch.device("cuda:0") # You can switch this to CPU or other GPUs if you'd like, depending on your environment
    device = torch.device("cpu")
    forecasts, tss = get_lag_llama_predictions(
        dataset=backtest_dataset, prediction_length=prediction_length, device=device, num_samples=num_samples
    )
    # return backtest_dataset, prediction_length, num_samples, device


def load_payroll():

    url = "time-series/payroll/Lohnkonto2022-2025_english.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    id = 12219134

    # 1. Filter for the desired ID
    df_filtered = df[df["ID"] == id].copy()

    # 2. Identify the monthly time columns (from "January_2022" to "Total Amount" exclusive)
    time_cols = df.columns[df.columns.get_loc("January_2022") : df.columns.get_loc("Total Amount")]

    # 3. Melt the time columns to long format
    df_long = df_filtered.melt(
        id_vars=["ID", "Description"], value_vars=time_cols, var_name="Month", value_name="Amount"
    )

    # 4. Clean values: convert commas to decimal points, dashes or NaNs to 0, and to float32
    df_long["Amount"] = df_long["Amount"].replace(",", ".", regex=True).replace("–", "0").fillna(0.0).astype("float32")

    # 5. Pivot to desired format: months as index, descriptions as columns
    df_pivoted = df_long.pivot_table(
        index="Month",
        columns="Description",
        values="Amount",
        aggfunc="first",  # use 'first' since there's only one value per ID/Description/Month
    ).fillna(0.0)

    # Optional: Reset index if you want Month as a column
    df_pivoted = df_pivoted.reset_index()

    # Optional: Convert Month to datetime
    df_pivoted["Month"] = pd.to_datetime(df_pivoted["Month"], format="%B_%Y")
    df_pivoted = df_pivoted.sort_values("Month")
    df_pivoted["ID"] = id

    df_pivoted.columns

    cols_except_month_and_id = df_pivoted.columns[~df_pivoted.columns.isin(["Month", "ID"])].tolist()
    cols_except_month_and_id

    dataset = PandasDataset.from_long_dataframe(
        df_pivoted, target="Gross Salary (Emphasized)", timestamp="Month", item_id="ID", freq="M"  # Monthly Start
    )

    backtest_dataset = dataset
    prediction_length = 9  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 5  # number of samples sampled from the probability distribution for each timestep
    device = torch.device(
        "cpu"
    )  # You can switch this to CPU or other GPUs if you'd like, depending on your environment
    # return backtest_dataset, prediction_length, num_samples, device
    forecasts, tss = get_lag_llama_predictions(
        dataset=backtest_dataset, prediction_length=prediction_length, device=device, num_samples=num_samples
    )
    len(forecasts)
    forecasts[0].samples.shape

    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter("%b, %d")
    plt.rcParams.update({"font.size": 15})

    # Iterate through the first 9 series, and plot the predicted samples
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx + 1)

        plt.plot(
            ts[-4 * prediction_length :].to_timestamp(),
            label="target",
        )
        forecast.plot(color="g")
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()


def load_aus():
    df = pd.read_csv(
        "https://gist.githubusercontent.com/dannymorris/ac176586e0236bd9278e9c81e06851a8/raw/54fd7c7520702d3dd7d4bd59c9dfbed5385af438/aus_retail.csv"
    )
    df = df.set_index("Month")

    df.head()
    metadata = {"prediction_length": 12, "freq": "1ME"}
    train_data = [{"start": df.index[0], "target": df[i].values[: -metadata["prediction_length"]]} for i in df.columns]
    test_data = [{"start": df.index[0], "target": df[i].values} for i in df.columns]

    train_ds = ListDataset(data_iter=train_data, freq=metadata["freq"])

    test_ds = ListDataset(data_iter=test_data, freq=metadata["freq"])
    device = torch.device(
        "cpu"
    )  # You can switch this to CPU or other GPUs if you'd like, depending on your environment
    forecasts_ctx_len_32, tss_ctx_len_32 = get_lag_llama_predictions(
        test_ds,
        prediction_length=metadata["prediction_length"],
        device=device,
        context_length=32,
        use_rope_scaling=False,
        num_samples=30,
    )


# load_aus()
# load_dataset_ts()
load_payroll()
function = load_payroll
# function = load_dataset_ts
forecasts, tss = get_lag_llama_predictions(
    dataset=backtest_dataset, prediction_length=prediction_length, device=device, num_samples=num_samples
)
len(forecasts)
forecasts[0].samples.shape

plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter("%b, %d")
plt.rcParams.update({"font.size": 15})

# Iterate through the first 9 series, and plot the predicted samples
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx + 1)

    plt.plot(
        ts[-4 * prediction_length :].to_timestamp(),
        label="target",
    )
    forecast.plot(color="g")
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()
