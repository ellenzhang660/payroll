from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


current_dir = Path(__file__).parent

df = pd.read_csv(f"{current_dir}/Lohnkonto2022-2025_english.csv", encoding="utf-8")
unique_names = df["Name"].unique().tolist()

pio.renderers.default = "browser"  # or "notebook" if you're using Jupyter


for person in unique_names:
    # Select payroll period columns
    # start_col = 8
    # end_col = 49
    period_cols = df.columns[df.columns.get_loc("January_2022") : df.columns.get_loc("Total Amount")].tolist()

    # Filter rows for the chosen person
    df_person = df[df["Name"] == person].copy()

    # Convert to numeric (replace comma decimal and fill NaNs with 0)
    df_person[period_cols] = df_person[period_cols].applymap(
        lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else 0
    )

    # Filter rows where **any** value in period_cols is non-zero (keep rows with at least one non-zero)
    df_person_filtered = df_person[df_person[period_cols].any(axis=1)].copy()

    # Count duplicates by period columns
    group_sizes = df_person_filtered.groupby(period_cols).size()

    # Filter to only groups with more than 1 row (duplicates)
    duplicates = group_sizes[group_sizes > 1]

    print("Found these redundant groups (period columns) with more than 1 row:")
    # print(duplicates)

    for group_vals in duplicates.index:
        # print(f"\nGroup with period columns = {group_vals}:")
        # Mask for rows in this group
        mask = (df_person_filtered[period_cols] == group_vals).all(axis=1)

        # Print only the Description column for these rows
        descriptions = df_person_filtered.loc[mask, "Description"]
        print(descriptions.to_list())

        # Melt to long format: one row per (Description, PayrollPeriod)
        df_long = df_person_filtered.melt(
            id_vars="Description", value_vars=period_cols, var_name="PayrollPeriod", value_name="Amount"
        )

    # Plot
    fig = px.line(
        df_long,
        x="PayrollPeriod",
        y="Amount",
        color="Description",
        markers=True,
        title=f"Payroll Components Over Time for {person}",
    )

    fig.update_layout(
        xaxis_title="Payroll Period",
        yaxis_title="Amount",
        legend_title="Description",
        autosize=False,
        width=1000,
        height=600,
    )

    # fig.write_image(f"src/dataset_visualizations/{person}.png")  # optional: static image
    fig.show()  # interactive plot in browser or notebook
