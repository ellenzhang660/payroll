import pandas as pd


df = pd.read_csv("time-series/payroll/Lohnkonto2022-2025_english.csv", encoding="utf-8")
unique_names = df["Name"].unique().tolist()

import pandas as pd
import plotly.express as px
import plotly.io as pio


pio.renderers.default = "browser"  # or "notebook" if you're using Jupyter

for person in unique_names:
    # Select payroll period columns
    start_col = 8
    end_col = 49
    period_cols = df.iloc[:, start_col : end_col + 1].columns.tolist()

    # Filter rows for the chosen person
    df_person = df[df["Name"] == person].copy()

    # Convert to numeric
    df_person[period_cols] = df_person[period_cols].applymap(
        lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else 0
    )

    # Melt to long format: one row per (Description, PayrollPeriod)
    df_long = df_person.melt(
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

    fig.write_image(f"time-series/dataset/{person}.png")  # optional: static image
    fig.show()  # interactive plot in browser or notebook


# for person in unique_names:
#     # Select payroll period columns: from 'I' to 'AX'
#     # If you have a list of all columns in df.columns, find indices:
#     start_col = 8 #I
#     end_col = 49
#     period_cols = df.iloc[:, 8:end_col+1].columns.tolist()

#     # Filter rows for the chosen person
#     df_person = df[df['Name'] == person]

#     # Extract payroll periods and Description
#     # We want Description as series labels, data from period_cols as values

#     # Convert payroll period data to numeric, replacing comma decimal separator if needed
#     df_person[period_cols] = df_person[period_cols].applymap(lambda x: float(str(x).replace(',', '.')) if pd.notnull(x) else 0)

#     # Plot each Description row as a separate line
#     plt.figure(figsize=(14, 8))

#     for _, row in df_person.iterrows():
#         plt.plot(period_cols, row[period_cols], marker='o', label=row['Description'])

#     plt.title(f"Payroll Components Over Time for {person}")
#     plt.xlabel("Payroll Period")
#     plt.ylabel("Amount")
#     plt.xticks(rotation=45)
#     plt.legend(loc='upper left', bbox_to_anchor=(1,1))
#     plt.grid(True)
#     plt.tight_layout()
#     filename = f"time-series/dataset/{person}.png"  # or .jpg, .pdf, etc.
#     plt.show()
#     plt.savefig(filename)
