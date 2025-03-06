import pandas as pd
import plotly.graph_objects as go

from final_project_btb.config import BLD, numeric_cols


def task_generating_plot_of_financial_variables(
    data=BLD / "clean_table.csv", produces=BLD / "financials_per_bank.png"
):
    """Generates a Radar Graph and saves it in BLD folder."""
    df_for_plot = pd.read_csv(data, engine="pyarrow")
    df_for_plot[numeric_cols] = (
        df_for_plot[numeric_cols] / df_for_plot[numeric_cols].max()
    )

    labels_map = {
        "paid_up_capital": "Paid-up Capital",
        "surp_and_prof": "Surplus and Profits",
        "deposits": "Deposits",
        "loans_and_discounts_stocks_and_securities": "Loans and Discounts",
        "cash_and_exchanges": "Cash and Exchanges",
    }

    formatted_labels = [labels_map[col] for col in numeric_cols]

    fig = go.Figure()
    for _, row in df_for_plot.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=row[numeric_cols].values,
                theta=formatted_labels,
                fill="toself",
                name=row["name_of_bank"],
            )
        )

    fig.update_layout(
        title={"text": "Financial Variables per Bank"},
        polar={"radialaxis": {"visible": True}},
    )

    fig.write_image(produces)
