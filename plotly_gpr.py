from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gpr_analysis import gpr_model_outputs, MODEL_OUTPUT


def plot_gpr_scatter(df: pd.DataFrame, feature: Literal["RC", "CTC", "NCTC"], m_outputs: MODEL_OUTPUT):
    gpr_X = m_outputs[0].data[0]
    Xplot = np.arange(min(gpr_X), max(gpr_X + 1))

    f_mean, f_var, y_mean, y_var = m_outputs[1]
    f_mean = np.squeeze(f_mean)
    y_mean = np.squeeze(y_mean)

    fig = go.Figure()

    # fig.add_trace(go.Scatter(x=Xplot, y=f_mean, mode="lines", name=f"{feature}_GPR_fmean"))

    fig.add_trace(go.Scatter(x=Xplot, y=y_mean, mode="lines", name=f"{feature}_GPR_ymean"))


    # Add traces
    fig.add_trace(go.Scatter(x=df["piece_year"], y=df[feature],
                             mode='markers',
                             name=f'{feature}',
                             text=df["corpus"],
                             customdata=df[['piece']],  # Include custom data ('piece' column)
                             hovertemplate=
                             "<b>%{text}</b><br><br>" +
                             "piece: %{customdata[0]}<br>" +  # Access 'piece' column using %{customdata.piece}
                             "year: %{x}<br>" +
                             "value: %{y}<br>" +
                             "<extra></extra>",
                             ))

    fig.show()


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")

    rc = gpr_model_outputs(df=result_df, feature="RC")
    ctc =gpr_model_outputs(df=result_df, feature="CTC")
    nctc =gpr_model_outputs(df=result_df, feature="NCTC")


    plot_gpr_scatter(result_df, feature="RC", m_outputs=rc)
    plot_gpr_scatter(result_df, feature="CTC", m_outputs=ctc)
    plot_gpr_scatter(result_df, feature="NCTC", m_outputs=nctc)


