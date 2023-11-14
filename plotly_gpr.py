from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gpr_analysis import gpr_model_outputs, MODEL_OUTPUT


def plot_gpr_scatter(df: pd.DataFrame, feature: Literal["RC", "CTC", "NCTC"], m_outputs: MODEL_OUTPUT):

    gpr_X = m_outputs[0].data[0]
    Xplot = np.arange(min(gpr_X), max(gpr_X + 1))

    f_mean, f_var, y_mean, y_var = m_outputs[1]

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df["piece_year"], y=df[feature],
                             mode='markers',
                             name=f'{feature}'),)

    fig.add_trace(go.Scatter(x=Xplot, y=f_mean, mode="lines", name=f"{feature}_GPR"))

    fig.show()


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")

    rc = gpr_model_outputs(df=result_df, feature="RC")

    plot_gpr_scatter(result_df, feature="RC", m_outputs=rc)
