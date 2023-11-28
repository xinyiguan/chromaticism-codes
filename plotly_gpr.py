from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gpr_analysis import gpr_model_outputs, MODEL_OUTPUT


def plot_gpr_scatter(df: pd.DataFrame, df_type: Literal["MajorMode", "MinorMode", "CombinedMode"],
                     feature: Literal["RC", "CTC", "NCTC"],
                     m_outputs: MODEL_OUTPUT, color: str, save_name:Optional[str]):
    gpr_X = m_outputs[0].data[0]
    Xplot = np.arange(min(gpr_X), max(gpr_X + 1))

    f_mean, f_var, y_mean, y_var = m_outputs[1]
    f_mean = np.squeeze(f_mean)
    # y_mean = np.squeeze(y_mean)

    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    # Add traces
    fig.add_trace(go.Scatter(x=df["piece_year"],
                             y=df[feature],
                             mode='markers',
                             marker=dict(
                                 size=10,
                                 color=color,
                                 opacity=0.3,
                                 symbol='circle',
                                 line=dict(width=2, color=color)
                             ),
                             name=f'{feature}',
                             text=df["corpus"],
                             customdata=df[['piece', 'globalkey']],  # Include custom data ('piece' column)
                             hovertemplate=
                             "<b>%{text}</b><br><br>" +
                             "piece: %{customdata[0]}<br>" +  # Access 'piece' column using %{customdata.piece}
                             "global key: %{customdata[1]}<br>" +
                             "year: %{x}<br>" +
                             "value: %{y}<br>" +
                             "index: %{customdata[1]}<br>" +
                             "<extra></extra>",
                             ))

    fig.add_trace(go.Scatter(x=Xplot, y=f_mean, mode="lines",
                             line=dict(
                                 color=color,
                                 width=3),
                             name=f"{feature}_GPR_fmean"))
    fig.show()
    if isinstance(save_name, str):
        fig.write_html(f"figs/interactive_figs/gpr_{df_type}_{feature}.html")





def experiment_interactive_gpr_CIs(major: pd.DataFrame, minor: pd.DataFrame, optimized: bool=True):
    scatter_colors_palette6 = ["#ad0041", "#ff6232", "#ffa94d", "#9cde9e", "#39c5a3", "#0088c1"]

    major_rc = gpr_model_outputs(df=major, df_type="Major",  feature="RC", lengthscale=10)
    major_ctc = gpr_model_outputs(df=major, df_type="Major",feature="CTC", lengthscale=10)
    major_nctc = gpr_model_outputs(df=major, df_type="Major",feature="NCTC", lengthscale=10)

    minor_rc = gpr_model_outputs(df=minor, df_type="Minor", feature="RC", lengthscale=10)
    minor_ctc = gpr_model_outputs(df=minor, df_type="Minor",feature="CTC", lengthscale=10)
    minor_nctc = gpr_model_outputs(df=minor, df_type="Minor", feature="NCTC", lengthscale=10)

    plot_gpr_scatter(df=major, df_type="MajorMode", feature="RC", m_outputs=major_rc, color=scatter_colors_palette6[0], save_name="major_rc_handtuned")
    plot_gpr_scatter(df=major, df_type="MajorMode", feature="CTC", m_outputs=major_ctc, color=scatter_colors_palette6[1], save_name="major_ctc_handtuned")
    plot_gpr_scatter(df=major, df_type="MajorMode", feature="NCTC", m_outputs=major_nctc, color=scatter_colors_palette6[2], save_name="major_nctc_handtuned")
    plot_gpr_scatter(df=minor, df_type="MinorMode", feature="RC", m_outputs=minor_rc, color=scatter_colors_palette6[3], save_name="minor_rc_handtuned")
    plot_gpr_scatter(df=minor, df_type="MinorMode",feature="CTC", m_outputs=minor_ctc, color=scatter_colors_palette6[4], save_name="minor_ctc_handtuned")
    plot_gpr_scatter(df=minor, df_type="MinorMode",feature="NCTC", m_outputs=minor_nctc, color=scatter_colors_palette6[5], save_name="minor_nctc_handtuned")


def experiment_interactive_gpr_5thsRange(df:pd.DataFrame):
    scatter_colors_palette_3 = ["#A894B0", "#57a1be", "#91ad70"]

    r_fifths_range = gpr_model_outputs(df=df, feature="r_fifths_range", df_type="Combined")
    ct_fifths_range = gpr_model_outputs(df=df, feature="ct_fifths_range", df_type="Combined")
    nct_fifths_range = gpr_model_outputs(df=df, feature="nct_fifths_range", df_type="Combined")

    plot_gpr_scatter(df=df, df_type="CombinedMode", feature="r_fifths_range", m_outputs=r_fifths_range, color=scatter_colors_palette_3[0], save_name="r_5thsRange")
    plot_gpr_scatter(df=df, df_type="CombinedMode", feature="ct_fifths_range", m_outputs=ct_fifths_range, color=scatter_colors_palette_3[1], save_name="ct_5thsRange")
    plot_gpr_scatter(df=df, df_type="CombinedMode", feature="nct_fifths_range", m_outputs=nct_fifths_range, color=scatter_colors_palette_3[2], save_name="nct_5thsRange")


if __name__ == "__main__":
    piece_df = pd.read_csv("data/piece_indices.tsv", sep="\t")

    major = pd.read_csv("data/majorkey_piece_indices.tsv", sep="\t")

    minor = pd.read_csv("data/minorkey_piece_indices.tsv", sep="\t")

    experiment_interactive_gpr_5thsRange(df=piece_df)
