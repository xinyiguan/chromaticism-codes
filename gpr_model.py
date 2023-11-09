from typing import Tuple, Optional, List, Union, Literal

import matplotlib
import numpy as np
import pandas as pd
import seaborn
import sklearn.gaussian_process.kernels as kn
from check_shapes import inherit_check_shapes
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
import tensorflow as tf

import matplotlib.pyplot as plt
import gpflow as gf
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary

import plotly.graph_objects as go
from scipy.signal import argrelextrema  # To find local extrema
from tensorflow import Tensor

from utils.util import rand_jitter, find_local_extrema
import plotly.express as px
import matplotlib.colors as mcolors

MODEL_OUTPUT = Tuple[gf.models.gpr.GPR, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None, str]


# helper funcs:

def assign_piece_coloring(df: pd.DataFrame, tuples_list: List[Tuple[str, str]]) -> np.ndarray:
    # Create a mapping of unique (corpus, piece) pairs to unique numbers
    unique_pairs = pd.DataFrame(tuples_list, columns=['corpus', 'piece']).drop_duplicates()
    unique_pairs['piece_coloring'] = range(1, len(unique_pairs) + 1)
    mapping = unique_pairs.set_index(['corpus', 'piece'])['piece_coloring'].to_dict()

    # Assign piece_coloring values to the DataFrame using the mapping
    df['piece_coloring'] = df[['corpus', 'piece']].apply(lambda x: mapping.get((x['corpus'], x['piece']), 0), axis=1)

    # Convert the 'piece_coloring' column to a NumPy array
    result = df['piece_coloring'].to_numpy()

    return result


# building blocks _________________________________________________________________________________________________

def gpr_model(X: np.ndarray, Y: np.ndarray, kernel: gf.kernels, optimize: bool) -> gf.models.gpr.GPR:
    m = gf.models.gpr.GPR(
        data=(X, Y),
        kernel=kernel,
    )
    if optimize:
        gf.optimizers.Scipy().minimize(
            closure=m.training_loss,
            variables=m.trainable_variables,
            track_loss_history=True
        )
    print_summary(m)
    return m


def model_outputs(df: pd.DataFrame, feature: str, sample: int = 5) -> MODEL_OUTPUT:
    X = df["piece_year"].to_numpy().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
    Y = df[feature].to_numpy().astype(float).reshape((-1, 1))

    k = gf.kernels.SquaredExponential()

    m = gpr_model(X=X, Y=Y, kernel=k, optimize=True)
    # m = gaussian_process_model_empirical_noise(X=X, Y=Y)

    f_mean, f_var = m.predict_f(Xplot, full_cov=False)
    y_mean, y_var = m.predict_y(Xplot)
    if isinstance(sample, int):
        samples = m.predict_f_samples(Xplot, sample)
    else:
        raise TypeError

    return m, (f_mean, f_var, y_mean, y_var), samples, feature


# analysis _________________________________________________________________________________________

def pearson_corr(series_1: np.ndarray, series_2: np.ndarray):
    pearson_corr_coefficient, p_value = pearsonr(series_1, series_2)
    return pearson_corr_coefficient, p_value


# plotting building blocks _________________________________________________________________________________________

def ax_observations_scatter(ax: matplotlib.axes.Axes,
                            X: np.ndarray, Y: np.ndarray,
                            hue_by: np.ndarray | None,
                            with_jitter: bool = True) -> matplotlib.axes.Axes:
    X = X.squeeze(axis=-1)
    Y = Y.squeeze(axis=-1)

    def map_array_to_colors(arr):
        unique_values = np.unique(arr)
        num_unique_values = len(unique_values)

        # Define the colormap using the recommended method
        cmap = mcolors.ListedColormap(cm.Dark2.colors[:num_unique_values])

        # Create a dictionary to map unique values to colors
        value_to_color = dict(zip(unique_values, cmap.colors))

        # Map the values in the array to colors, using "gray" for 0 values
        color_list = [value_to_color[val] if val != 0 else "gray" for val in arr]

        return color_list

    if hue_by is None:
        color = "gray"
        alpha = 0.4
    elif isinstance(hue_by, str):
        color = hue_by
        alpha = 0.4
    elif isinstance(hue_by, np.ndarray):
        color = map_array_to_colors(hue_by)
        alpha = [0.4 if col == "gray" else 1.0 for col in color]
    else:
        raise TypeError

    # adding jitter:
    if with_jitter:
        # only add jitter on the x-axis not y-axis
        ax.scatter(rand_jitter(X), Y, c=color, s=20, label="Observations", alpha=alpha)
    else:
        ax.scatter(X, Y, c=color, s=20, label="Observations", alpha=alpha)

    return ax


def ax_gpr_prediction(ax: matplotlib.axes.Axes,
                      m_outputs: MODEL_OUTPUT,
                      fmean_color: str,
                      fvar_color: Optional[str],
                      plot_samples: int | None,
                      plot_f_uncertainty: bool = True,
                      plot_y_uncertainty: bool = True
                      ) -> matplotlib.axes.Axes:
    """
    plot the regression mean line f and the std
    """
    f_mean, f_var, y_mean, y_var = m_outputs[1]
    samples = m_outputs[2]

    X = m_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))

    f_lower = f_mean.numpy() - 1.96 * np.sqrt(f_var)
    f_upper = f_mean.numpy() + 1.96 * np.sqrt(f_var)
    y_lower = y_mean.numpy() - 1.96 * np.sqrt(y_var)
    y_upper = y_mean.numpy() + 1.96 * np.sqrt(y_var)
    feature = m_outputs[3]

    ax.plot(Xplot, f_mean, "-", color=fmean_color, label=f"{feature} GPR prediction", linewidth=2)

    if plot_f_uncertainty:
        ax.plot(Xplot, f_lower, "--", color=fvar_color, label="f 95% confidence", alpha=0.2)
        ax.plot(Xplot, f_upper, "--", color=fvar_color, alpha=0.2)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fvar_color, alpha=0.2
        )

    if plot_y_uncertainty:
        ax.plot(Xplot, y_lower, ".", color=fvar_color, label="Y 95% confidence", alpha=0.1)
        ax.plot(Xplot, y_upper, ".", color=fvar_color, alpha=0.1)
        ax.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
        )

    if plot_samples:
        ax.plot(Xplot, samples[:, :, 0].numpy().T, 'gray', linewidth=0.5, alpha=0.6)

    # ax.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    # ax.plot(Xplot, y_upper, ".", color="C0")
    # ax.fill_between(
    #     Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    # )

    ax.legend()
    return ax


def ax_full_gpr_model(ax: matplotlib.axes.Axes,
                      m_outputs: MODEL_OUTPUT,
                      fmean_color: str,
                      fvar_color: Optional[str],
                      hue_by: np.ndarray | None,
                      plot_samples: bool,
                      plot_f_uncertainty: bool = True,
                      plot_y_uncertainty: bool = True
                      ) -> matplotlib.axes.Axes:
    """
    plot the scatter ax and the gpr prediction ax
    """
    feature = m_outputs[3]
    ax.set_title(f'{feature}', fontsize=12)
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])

    ax_observations_scatter(ax=ax, X=X, Y=Y, hue_by=hue_by, with_jitter=True)
    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color, fvar_color=fvar_color,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)
    return ax


def ax_gpr_mean(ax: matplotlib.axes.Axes,
                model_outputs: MODEL_OUTPUT,
                color, linewidth: int) -> matplotlib.axes.Axes:
    f_mean = model_outputs[1][0].numpy().flatten()

    X = model_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
    feature = model_outputs[3]

    ext_max_indices, ext_min_indices = find_local_extrema(f_mean)

    ax.plot(Xplot, f_mean, "-", label=f"{feature}", color=color, linewidth=linewidth)

    ax.plot(Xplot[ext_max_indices], f_mean[ext_max_indices], 'yo', label=f'Local Maxima {feature}')
    ax.plot(Xplot[ext_min_indices], f_mean[ext_min_indices], 'bo', label=f'Local Minima {feature}')

    return ax


# plotting _________________________________________________________________________________________________________

def plot_fifths_range(model_outputs_list: List[MODEL_OUTPUT],
                      hue_by: np.ndarray | str | None,
                      mean_color: List,
                      plot_samples: bool = False
                      ) -> plt.Figure:
    num_features = len(model_outputs_list)

    # fig = plt.figure(figsize=(8 * num_features, 5))
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 6), sharey=True)

    text_kws = {
        "rotation": 90,
        "horizontalalignment": "center",
        "verticalalignment": "center"
    }

    for i, (ax, out) in enumerate(zip(axes, model_outputs_list)):
        ax_full_gpr_model(ax=ax, hue_by=hue_by,
                          m_outputs=out,
                          fmean_color=mean_color[i], fvar_color=mean_color[i],
                          plot_samples=plot_samples,
                          plot_f_uncertainty=True,
                          plot_y_uncertainty=False)

        ax.axhline(6, c="gray", linestyle="--", lw=1)  # dia / chrom.
        ax.axhline(12, c="gray", linestyle="--", lw=1)  # chr. / enh.

        ax.text(1965, 3, "diatonic", **text_kws)
        ax.text(1965, 9, "chromatic", **text_kws)
        ax.text(1965, 23, "enharmonic", **text_kws)

        ax.set_ylabel("Fifths range")
        ax.set_xlabel("Year")

        ax.set_ybound(0, 30)
        ax.legend(loc="upper left")

        # ax_index += 1

    fig.tight_layout()
    return fig


def plot_multiple_full_gpr_models(model_outputs_list: List[MODEL_OUTPUT],
                                  hue_by: np.ndarray | str | None,
                                  mean_color: List,
                                  plot_samples: bool = False
                                  ) -> plt.Figure:
    num_features = len(model_outputs_list)

    # fig = plt.figure(figsize=(8 * num_features, 5))
    fig, axes = plt.subplots(num_features,1, figsize=(10, 5 * num_features), sharex=True)


    for i, (ax, out) in enumerate(zip(axes, model_outputs_list)):

        ax_full_gpr_model(ax=ax, hue_by=hue_by,
                          m_outputs=out,
                          fmean_color=mean_color[i], fvar_color=mean_color[i],
                          plot_samples=plot_samples,
                          plot_f_uncertainty=True,
                          plot_y_uncertainty=False)

        ax.axhline(0, c="#e1ad01", linestyle="--", lw=2)  # dia / chrom.

        ax.set_ylabel("Chromaticity index")
        ax.set_xlabel("Year")

        # ax.set_ybound(0, 30)
        ax.legend(loc="upper left")

        # ax_index += 1

    fig.tight_layout()
    return fig


def old_plot_multiple_full_gpr_models(model_outputs_list: List[MODEL_OUTPUT],
                                      hue_by: np.ndarray | str | None) -> plt.Figure:
    num_features = len(model_outputs_list)

    fig = plt.figure(figsize=(8, 6 * num_features))

    # fmean_colors = [f'C{i}' for i in range(9)]

    ax_index = 1
    for i, out in enumerate(model_outputs_list):
        ax = fig.add_subplot(num_features, 1, ax_index)
        ax_full_gpr_model(ax=ax, hue_by=hue_by,
                          m_outputs=model_outputs_list[i],
                          fmean_color="C3",
                          fvar_color="C0", plot_samples=False, plot_y_uncertainty=False)
        ax_index += 1

    fig.tight_layout()
    return fig


def plot_all_gpr_predictions(model_outputs_list: List[MODEL_OUTPUT]) -> plt.Figure:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))

    ax_gpr_prediction(ax[0, 0], model_outputs_list[0], fmean_color="C0", fvar_color="C0")
    ax_gpr_prediction(ax[0, 1], model_outputs_list[1], fmean_color="C1", fvar_color="C1")
    ax_gpr_prediction(ax[1, 0], model_outputs_list[2], fmean_color="C2", fvar_color="C2")
    ax_gpr_prediction(ax[1, 1], model_outputs_list[3], fmean_color="C3", fvar_color="C3")

    fig.tight_layout()
    return fig


def plot_gpr_fmeans(model_outputs_list: List[MODEL_OUTPUT],
                    color: Optional[List],
                    labels: List) -> plt.Figure:
    num_features = len(model_outputs_list)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Gaussian Process Regression Mean', fontsize=15)

    lns = []

    if color is None:
        color = plt.cm.viridis(np.linspace(0, 1, num_features))

    # Create a secondary axis that shares the same y-axis on the right
    ax2 = ax1.twinx()

    for i, out in enumerate(model_outputs_list):
        f_mean, f_var, _, _ = out[1]

        f_lower = f_mean.numpy() - 1.96 * np.sqrt(f_var)
        f_upper = f_mean.numpy() + 1.96 * np.sqrt(f_var)

        X = out[0].data[0]
        Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
        feature = out[3]
        # ext_max_indices, ext_min_indices = find_local_extrema(f_mean.numpy())

        if i == 0:
            ax = ax1
        else:
            ax = ax2

        ln = ax.plot(Xplot, f_mean, "-", label=labels[i], color=color[i], linewidth=2)
        # ax.plot(Xplot, f_lower, "--", color=color[i], label="f 95% confidence", alpha=0.2)
        # ax.plot(Xplot, f_upper, "--", color=color[i], alpha=0.2)
        # ax.fill_between(
        #     Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color[i], alpha=0.3
        # )

        # ax.plot(Xplot[ext_max_indices], f_mean[ext_max_indices], "^", label=f'Local Maxima {feature}')
        # ax.plot(Xplot[ext_min_indices], f_mean[ext_min_indices], "v", label=f'Local Minima {feature}')
        lns.extend(ln)

    labs = [l.get_label() for l in lns]

    ax1.tick_params(axis='x', labelcolor="black", labelbottom=True, labeltop=False)

    ax1.legend(lns, labs, loc=2)

    # ax1.grid()
    ax1.set_xlabel("Time (year)")
    ax1.set_ylabel(r'$\mathregular{RC_p}$')
    ax2.set_ylabel(r'$\mathregular{CTC_p} & \mathregular{NCTC_p}$')

    plt.tight_layout()
    fig.show()
    return fig


def plot_gpr_fmeans_scaled(model_outputs_list: List[MODEL_OUTPUT],
                           color: Optional[List],
                           labels: List) -> plt.Figure:
    num_features = len(model_outputs_list)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Gaussian Process Regression Mean', fontsize=15)

    lns = []

    if color is None:
        color = plt.cm.viridis(np.linspace(0, 1, num_features))

    # Create a secondary axis that shares the same y-axis on the right
    ax = ax1.twinx().twiny()

    for i, out in enumerate(model_outputs_list):
        f_mean, f_var, _, _ = out[1]

        X = out[0].data[0]
        Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))

        # if i == 0:
        # ax = ax1
        # else:
        #     ax = ax2

        ln = ax.plot(Xplot, f_mean, "-", label=labels[i], color=color[i], linewidth=2)
        lns.extend(ln)

    labs = [l.get_label() for l in lns]

    ax1.tick_params(axis='x', labelcolor="black", labelbottom=True, labeltop=False)

    ax1.legend(lns, labs, loc=2)

    # ax1.grid()
    ax1.set_xlabel("Time (year)")
    ax1.set_ylabel('chromaticity index')
    # ax2.set_ylabel(r'$\mathregular{CTC_p} & \mathregular{NCTC_p}$')

    plt.tight_layout()
    fig.show()
    return fig


def plot_gpr_fmeans_derivatives(model_outputs_list: List[MODEL_OUTPUT],
                                color: Optional[List],
                                labels: List) -> plt.Figure:
    num_features = len(model_outputs_list)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Gaussian Process Regression Mean (first-order derivative)', fontsize=15)

    lns = []

    if color is None:
        color = plt.cm.viridis(np.linspace(0, 1, num_features))

    for i, out in enumerate(model_outputs_list):
        f_mean, f_var, _, _ = out[1]

        X = out[0].data[0]
        Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))

        # Calculate the empirical derivative using finite differences
        derivative = np.gradient(f_mean.numpy().flatten(), Xplot.flatten())

        ln = ax1.plot(Xplot, derivative, "-", label=labels[i], color=color[i], alpha=0.7)

        lns.extend(ln)

        # Set the x-axis ticks and rotate the labels
        x_ticks = np.arange(min(Xplot), max(Xplot) + 1, step=25)  # Adjust the step as needed
        x_tick_labels = x_ticks.flatten().astype(int)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels, rotation=45, ha="right")

    labs = [l.get_label() for l in lns]

    ax1.tick_params(axis='both', labelcolor="black", labelbottom=True, labeltop=False)

    ax1.legend(lns, labs, loc='upper right')  # Move the legend to the upper right

    ax1.set_xlabel("Time (year)")
    ax1.set_ylabel('First-order derivative')

    plt.tight_layout()
    plt.show()  # Use plt.show() to display the plot
    return fig


# plotly interactive plots ________________________________________________________________________________________

def plotly_fifth_range(df: pd.DataFrame, feature: Literal["root", "ct", "nct"]):
    import plotly.express as px
    fig = px.scatter(df, x="piece_year", y=f'{feature}_fifths_range',
                     hover_data=['corpus', "piece", "corpus_id", "piece_id"])
    fig.show()


def plotly_2features_scatter(df: pd.DataFrame, feature1: str, feature2):
    import plotly.express as px
    fig = px.scatter(df, x=feature1, y=feature2, hover_data=['corpus', "piece"], color="piece_year",
                     color_continuous_scale="purpor")
    fig.update_layout(showlegend=True,
                      plot_bgcolor='white'
                      )
    fig.show()


def plotly_3features_scatter(df: pd.DataFrame, features: List[str], hue_by: List):
    # color = df["piece_year"].tolist()
    color = hue_by

    fig = go.Figure(data=[go.Scatter3d(
        x=df[features[0]],
        y=df[features[1]],
        z=df[features[2]],
        mode='markers',
        marker_symbol='diamond',
        marker=dict(
            size=4,
            color=color,  # set color to an array/list of desired values
            colorscale='darkmint',  # choose a colorscale
            opacity=0.6
        ),
        text=df["corpus"],
        customdata=df[['piece', 'piece_year']],
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "piece: %{customdata[0]}<br>" +
        "year: %{customdata[1]}<br>" +
        "root_chromaticity: %{x}<br>" +
        "ct_chromaticity: %{y}<br>" +
        "nct_chromaticity: %{z}<br>" +
        "<extra></extra>",
    )])

    fig.update_layout(title=f'{features} chromaticity 3D plot', title_x=0.5)

    fig.update_layout(scene=dict(
        xaxis_title=f'{features[0]}',
        xaxis=dict(
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ),
        yaxis_title=f'{features[1]}',
        yaxis=dict(
            backgroundcolor="rgb(230, 200,230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white"),
        zaxis_title=f'{features[2]}',
        zaxis=dict(
            backgroundcolor="rgb(230, 230,200)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white")))
    fig.show()
    return fig


def plotly_feature_gpr(df: pd.DataFrame, m_outputs: MODEL_OUTPUT):
    import plotly.graph_objects as go
    import plotly.express as px

    f_mean, f_var, _, _ = m_outputs[1]
    feature = m_outputs[3]
    X = m_outputs[0].data[0].numpy().squeeze(axis=-1)
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
    f_mean = f_mean.numpy()
    Y = m_outputs[0].data[1].numpy().squeeze(axis=-1)

    fig = go.Figure()
    fig.update_layout(title=f'{feature}', xaxis_title='Year', yaxis_title=f'{feature}', title_x=0.5)

    fig.add_trace(go.Scatter(
        x=X,
        y=Y,
        mode='markers',
        name='Observations',
        # marker=dict(size=8, color=df["corpus_id"], colorscale='Viridis', opacity=0.5),
        marker=dict(size=8, opacity=0.5),
        text=df["corpus"],
        customdata=df[['piece', "corpus_id", "piece_id"]],  # Include custom data ('piece' column)
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "piece: %{customdata[0]}<br>" +  # Access 'piece' column using %{customdata.piece}
        "year: %{x}<br>" +
        "value: %{y}<br>" +
        "corpus id: %{customdata[1]}<br>" +
        "piece id: %{customdata[2]}<br>" +
        "<extra></extra>",
    ))

    fig.add_trace(
        go.Scatter(x=Xplot[:, 0], y=f_mean[:, 0], mode='lines', name='mean',
                   line=dict(width=3, color='darkorange')))

    fig.update_layout(showlegend=True,
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
    fig.show()


if __name__ == "__main__":
    result_df = pd.read_csv("old_data/piecelevel_chromaticities.tsv", sep="\t")
    # hue_by = result_df["corpus_id"].to_numpy().astype(int).reshape((-1, 1))

    # weighted_df = pd.read_csv("data/weighted_piecelevel_chromaticities.tsv", sep="\t")
    # weighted_r = gpr_model_outputs(df=weighted_df, feature="weighted_r_chromaticity")
    # weighted_ct = gpr_model_outputs(df=weighted_df, feature="weighted_ct_chromaticity")
    # weighted_nct = gpr_model_outputs(df=weighted_df, feature="weighted_nct_chromaticity")
    #
    # p = plot_multiple_full_gpr_models(model_outputs_list=[weighted_r, weighted_ct, weighted_nct],
    #                                   hue_by=None)
    # p.savefig("figs/weighted_full_gpr.pdf")

    # hue = assign_piece_coloring(result_df, tuples_list=[("corelli", "op04n07c"),
    #                                                     ("schubert_dances", "D041trio06b"),
    #                                                     ("liszt_pelerinage", "160.03_Pastorale")])

    # hue = assign_piece_coloring(result_df, tuples_list=[("schubert_dances", "D041trio05b"),
    #                                                     ("schubert_winterreise", "n24"),
    #                                                     ("grieg_lyric_pieces", "op47n04"),
    #                                                     ("poulenc_mouvements_perpetuels", "01_assez_modere")])

    # greig_df = result_df[result_df["corpus"]=="grieg_lyric_pieces"]
    # greig_df_long = pd.melt(greig_df, id_vars=["piece"],
    #                         value_vars=["root_fifths_range", "ct_fifths_range", "nct_fifths_range"],
    #                         var_name="fifths_range_type", value_name="fifths_range_value")
    # # plot barplot
    #
    # bars = seaborn.barplot(greig_df_long, x="piece", y="fifths_range_value", hue="fifths_range_type")
    # bars.set_xticklabels(bars.get_xticklabels(), rotation=90)
    # plt.show()

    # corpus_df = pd.read_csv("data/corpuslevel_chromaticities.tsv", sep="\t")
    # # hue_by = result_df["corpus"].to_numpy().astype(int).reshape((-1, 1))
    #
    # result_df["corpus_id"] = pd.factorize(result_df["corpus"])[0] + 1
    # hue_by = result_df["corpus_id"].to_numpy().astype(int).reshape((-1, 1))

    # plotly_3features_scatter(result_df,
    #                          features=["mean_r_chromaticity", "mean_ct_chromaticity", "mean_nct_chromaticity"],
    #                          hue_by=hue_by)
    # plt.show()

    #
    #
    mean_r_chromaticity = model_outputs(df=result_df, feature="mean_r_chromaticity")
    mean_ct_chromaticity = model_outputs(df=result_df, feature="mean_ct_chromaticity")
    mean_nct_chromaticity = model_outputs(df=result_df, feature="mean_nct_chromaticity")
    #
    # p = plot_multiple_full_gpr_models(
    #     model_outputs_list=[mean_r_chromaticity, mean_ct_chromaticity, mean_nct_chromaticity],
    #     hue_by=None)
    # p.savefig("figs/chromaticity_full_gpr.pdf")

    # plotly_ct_nct(result_df)
    # plotly_3features_scatter(result_df, features=["mean_r_chromaticity", "mean_ct_chromaticity", "mean_nct_chromaticity"],
    #                          hue_by=result_df["corpus_id"].tolist())

    # root_5th = gpr_model_outputs(result_df, feature="root_fifths_range")
    # ct_5th = gpr_model_outputs(result_df, feature="ct_fifths_range")
    # nct_5th = gpr_model_outputs(result_df, feature="nct_fifths_range")
    #
    #
    # p = plot_fifths_range([root_5th, ct_5th, nct_5th], hue_by=None, mean_color=["#FF2C00", "#0C5DA5", "#00B945"])
    # p.savefig("figs/fifths_range.pdf")

    # r_chromaticity = gpr_model_outputs(result_df, feature="mean_r_chromaticity")
    # ct_chromaticity = gpr_model_outputs(result_df, feature="mean_ct_chromaticity")
    # nct_chromaticity = gpr_model_outputs(result_df, feature="mean_nct_chromaticity")
    #
    # p=plot_gpr_fmeans_derivatives(model_outputs_list=[r_chromaticity, ct_chromaticity, nct_chromaticity],
    #                             color=["#FF2C00", "#0C5DA5", "#00B945"],
    #                             labels=[r"$\frac{dRC_p}{dt}$", r"$\frac{dCTC_p}{dt}$", r"$\frac{dNCTC_p}{dt}$"])

    # p.savefig("figs/gpr_fmeans_first_derivatives.pdf")
    # plotly_feature_gpr(result_df, r_chromaticity)
    # plotly_feature_gpr(result_df, ct_chromaticity)
    # plotly_feature_gpr(result_df, nct_chromaticity)

    # plotly_feature_gpr(result_df, ct_chromaticity)
    # plotly_feature_gpr(result_df, nct_chromaticity)

    # p=plot_multiple_full_gpr_models(hue_by=hue_by, model_outputs_list=[r_chromaticity, ct_chromaticity, nct_chromaticity])
    # p.savefig("figs/chromaticity_full_gpr.svg")
    # p.savefig("figs/chromaticity_full_gpr.pdf")

    # # GPR prediction f-mean
    # p = plot_gpr_fmeans([r_chromaticity, ct_chromaticity, nct_chromaticity],
    #                            color=["#FF2C00", "#0C5DA5", "#00B945"],
    #                            labels=[r"$RC_p$", r"$CTC_p$", r"$NCTC_p$"])
    # p.savefig("figs/gpr_fmeans.pdf")

    # f = plotly_3features_scatter(result_df,
    #                              features=["mean_r_chromaticity", "mean_ct_chromaticity", "mean_nct_chromaticity"],
    #                              hue_by=result_df["corpus_id"].to_numpy())
    # f.write_html("figs/piece_level_3D_scatter.html")

    # plotly_feature_gpr(result_df, mean_root2lk)
    p = plot_multiple_full_gpr_models(
        model_outputs_list=[mean_r_chromaticity, mean_ct_chromaticity, mean_nct_chromaticity], hue_by=None, mean_color=["#FF2C00", "#0C5DA5", "#00B945"])

    p.savefig(fname="figs/gpr_full_results.pdf")
    # plot_gpr_fmeans(model_outputs_list=[mean_root2lk, mean_ct2lk, mean_nct2lk], color=["red", "blue", "green"]).savefig("figs/gpr_means.pdf")
