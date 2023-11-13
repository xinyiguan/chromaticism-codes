from typing import Optional, List, Literal

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import numpy as np

from gpr_analysis import gpr_model_outputs
from gpr_analysis import MODEL_OUTPUT
from utils.util import rand_jitter


# GPR plotting ______________________________________________________________________
def ax_observations_scatter(ax: matplotlib.axes.Axes,
                            X: np.ndarray, Y: np.ndarray,
                            hue_by: np.ndarray | None,
                            scatter_colormap: str = None,
                            with_jitter: bool = True) -> matplotlib.axes.Axes:
    X = X.squeeze(axis=-1)
    Y = Y.squeeze(axis=-1)

    def map_array_to_colors(arr, color_map: str | None):
        unique_values = np.unique(arr)
        num_unique_values = len(unique_values)

        # Define the colormap using the recommended method
        # cmap = mcolors.ListedColormap(cm.Dark2.colors[:num_unique_values])
        cmap = matplotlib.colormaps[color_map]

        # Create a dictionary to map unique values to colors
        value_to_color = dict(zip(unique_values, cmap.colors))

        # Map the values in the array to colors, using "gray" for 0 values
        color_list = [value_to_color[val] if val != 0 else "gray" for val in arr]

        return color_list

    if hue_by is None:
        color = "gray"
        # alpha = 0.4
    elif isinstance(hue_by, str):
        color = hue_by
        # alpha = 0.4

    elif isinstance(hue_by, np.ndarray):
        if scatter_colormap:
            color = map_array_to_colors(hue_by, scatter_colormap)
            # alpha = [0.4 if col == "gray" else 0.4 for col in color]
        else:
            raise ValueError
    else:
        raise TypeError

    # adding jitter:
    if with_jitter:
        # only add jitter on the x-axis not y-axis
        ax.scatter(rand_jitter(X), Y, c=color, s=20, label="Observations", alpha=0.4)
    else:
        ax.scatter(X, Y, c=color, s=20, label="Observations", alpha=0.4)

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

    ax.plot(Xplot, f_mean, "-", color=fmean_color, label=f"{feature} GPR prediction", linewidth=2.5)

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
                      scatter_colormap: str = None,
                      plot_f_uncertainty: bool = True,
                      plot_y_uncertainty: bool = False
                      ) -> matplotlib.axes.Axes:
    """
    plot the scatter ax and the gpr prediction ax
    """
    feature = m_outputs[3]
    # ax.set_title(f'{feature}\n', fontsize=12)
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])

    ax_observations_scatter(ax=ax, X=X, Y=Y, hue_by=hue_by, with_jitter=True, scatter_colormap=scatter_colormap)
    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color, fvar_color=fvar_color,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)
    return ax


def plot_gpr_fifths_range(model_outputs_list: List[MODEL_OUTPUT],
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


def plot_gpr_chromaticity(model_outputs_list: List[MODEL_OUTPUT],
                          hue_by: np.ndarray | str | List[str],
                          mean_colors: List,
                          add_era_division: bool=False,
                          scatter_colormap: List = None
                          ) -> plt.Figure:
    num_features = len(model_outputs_list)

    fig = plt.figure(figsize=(9, 5 * num_features), layout='constrained')

    ax_index = 1
    for i, out in enumerate(model_outputs_list):
        ax = fig.add_subplot(num_features, 1, ax_index)
        if isinstance(hue_by, str):
            color = hue_by
        elif isinstance(hue_by, List):
            color = hue_by[i]
        elif isinstance(hue_by, np.ndarray):
            color = hue_by
            ax_full_gpr_model(ax=ax, hue_by=color,
                              m_outputs=model_outputs_list[i], fmean_color=mean_colors[i],
                              fvar_color=mean_colors[i], plot_samples=False, scatter_colormap=scatter_colormap[i])
        else:
            raise TypeError
        ax_full_gpr_model(ax=ax, hue_by=color,
                          m_outputs=model_outputs_list[i], fmean_color=mean_colors[i],
                          fvar_color=mean_colors[i], plot_samples=False)

        ax.axhline(0, c="gray", linestyle="--", lw=1)
        if add_era_division:
            ax.axvline(1662, c="lightblue", linestyle="dotted", lw=1)
            ax.axvline(1761, c="lightblue", linestyle="dotted", lw=1)
            ax.axvline(1820, c="lightblue", linestyle="dotted", lw=1)
            ax.axvline(1869, c="lightblue", linestyle="dotted", lw=1)

        ax_index += 1

    # fig.legend(loc='upper left')
    # fig.tight_layout()
    return fig


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    beethoven_df = pd.read_csv("data/beethoven_chromaticity.tsv", sep="\t")

    r_fifths_range = gpr_model_outputs(df=result_df, feature="r_fifths_range")
    ct_fifths_range = gpr_model_outputs(df=result_df, feature="ct_fifths_range")
    nct_fifths_range = gpr_model_outputs(df=result_df, feature="nct_fifths_range")

    fifths_range = plot_gpr_fifths_range([r_fifths_range, ct_fifths_range, nct_fifths_range],
                                         hue_by=None,
                                         mean_color=["#0b5572", "#ddb455", "#6F4E7C"])

    fifths_range.savefig(fname="figs/Figure_gpr_fifths_range.pdf")

    rc = gpr_model_outputs(df=result_df, feature="RC")
    ctc = gpr_model_outputs(df=result_df, feature="CTC")
    nctc = gpr_model_outputs(df=result_df, feature="NCTC")

    chromaticities = plot_gpr_chromaticity(model_outputs_list=[rc, ctc, nctc],
                                           # mean_colors=["#6a1215", "#106a96", "#db7134"],
                                           # hue_by=["#ab1d22", "#108b96", "#db9b34"]
                                           mean_colors=["#0b5572", "#C6A24C", "#6F4E7C"],
                                           hue_by=["#57a1be", "#E7CA88", "#A894B0"]
                                           )
    chromaticities.savefig(fname="figs/Figure_gpr_chromaticities.pdf")

    beethoven_rc = gpr_model_outputs(df=beethoven_df, feature="RC")
    beethoven_ctc = gpr_model_outputs(df=beethoven_df, feature="CTC")
    beethoven_nctc = gpr_model_outputs(df=beethoven_df, feature="NCTC")
    beethoven_chromaticities = plot_gpr_chromaticity(model_outputs_list=[beethoven_rc, beethoven_ctc, beethoven_nctc],
                                                     mean_colors=["#0b5572", "#C6A24C", "#6F4E7C"],
                                                     hue_by=["#57a1be", "#E7CA88", "#A894B0"]
                                                     )
    beethoven_chromaticities.savefig(fname="figs/Figure_gpr_beethoven_chromaticities.pdf")
