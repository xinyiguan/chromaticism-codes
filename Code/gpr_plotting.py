from typing import Optional, List, Literal

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from gpr_analysis import gpr_model_outputs, MODEL_OUTPUT
from Code.utils.util import rand_jitter


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
    feature = m_outputs[-1]

    ax.plot(Xplot, f_mean, "-", color=fmean_color, label=f"{feature} f mean", linewidth=2.5)

    if plot_f_uncertainty:
        ax.plot(Xplot, f_lower, "--", color=fvar_color, label="f 95% confidence", alpha=0.3)
        ax.plot(Xplot, f_upper, "--", color=fvar_color, alpha=0.3)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fvar_color, alpha=0.3
        )

    if plot_y_uncertainty:
        ax.plot(Xplot, y_lower, "-", color=fvar_color, label="Y 95% confidence", linewidth=1, alpha=0.5)
        ax.plot(Xplot, y_upper, "-", color=fvar_color, linewidth=1, alpha=0.5)
        ax.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=fvar_color, alpha=0.2
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
                      ax_title: str = None,
                      scatter_colormap: str = None,
                      plot_f_uncertainty: bool = True,
                      plot_y_uncertainty: bool = True
                      ) -> matplotlib.axes.Axes:
    """
    plot the scatter ax and the gpr prediction ax
    """
    feature = m_outputs[-1]
    # ax.set_title(f'{feature}\n', fontsize=12)
    ax.set_title(ax_title)
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])

    ax_observations_scatter(ax=ax, X=X, Y=Y, hue_by=hue_by, with_jitter=True, scatter_colormap=scatter_colormap)
    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color, fvar_color=fvar_color,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)
    ax.legend(title=r"$\lambda$={:.3f}".format(m_outputs[-2]), loc="upper left")

    return ax


def plot_gpr_fifths_range(model_outputs_list: List[MODEL_OUTPUT],
                          hue_by: np.ndarray | str | None,
                          mean_color: List,
                          plot_samples: bool = False,
                          add_era_division: bool = False
                          ) -> plt.Figure:
    num_features = len(model_outputs_list)

    # fig = plt.figure(figsize=(8 * num_features, 5))
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5), sharey=True)

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

        if add_era_division:
            ax.axvline(1662, c="gray", linestyle="dotted", lw=1)
            ax.axvline(1761, c="gray", linestyle="dotted", lw=1)
            ax.axvline(1820, c="gray", linestyle="dotted", lw=1)
            ax.axvline(1869, c="gray", linestyle="dotted", lw=1)

        ax.set_ylabel("Fifths range", fontsize=13)
        ax.set_xlabel("Year", fontsize=13)
        # ax.set_xticks(fontsize=12)
        # ax.set_yticks(fontsize=12)

        ax.set_ybound(0, 30)
        ax.legend(loc="upper left")

        # ax_index += 1



    fig.tight_layout()
    return fig


def plot_gpr_chromaticity(model_outputs_list: List[MODEL_OUTPUT],
                          hue_by: np.ndarray | str | List[str],
                          mean_colors: List,
                          add_era_division: bool = False,
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
            raise TypeError(f'{type(hue_by)}')
        ax_full_gpr_model(ax=ax, hue_by=color,
                          m_outputs=model_outputs_list[i], fmean_color=mean_colors[i],
                          fvar_color=mean_colors[i], plot_samples=False)

        ax.axhline(0, c="gray", linestyle="--", lw=1)
        if add_era_division:
            ax.axvline(1662, c="lightgray", linestyle="dotted", lw=1)
            ax.axvline(1761, c="lightgray", linestyle="dotted", lw=1)
            ax.axvline(1820, c="lightgray", linestyle="dotted", lw=1)
            ax.axvline(1869, c="lightgray", linestyle="dotted", lw=1)

        ax.legend(loc="upper left")
        # ax.legend(title=r"$\lambda$={}".format(model_outputs_list[i][-2]))
        # ax.get_legend().set_title(r"$\lambda$={}".format(model_outputs_list[i][-2]))
        ax_index += 1

    # fig.legend(loc='upper left')
    # fig.tight_layout()
    return fig


def experiment_gpr_chromaticity(df: pd.DataFrame, fig_name: str):
    mean_colors_palette = ["#6F4E7C", "#0b5572", "#37604e"]
    scatter_colors_palette = ["#A894B0", "#57a1be", "#91ad70"]

    rc = gpr_model_outputs(df=df, feature="RC", df_type="Combined")
    ctc = gpr_model_outputs(df=df, feature="CTC", df_type="Combined")
    nctc = gpr_model_outputs(df=df, feature="NCTC", df_type="Combined")

    rc_gpr = plot_gpr_chromaticity(model_outputs_list=[rc],
                                   mean_colors=[mean_colors_palette[0]],
                                   hue_by=scatter_colors_palette[0],
                                   add_era_division=True
                                   )
    rc_gpr.savefig(fname=f"figs/Figure_gpr_rc_{fig_name}.pdf")

    ctc_gpr = plot_gpr_chromaticity(model_outputs_list=[ctc],
                                    mean_colors=[mean_colors_palette[1]],
                                    hue_by=scatter_colors_palette[1],
                                    add_era_division=True
                                    )
    ctc_gpr.savefig(fname=f"figs/Figure_gpr_ctc_{fig_name}.pdf")

    nctc_gpr = plot_gpr_chromaticity(model_outputs_list=[nctc],
                                     mean_colors=[mean_colors_palette[2]],
                                     hue_by=scatter_colors_palette[2],
                                     add_era_division=True
                                     )
    nctc_gpr.savefig(fname=f"figs/Figure_gpr_nctc_{fig_name}.pdf")


def experiment_gpr_chromaticities(major: pd.DataFrame, minor: pd.DataFrame,
                                  add_era_division: bool = True,
                                  optimized: bool = True):
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12),
                            layout="constrained")

    # handtune_lengthscale = [15, 15, 15, 15, 15, 15]
    handtune_lengthscale = [10, 10, 10, 10, 10, 10]

    optimized_lengthscale = [None, None, None, None, None, None]

    # mean_colors_palette = ["#6F4E7C", "#0b5572", "#37604e"]
    # scatter_colors_palette = ["#A894B0", "#57a1be", "#91ad70"]
    mean_colors_palette6 = ["#ad0041", "#ff6232", "#ffa94d", "#9cde9e", "#39c5a3", "#0088c1"]
    scatter_colors_palette6 = ["#ad0041", "#ff6232", "#ffa94d", "#9cde9e", "#39c5a3", "#0088c1"]

    if optimized:
        l = optimized_lengthscale
        name = "optimized"
    else:
        l = handtune_lengthscale
        name = "handtuned"

    major_rc = gpr_model_outputs(df=major, feature="RC", df_type="Major", lengthscale=l[0])
    major_ctc = gpr_model_outputs(df=major, feature="CTC", df_type="Major", lengthscale=l[1])
    major_nctc = gpr_model_outputs(df=major, feature="NCTC", df_type="Major", lengthscale=l[2])

    minor_rc = gpr_model_outputs(df=minor, feature="RC", df_type="Minor", lengthscale=l[3])
    minor_ctc = gpr_model_outputs(df=minor, feature="CTC", df_type="Minor", lengthscale=l[4])
    minor_nctc = gpr_model_outputs(df=minor, feature="NCTC", df_type="Minor", lengthscale=l[5])

    # major rc
    ax_full_gpr_model(ax=axs[0, 0],
                      ax_title="RC (major)",
                      m_outputs=major_rc,
                      hue_by=scatter_colors_palette6[0],
                      fmean_color=mean_colors_palette6[0],
                      fvar_color=mean_colors_palette6[0], plot_samples=False)

    # major ctc
    ax_full_gpr_model(ax=axs[1, 0],
                      ax_title="CTC (major)",
                      m_outputs=major_ctc,
                      hue_by=scatter_colors_palette6[1],
                      fmean_color=mean_colors_palette6[1],
                      fvar_color=mean_colors_palette6[1],
                      plot_samples=False)

    # major nctc
    ax_full_gpr_model(ax=axs[2, 0],
                      ax_title="NCTC (major)",
                      m_outputs=major_nctc,
                      hue_by=scatter_colors_palette6[2],
                      fmean_color=mean_colors_palette6[2],
                      fvar_color=mean_colors_palette6[2],
                      plot_samples=False)

    # minor rc
    ax_full_gpr_model(ax=axs[0, 1],
                      ax_title="RC (minor)",
                      m_outputs=minor_rc,
                      hue_by=scatter_colors_palette6[3],
                      fmean_color=mean_colors_palette6[3],
                      fvar_color=mean_colors_palette6[3],
                      plot_samples=False)

    # minor ctc
    ax_full_gpr_model(ax=axs[1, 1], ax_title="CTC (minor)",
                      m_outputs=minor_ctc,
                      hue_by=scatter_colors_palette6[4],
                      fmean_color=mean_colors_palette6[4],
                      fvar_color=mean_colors_palette6[4],
                      plot_samples=False)

    # minor nctc
    ax_full_gpr_model(ax=axs[2, 1], ax_title="NCTC (minor)",
                      m_outputs=minor_nctc,
                      hue_by=scatter_colors_palette6[5],
                      fmean_color=mean_colors_palette6[5],
                      fvar_color=mean_colors_palette6[5],
                      plot_samples=False)

    axs[0, 0].set_ybound(-3, 5)
    axs[0, 1].set_ybound(-3, 5)
    axs[1, 0].set_ybound(-3, 8)
    axs[1, 1].set_ybound(-3, 12)
    axs[2, 0].set_ybound(-5, 20)
    axs[2, 1].set_ybound(-5, 30)

    for i in range(3):
        for j in range(2):
            if add_era_division:
                axs[i, j].axvline(1662, c="gray", linestyle="dotted", lw=1)
                axs[i, j].axvline(1761, c="gray", linestyle="dotted", lw=1)
                axs[i, j].axvline(1820, c="gray", linestyle="dotted", lw=1)
                axs[i, j].axvline(1869, c="gray", linestyle="dotted", lw=1)

    plt.savefig(fname=f"figs/Figure_gpr_MajorMinor_EraDivision_{name}.pdf")

    return fig


def experiment_gpr_5thRange(df: pd.DataFrame, df_type: Literal["CombinedMode", "MajorMode", "MinorMode"]):
    mean_colors_palette = ["C0", "C2", "C4"]
    scatter_colors_palette = ["#A894B0", "#57a1be", "#91ad70"]

    r_fifths_range = gpr_model_outputs(df=df, feature="r_fifths_range", df_type="Combined")
    ct_fifths_range = gpr_model_outputs(df=df, feature="ct_fifths_range", df_type="Combined")
    nct_fifths_range = gpr_model_outputs(df=df, feature="nct_fifths_range", df_type="Combined")

    fifths_range = plot_gpr_fifths_range([r_fifths_range, ct_fifths_range, nct_fifths_range],
                                         hue_by=None,
                                         mean_color=mean_colors_palette,
                                         add_era_division=True)

    fifths_range.savefig(fname=f"figs/Figure_gpr_fifths_range_{df_type}.pdf")


if __name__ == "__main__":
    combined_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    major_df = pd.read_csv("data/majorkey_piece_indices.tsv", sep="\t")
    minor_df = pd.read_csv("data/minorkey_piece_indices.tsv", sep="\t")

    experiment_gpr_chromaticities(major=major_df, minor=minor_df)
    experiment_gpr_chromaticities(major_df, minor_df, optimized=False)
    experiment_gpr_5thRange(combined_df, df_type="CombinedMode")
    experiment_gpr_5thRange(major_df, df_type="MajorMode")
    experiment_gpr_5thRange(minor_df, df_type="MinorMode")
