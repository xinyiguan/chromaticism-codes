from typing import Literal, Optional, List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

from matplotlib import pyplot as plt
from matplotlib import cm
from analysis import get_period_df, correlation, MODEL_OUTPUT, model_outputs
from utils.util import rand_jitter


def piece_distribution_histogram(tsv_path: str = "data/piece_chromaticities.tsv", save_fig: bool = False):
    df = pd.read_csv(tsv_path, sep="\t")

    # global variables
    DPI = 300

    # Create the histogram plot
    sns.histplot(df["piece_year"], kde=True, stat="probability", bins=40,
                 kde_kws={'bw_adjust': 0.6})

    # Get the data from the plot
    lines = plt.gca().get_lines()
    xs, ys = lines[0].get_data()

    # Find the local minima in the histogram
    mininds = []
    a, b = -1, -1
    for i, c in enumerate(ys):
        if a > b and b < c:
            mininds.append(i)
        a, b = b, c

    # Extract the x-values of the minima
    t1, _, t2, t3, t4 = [xs[i] for i in mininds]

    # Add vertical lines at the minima
    for b in [t1, t2, t3, t4]:
        plt.axvline(b, c="gray", ls="--", zorder=-2)

    t1 = round(t1)
    t2 = round(t2)
    t3 = round(t3)
    t4 = round(t4)

    print(f'{t1=}, {t2=}, {t3=}, {t4=} ')

    # Set labels and show the plot
    plt.xlabel("year")
    plt.ylabel("probability")
    if save_fig:
        plt.savefig("figs/Figure_histogram.pdf", dpi=DPI)
        plt.show()
    return t1, t2, t3, t4


def plot_chromaticity_indices_corr(df: pd.DataFrame):
    # rename the col names of CI in the df:
    df = df.rename(columns={'rc': 'RC',
                            'ctc': 'CTC',
                            'nctc': 'NCTC'})

    fig, axs = plt.subplots(3, 5, layout="constrained", figsize=(18, 10))

    colors = ["#580F41", "#173573", "#c4840c"]
    periods = ["renaissance", "baroque", "classical", "early_romantic", "late_romantic"]
    years = ["<1662", "1662-1761", "1761-1820", "1820-1869", ">1869"]

    for i, period in enumerate(periods):
        period_df = get_period_df(df, period)

        # Set column subtitle
        axs[0, i].set_title(years[i], fontweight="bold", size="x-large", family="sans-serif")

        for j, (x_var, y_var) in enumerate([("RC", "CTC"),
                                            ("RC", "NCTC"),
                                            ("CTC", "NCTC")]):
            g = sns.regplot(ax=axs[j, i], data=period_df, x=x_var, y=y_var, color=colors[j])

            r = round(correlation(period_df[x_var], period_df[y_var])["r"].values[0], 3)
            p = correlation(period_df[x_var], period_df[y_var])["p-val"].values[0]

            if p < 0.001:
                p_text = 'p < .001'
            elif 0.001 < p < 0.05:
                p_text = 'p < .05'
            else:
                p_text = f'p = {p:.3f}'

            x_limit = axs[j, i].get_xlim()
            y_limit = axs[j, i].get_ylim()
            x_pos = x_limit[1] - 0.03 * (x_limit[1] - x_limit[0])
            y_pos = y_limit[1] - 0.03 * (y_limit[1] - y_limit[0])

            g.text(x_pos, y_pos, f'r = {r}, {p_text}', fontstyle='italic', ha='right', va='top')

    plt.savefig("figs/Figure_CI_corr.pdf", dpi=300)

    plt.show()


# GPR plotting ______________________________________________________________________
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


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")

    rc = model_outputs(df=result_df, feature="rc")
    ctc = model_outputs(df=result_df, feature="ct")
    nctc = model_outputs(df=result_df, feature="nctc")

    fifths_range = plot_gpr_fifths_range(["rc", "ctc", "nctc"], hue_by=None, mean_color=["#FF2C00", "#0C5DA5", "#00B945"])
    fifths_range.savefig(fname="figs/Figure_gpr_fifths_range.pdf")


    