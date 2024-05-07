import os
import pickle
from contextlib import redirect_stdout
from typing import Literal, Optional, Tuple, List

import numpy as np
import gpflow as gf
import pandas as pd
from gpflow.utilities import print_summary
from seaborn import color_palette
from tensorflow import Tensor

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Code.analysis import get_piece_df_by_localkey_mode
from Code.utils.auxiliary import create_results_folder, map_array_to_colors, rand_jitter, Fabian_periods, \
    Johannes_periods, mean_var_after_log, median_CI_after_log, color_palette5, color_palette4
from Code.utils.util import load_file_as_df
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot, show, grid, xlabel, ylabel, title, figure
import plotly.express as px
import seaborn as sns

# MODEL_OUTPUT type: (model, (fmean, fvar, ymean, yvar), f_samples, lengthscale, feature_index)
MODEL_OUTPUT = Tuple[gf.models.gpr.GPR, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None, np.ndarray, str]


# %% general GPR model
def gpr_model(X: np.ndarray, Y: np.ndarray,
              kernel: gf.kernels,
              optimize: bool) -> gf.models.gpr.GPR:
    # new model: modelling the log-precipitation (using logY for the model and later convert it back
    # from log-precipitation into precipitation space)

    epsilon = 1e-6  # add a small epsilon to avoid inf at 0
    logY = np.log(Y + epsilon)

    # old model
    m = gf.models.gpr.GPR(
        data=(X, logY),
        kernel=kernel,
    )

    if optimize:
        gf.optimizers.Scipy().minimize(
            closure=m.training_loss,
            variables=m.trainable_variables,
            track_loss_history=True
        )

    return m


def gpr_model_outputs(df: pd.DataFrame,
                      model_name: str,
                      feature_index: Literal["WLC", "OLC", "avg_WLD", "WL_5th_range", "OL_5th_range"],
                      lengthscale: Optional[float],
                      sample: Optional[int],
                      repo_dir: str) -> MODEL_OUTPUT:
    # normalize the
    X = df["piece_year"].to_numpy().squeeze().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X[:, 0]), max(X[:, 0]) + 1).reshape((-1, 1))
    Y = df[feature_index].to_numpy().squeeze().astype(float).reshape((-1, 1))

    # set kernel length scale, by default, we use the optimization.
    if isinstance(lengthscale, int):
        k = gf.kernels.SquaredExponential(lengthscales=lengthscale)
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=False)
    else:
        k = gf.kernels.SquaredExponential()
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=True)

    f_mean, f_var = m.predict_f(Xplot, full_cov=False)
    y_mean, y_var = m.predict_y(Xplot)

    if isinstance(sample, int):
        samples = m.predict_f_samples(Xplot, sample)
    elif sample is None:
        samples = None
    else:
        raise TypeError

    ls = k.lengthscales.numpy()

    analysis_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    # save model outputs:
    model_outputs = (m, (f_mean, f_var, y_mean, y_var), samples, ls, feature_index)
    model_out_path = f'{analysis_dir}models/'
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    pickle_file_path = f'{model_out_path}{model_name}_ModelOutputs.pickle'
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(model_outputs, f)

    # save model params:
    params_file_path = f'{model_out_path}{model_name}_Params.txt'
    with open(params_file_path, 'a') as f:
        with redirect_stdout(f):
            f.write(f'{model_name}\n')
            print_summary(m)
            f.write('\n')  # Add a newline for separation

    return model_outputs


# %% [axs] model plotting functions

def ax_scatter_observations(ax: Axes,
                            X: np.ndarray, Y: np.ndarray,
                            hue_by: Optional[np.ndarray],
                            scatter_colormap: Optional[str | List[str]],
                            jitter: bool = True
                            ) -> Axes:
    X = X.squeeze(axis=-1)
    Y = Y.squeeze(axis=-1)

    if hue_by is None:
        color = "gray"
    elif isinstance(hue_by, str):
        color = hue_by

    elif isinstance(hue_by, np.ndarray):
        if scatter_colormap:
            color = map_array_to_colors(arr=hue_by, color_map=scatter_colormap)

        else:
            raise ValueError
    else:
        raise TypeError

    if jitter:
        # only add jitter only on the x-axis
        ax.scatter(rand_jitter(X, scale=0.01), Y, c=color, s=12, alpha=0.3
                   , label="Observations"
                   )
    else:
        ax.scatter(X, Y, c=color, s=12, alpha=0.3
                   , label="Observations"
                   )

    return ax


def ax_regplot_observations(ax: Axes,
                            X: np.ndarray, Y: np.ndarray):
    """
    Least squares polynomial fit
    """
    reshaped_X = X.flatten()
    reshaped_Y = Y.flatten()

    b, a = np.polyfit(reshaped_X, reshaped_Y, deg=1)
    Xplot = np.linspace(min(reshaped_X) - 10, max(reshaped_X) + 10)
    # Xplot = np.arange(min(X)-10, max(X) + 10).reshape((-1, 1))
    ax.plot(Xplot, a + b * Xplot, color="black", lw=1, linestyle=':')
    return ax


def ax_gpr_prediction(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
                      prediction_stat: Literal["mean", "median"],
                      fmean_color: Optional[str],
                      plot_samples: int | None,
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool
                      ) -> Axes:
    """
    plot the regression mean line f and the std
    """
    # unpack model outputs:
    _, (f_mean, f_var, y_mean, y_var), f_samples, lengthscale, modeled_feature = m_outputs

    # transform back the precipitation space

    if prediction_stat == "mean":
        exp_f, exp_f_var = mean_var_after_log(mu=np.array(f_mean), var=np.array(f_var))
        exp_y, exp_y_var = mean_var_after_log(mu=np.array(y_mean), var=np.array(y_var))

        f_lower = exp_f - 1.96 * np.sqrt(exp_f_var)
        f_upper = exp_f + 1.96 * np.sqrt(exp_f_var)
        y_lower = exp_y - 1.96 * np.sqrt(exp_y_var)
        y_upper = exp_y + 1.96 * np.sqrt(exp_y_var)

        exp_f_samples, _ = mean_var_after_log(mu=np.array(f_samples), var=np.array(f_var))

    else:
        exp_f, (f_lower, f_upper) = median_CI_after_log(mu=np.array(f_mean), var=np.array(f_var))
        exp_y, (y_lower, y_upper) = median_CI_after_log(mu=np.array(y_mean), var=np.array(y_var))
        exp_f_samples, _ = median_CI_after_log(mu=np.array(f_samples), var=np.array(f_var))

    X = m_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))

    if isinstance(fmean_color, str):
        fmean_color = fmean_color
    else:
        fmean_color = "black"

    ax.plot(Xplot, exp_f, "-", color=fmean_color, alpha=0.7, label=f"f {prediction_stat}({modeled_feature})", linewidth=2)
    ax.set_ylim(0, 3)

    if plot_f_uncertainty:
        fvar_color = 'gray'
        ax.plot(Xplot, f_lower, "--", color=fvar_color, label="f 95% confidence", alpha=0.35)
        ax.plot(Xplot, f_upper, "--", color=fvar_color, alpha=0.35)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fvar_color, alpha=0.35
        )
    if plot_y_uncertainty:
        ax.plot(Xplot, y_lower, "-", color=fmean_color, label="Y 95% confidence", linewidth=1, alpha=0.5)
        ax.plot(Xplot, y_upper, "-", color=fmean_color, linewidth=1, alpha=0.5)
        ax.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=fmean_color, alpha=0.2
        )
    if plot_samples:
        ax.plot(Xplot, exp_f_samples[:, :, 0].numpy().T, 'darkgray', linewidth=0.5, alpha=0.6)

    ax.legend(loc="upper left")
    return ax


def ax_full_gpr_model_doubleyticks(ax: Axes,
                                   m_outputs: MODEL_OUTPUT,
                                   prediction_stat: Literal["mean", "median"],
                                   ax_title: str,
                                   fmean_color: Optional[str],
                                   # fvar_color: Optional[str],
                                   plot_samples: int | None,
                                   plot_f_uncertainty: bool,
                                   plot_y_uncertainty: bool,
                                   scatter_colormap: Optional[str | List[str]],
                                   scatter_hue_by: Optional[np.ndarray],
                                   scatter_jitter: bool,
                                   show_second_yticks: bool
                                   ):
    """
    plot the combined scatter ax and the gpr prediction ax
    """
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])
    expY = np.exp(Y)  # convert back to the precipitation space (before log)

    ax.set_title(ax_title)
    ax_scatter_observations(ax=ax, X=X, Y=expY, hue_by=scatter_hue_by,
                            jitter=scatter_jitter,
                            scatter_colormap=scatter_colormap)
    ax2 = ax.twinx()
    ax_gpr_prediction(ax=ax2, m_outputs=m_outputs, fmean_color=fmean_color,
                      prediction_stat=prediction_stat,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)

    # produce a legend with the unique colors from the scatter
    ax.legend(title=r"$\lambda$={:.1f}".format(m_outputs[-2]),
              loc="upper left")

    ax.tick_params(axis='y', labelcolor="#003153")

    ax2.legend(loc="upper right", labelcolor=fmean_color)

    if show_second_yticks:
        ax2.tick_params(axis='y', labelcolor=fmean_color)
    else:
        ax2.set_yticks([])


def ax_full_gpr_model(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
                      prediction_stat: Literal["mean", "median"],
                      ax_title: str,
                      fmean_color: Optional[str],
                      plot_samples: int | None,
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool,
                      scatter_colormap: Optional[str | List[str]],
                      scatter_hue_by: Optional[np.ndarray],
                      scatter_jitter: bool,
                      ylim: Optional[Tuple[int, int]],
                      legend_loc: Optional[str]):
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])
    expY = np.exp(Y)  # convert back to the precipitation space (before log)

    ax.set_title(ax_title)
    ax_scatter_observations(ax=ax, X=X, Y=expY, hue_by=scatter_hue_by,
                            jitter=scatter_jitter,
                            scatter_colormap=scatter_colormap)

    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color,
                      prediction_stat=prediction_stat,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)
    ax_regplot_observations(ax=ax, X=X, Y=expY)

    if ylim:
        ax.set_ylim([ylim[0], ylim[1]])
    if legend_loc:
        ax.legend(title=r"$\lambda$={:.1f}".format(m_outputs[-2]), loc=legend_loc)
    else:
        ax.get_legend().remove()


# %% GPR models for chromaticity plots

def plot_gpr_chromaticities_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
                                    prediction_stat: Literal["mean", "median"],
                                    era_division: Literal["Fabian", "Johannes"],
                                    lengthscale: Optional[float],
                                    ylim: Optional[Tuple[int, int]],
                                    plot_samples: int | None,
                                    plot_f_uncertainty: bool,
                                    plot_y_uncertainty: bool,
                                    repo_dir: str,
                                    fname_anno: Optional[str],
                                    save: bool = False
                                    ):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    # computing the models:

    major_wlc = gpr_model_outputs(df=major_df, model_name="WLC(major)", repo_dir=repo_dir,
                                  feature_index="WLC", lengthscale=lengthscale, sample=plot_samples)
    major_olc = gpr_model_outputs(df=major_df, model_name="OLC(major)", repo_dir=repo_dir,
                                  feature_index="OLC", lengthscale=lengthscale, sample=plot_samples)

    minor_wlc = gpr_model_outputs(df=minor_df, model_name="WLC(minor)", repo_dir=repo_dir,
                                  feature_index="WLC", lengthscale=lengthscale, sample=plot_samples)
    minor_olc = gpr_model_outputs(df=minor_df, model_name="OLC(minor)", repo_dir=repo_dir,
                                  feature_index="OLC", lengthscale=lengthscale, sample=plot_samples)

    # plotting:
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(9.5, 6), sharex=True, sharey=True,
                            layout="constrained")

    wlc_fmean_color = '#443C68'
    olc_fmean_color = '#1B4242'
    maj_min_palette = 'husl'


    major_corpora_col = major_df["corpus_id"].to_numpy()
    num_major_corpora = major_df["corpus_id"].unique().shape[0]
    major_scatter_color = color_palette(maj_min_palette, n_colors=num_major_corpora)


    # major wlc:

    ax_full_gpr_model(ax=axs[0, 0],
                      ax_title="WLC (major)",
                      m_outputs=major_wlc,
                      prediction_stat=prediction_stat,
                      fmean_color=wlc_fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      legend_loc=None)
    # major olc:
    ax_full_gpr_model(ax=axs[1, 0],
                      ax_title="OLC (major)",
                      m_outputs=major_olc,
                      prediction_stat=prediction_stat,
                      fmean_color=olc_fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      legend_loc=None)

    minor_corpora_col = minor_df["corpus_id"].to_numpy()
    num_minor_corpora = minor_df["corpus_id"].unique().shape[0]
    minor_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_minor_corpora)

    # minor wlc:
    ax_full_gpr_model(ax=axs[0, 1],
                      ax_title="WLC (minor)",
                      m_outputs=minor_wlc,
                      prediction_stat=prediction_stat,
                      fmean_color=wlc_fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_colormap=minor_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      legend_loc=None)

    # minor olc:
    ax_full_gpr_model(ax=axs[1, 1],
                      ax_title="OLC (minor)",
                      m_outputs=minor_olc,
                      prediction_stat=prediction_stat,
                      fmean_color=olc_fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_jitter=True,
                      scatter_colormap=minor_scatter_color,
                      ylim=ylim,
                      legend_loc=None)

    # add vertical-lines to era boundaries
    for ax in axs.flatten():
        if era_division == "Fabian":
            div_yrs = [1592, 1662, 1761, 1820, 1869]
            era_str = ["Renaissance", "Baroque", "Classical", "E.Rom", "L.Rom"]
            for i, x in enumerate(div_yrs):
                if i != 0:
                    ax.axvline(x=x, ymin=0.93, ymax=1, c="black", lw=0.5)
                if i == 0:
                    ax.text(x=x, y=4.7, s=era_str[i], fontsize="x-small")
                elif i == 1:
                    ax.text(x=x + 25, y=4.7, s=era_str[i], fontsize="x-small")
                else:
                    ax.text(x=x + 10, y=4.7, s=era_str[i], fontsize="x-small")


    fig.supylabel("Chromaticity", fontweight="bold")
    fig.supxlabel("Year", fontweight="bold")

    fig_path = f'{result_dir}figs/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if fname_anno:
        fname_anno = f'_{fname_anno}'
    else:
        fname_anno = ""
    if save:
        plt.savefig(f'{fig_path}gpr_chrom_{prediction_stat}_{era_division}{fname_anno}.pdf', dpi=200)
    plt.show()
# %% GPR models for dissonance

def plot_gpr_dissonance(df: pd.DataFrame,
                        prediction_stat: Literal["mean", "median"],
                        era_division: Literal["Fabian", "Johannes"],
                        lengthscale: Optional[float],
                        ylim: Optional[Tuple[float, float]],
                        plot_samples: int | None,
                        plot_f_uncertainty: bool,
                        plot_y_uncertainty: bool,
                        repo_dir: str,
                        fname_anno: Optional[str],
                        save: bool = False
                        ):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    major_df = get_piece_df_by_localkey_mode(df=df, mode="major")
    minor_df = get_piece_df_by_localkey_mode(df=df, mode="minor")

    # computing the models:

    major_WLD = gpr_model_outputs(df=major_df, model_name="WLD(major)", repo_dir=repo_dir,
                                  feature_index="avg_WLD", lengthscale=lengthscale, sample=plot_samples)

    minor_WLD = gpr_model_outputs(df=minor_df, model_name="WLD(minor)", repo_dir=repo_dir,
                                  feature_index="avg_WLD", lengthscale=lengthscale, sample=plot_samples)

    # plotting:
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 3.5), sharex=True, sharey=True,
                            layout="constrained")
    fmean_color = "#750E21"

    maj_min_palette = 'husl'

    # major wld:
    major_corpora_col = major_df["corpus_id"].to_numpy()
    num_major_corpora = major_df["corpus_id"].unique().shape[0]
    major_scatter_color = color_palette(maj_min_palette, n_colors=num_major_corpora)

    ax_full_gpr_model(ax=axs[0],
                      ax_title="WLD (major)",
                      m_outputs=major_WLD,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      # show_second_yticks=False
                      ylim=ylim,
                      legend_loc=None)

    # minor wld:
    minor_corpora_col = minor_df["corpus_id"].to_numpy()
    num_minor_corpora = minor_df["corpus_id"].unique().shape[0]
    minor_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_minor_corpora)

    ax_full_gpr_model(ax=axs[1],
                      ax_title="WLD (minor)",
                      m_outputs=minor_WLD,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_colormap=minor_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      legend_loc=None
                      )

    # add vertical-lines to era boundaries
    for ax in axs:
        if era_division == "Fabian":
            div_yrs = [1592, 1662, 1761, 1820, 1869]
            era_str = ["Renaissance", "Baroque", "Classical", "E.Rom", "L.Rom"]
            for i, x in enumerate(div_yrs):
                if i != 0:
                    ax.axvline(x=x, ymax=.05, c="black", lw=0.5)
                if i == 0:
                    ax.text(x=x, y=.02, s=era_str[i], fontsize="x-small")
                elif i == 1:
                    ax.text(x=x + 25, y=.02, s=era_str[i], fontsize="x-small")
                else:
                    ax.text(x=x + 10, y=.02, s=era_str[i], fontsize="x-small")

    fig.supylabel("Dissonance", fontweight="bold")
    fig.supxlabel("Year", fontweight="bold")


    fig_path = f'{result_dir}figs/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if fname_anno:
        fname_anno = f'_{fname_anno}'
    else:
        fname_anno = ""

    if save:
        plt.savefig(f'{fig_path}gpr_diss_{prediction_stat}_{era_division}{fname_anno}.pdf', dpi=200)

    plt.show()


def plotly_gpr(df: pd.DataFrame, mode: Literal["major", "minor"],
               index_type: Literal["WLC", "OLC", "WLD"]):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)

    fig = px.scatter(mode_df, x="piece_year", y=index_type, color="corpus",
                     hover_data=['corpus', 'corpus_year', 'piece', 'piece_year', 'WLC'])

    fig_path = f'{result_dir}plotly/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig.write_html(f'{fig_path}GPR_{index_type}_{mode}.html')


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo_dir = f'{user}/Codes/chromaticism-codes/'

    print(f'Loading dfs ...')
    piece_indices = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/piece_level_indices_by_mode.pickle")

    dissoance_piece_by_mode = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/dissonance_piece_by_mode.pickle")
    chromaticity_piece_major = load_file_as_df(f'{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_major.pickle')
    chromaticity_piece_minor = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_minor.pickle")

    print(f'Plotly figures ...')
    plotly_gpr(df=piece_indices, index_type="WLC", mode="major")
    plotly_gpr(df=piece_indices, index_type="WLC", mode="minor")
    plotly_gpr(df=piece_indices, index_type="OLC", mode="major")
    plotly_gpr(df=piece_indices, index_type="OLC", mode="minor")
    plotly_gpr(df=piece_indices, index_type="WLD", mode="major")
    plotly_gpr(df=piece_indices, index_type="WLD", mode="minor")

    print(f'GPR for dissonance ...')
    plot_gpr_dissonance(df=dissoance_piece_by_mode,
                        prediction_stat="mean",
                        era_division="Fabian", lengthscale=20,
                        plot_samples=False,
                        plot_y_uncertainty=False,
                        plot_f_uncertainty=True,
                        repo_dir=repo_dir,
                        ylim=(0, 1),
                        fname_anno=None,
                        save=True)

    print(f'GPR for chromaticity ...')
    plot_gpr_chromaticities_by_mode(major_df=chromaticity_piece_major, minor_df=chromaticity_piece_minor,
                                    prediction_stat="mean",
                                    era_division="Fabian", lengthscale=20,
                                    plot_samples=False,
                                    plot_y_uncertainty=False,
                                    plot_f_uncertainty=True,
                                    repo_dir=repo_dir,
                                    ylim=(0, 5),
                                    fname_anno="capped", save=True)
