import os
import pickle
from contextlib import redirect_stdout
from typing import Literal, Optional, Tuple, List

import numpy as np
import gpflow as gf
import pandas as pd
from gpflow.utilities import print_summary
from tensorflow import Tensor

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Code.analysis import get_piece_df_by_localkey_mode
from Code.utils.auxiliary import create_results_folder, map_array_to_colors, rand_jitter, mean_var_after_log, \
    median_CI_after_log, add_period_text_to_ax
from Code.utils.util import load_file_as_df, corpora_colors
import plotly.express as px
import seaborn as sns

# MODEL_OUTPUT type: (model, (fmean, fvar, ymean, yvar), f_samples, lengthscale, feature_index)
MODEL_OUTPUT = Tuple[
    gf.models.gpr.GPR, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tensor | None, np.ndarray, str]

MAJOR_regline_COLOR = "#be0e23"
MINOR_regline_COLOR = "#073E7F"


# %% general GPR model
def gpr_model(X: np.ndarray, Y: np.ndarray,
              kernel: gf.kernels,
              optimize: bool) -> gf.models.gpr.GPR:
    # new model: modelling the log-precipitation (using logY for the model and later convert it back
    # from log-precipitation into precipitation space)

    # epsilon = 1e-6  # add a small epsilon to avoid inf at 0
    # logY = np.log(Y + epsilon)

    # old model
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

    return m


def gpr_model_outputs(df: pd.DataFrame,
                      model_name: str,
                      feature_index: Literal["WLC", "OLC", "avg_WLD", "WL_5th_range", "OL_5th_range", "r_WLC_WLD"],
                      logY: bool,
                      lengthscale: Optional[float],
                      sample: Optional[int],
                      repo_dir: str) -> MODEL_OUTPUT:
    # normalize the
    X = df["piece_year"].to_numpy().squeeze().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X[:, 0]), max(X[:, 0]) + 1).reshape((-1, 1))
    Y = df[feature_index].to_numpy().squeeze().astype(float).reshape((-1, 1))

    if logY:
        # transform to the log space
        epsilon = 1e-6  # add a small epsilon to avoid inf at 0
        Y = np.log(Y + epsilon)

    # set kernel length scale, by default, we use the optimization.
    if isinstance(lengthscale, int):
        k = gf.kernels.SquaredExponential(lengthscales=lengthscale)
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=False)
    else:
        k = gf.kernels.SquaredExponential()
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=True)

    f_mean, f_var = np.array(m.predict_f(Xplot, full_cov=False))
    y_mean, y_var = np.array(m.predict_y(Xplot))

    if logY:
        # transform back the precipitation space
        f_mean, f_var = mean_var_after_log(mu=f_mean, var=f_var)
        y_mean, y_var = mean_var_after_log(mu=y_mean, var=y_var)

    if sample:
        raise NotImplementedError
    else:
        samples = None

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
        ax.scatter(rand_jitter(X, scale=0.01), Y, s=13, alpha=0.65,
                   facecolors='none', edgecolors=color,
                   linewidth=1, label="Observations"
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
    ax.plot(Xplot, a + b * Xplot, color="black", lw=1.5, linestyle=':')
    return ax


def ax_gpr_prediction(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
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

    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    X = m_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))

    if isinstance(fmean_color, str):
        fmean_color = fmean_color
    else:
        fmean_color = "black"

    ax.plot(Xplot, f_mean, "-", color=fmean_color, alpha=0.7, label=f"f mean ({modeled_feature})",
            linewidth=2)
    ax.set_ylim(0, 3)

    if plot_f_uncertainty:
        fvar_color = 'gray'
        ax.plot(Xplot, f_lower, "--", color=fvar_color, label="f 95% confidence", alpha=0.2)
        ax.plot(Xplot, f_upper, "--", color=fvar_color, alpha=0.2)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fvar_color, alpha=0.2
        )
    if plot_y_uncertainty:
        ax.plot(Xplot, y_lower, "-", color=fmean_color, label="Y 95% confidence", linewidth=1, alpha=0.5)
        ax.plot(Xplot, y_upper, "-", color=fmean_color, linewidth=1, alpha=0.5)
        ax.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=fmean_color, alpha=0.2
        )
    if plot_samples:
        raise NotImplementedError
    ax.legend(loc="upper left")
    return ax


def ax_full_gpr_model(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
                      ax_title: str,
                      fmean_color: Optional[str],
                      plot_samples: int | None,
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool,
                      scatter_colormap: Optional[str | List[str]],
                      scatter_hue_by: Optional[np.ndarray],
                      scatter_jitter: bool,
                      ylim: Optional[Tuple[int, int]],
                      xlim: Optional[Tuple[int, int]] = None,
                      legend_loc: Optional[str] = None):
    X = np.array(m_outputs[0].data[0])
    Y = np.array(m_outputs[0].data[1])
    expY = np.exp(Y)  # convert back to the precipitation space (before log)

    ax.set_title(ax_title)
    ax_scatter_observations(ax=ax, X=X, Y=expY, hue_by=scatter_hue_by,
                            jitter=scatter_jitter,
                            scatter_colormap=scatter_colormap)

    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color,
                      plot_samples=plot_samples, plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty)
    ax_regplot_observations(ax=ax, X=X, Y=expY)

    if ylim:
        ax.set_ylim([ylim[0], ylim[1]])
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    if legend_loc:
        ax.legend(title=r"$\lambda$={:.1f}".format(m_outputs[-2]), loc=legend_loc)
    else:
        ax.get_legend().remove()


# %% GPR models for chromaticity plots

def plot_gpr_chromaticities_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
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

    major_ilc = gpr_model_outputs(df=major_df, logY=True, model_name="ILC(major)", repo_dir=repo_dir,
                                  feature_index="WLC", lengthscale=lengthscale, sample=plot_samples)
    major_olc = gpr_model_outputs(df=major_df, logY=True, model_name="OLC(major)", repo_dir=repo_dir,
                                  feature_index="OLC", lengthscale=lengthscale, sample=plot_samples)

    minor_ilc = gpr_model_outputs(df=minor_df, logY=True, model_name="ILC(minor)", repo_dir=repo_dir,
                                  feature_index="WLC", lengthscale=lengthscale, sample=plot_samples)
    minor_olc = gpr_model_outputs(df=minor_df, logY=True, model_name="OLC(minor)", repo_dir=repo_dir,
                                  feature_index="OLC", lengthscale=lengthscale, sample=plot_samples)

    # plotting:
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(9.5, 6), sharex=True, sharey=True,
                            layout="constrained")

    maj_min_palette = 'husl'
    # maj_min_palette = corpora_colors

    major_corpora_col = major_df["corpus_id"].to_numpy()
    num_major_corpora = major_df["corpus_id"].unique().shape[0]
    major_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_major_corpora)

    xlim=(1575, 1950)
    # major ilc:
    ax_full_gpr_model(ax=axs[0, 0],
                      ax_title="ILC (major)",
                      m_outputs=major_ilc,
                      fmean_color=MAJOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None)
    # major olc:
    ax_full_gpr_model(ax=axs[1, 0],
                      ax_title="OLC (major)",
                      m_outputs=major_olc,
                      fmean_color=MAJOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None)

    minor_corpora_col = minor_df["corpus_id"].to_numpy()
    num_minor_corpora = minor_df["corpus_id"].unique().shape[0]
    minor_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_minor_corpora)

    # minor Ilc:
    ax_full_gpr_model(ax=axs[0, 1],
                      ax_title="ILC (minor)",
                      m_outputs=minor_ilc,
                      fmean_color=MINOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_colormap=minor_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None)

    # minor olc:
    ax_full_gpr_model(ax=axs[1, 1],
                      ax_title="OLC (minor)",
                      m_outputs=minor_olc,
                      fmean_color=MINOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_jitter=True,
                      scatter_colormap=minor_scatter_color,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None)

    # add vertical-lines to era boundaries
    for ax in axs.flatten():
        # add_period_text_to_ax(ax=ax, y0=4.7, ymin=0.93, ymax=1)

        div_yrs = [1592, 1662, 1761, 1820, 1869]
        era_str = ["Renaissance", "Baroque", "Classical", "E.Rom", "L.Rom"]
        for i, x in enumerate(div_yrs):
            if i != 0:
                ax.axvline(x=x, ymin=0.92, ymax=1, c="black", lw=0.5)
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
        plt.savefig(f'{fig_path}gpr_chrom{fname_anno}.pdf', dpi=200)
    plt.show()


# %% GPR models for dissonance

def plot_gpr_dissonance(df: pd.DataFrame,
                        prediction_stat: Literal["mean", "median"],
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

    major_WLD = gpr_model_outputs(df=major_df, logY=True, model_name="ILD(major)", repo_dir=repo_dir,
                                  feature_index="avg_WLD", lengthscale=lengthscale, sample=plot_samples)

    minor_WLD = gpr_model_outputs(df=minor_df, logY=True, model_name="ILD(minor)", repo_dir=repo_dir,
                                  feature_index="avg_WLD", lengthscale=lengthscale, sample=plot_samples)

    # plotting:
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 3.5), sharex=True, sharey=True,
                            layout="constrained")
    # fmean_color = "#074f42"

    maj_min_palette = 'husl'
    xlim=(1575, 1950)

    # major wld:
    major_corpora_col = major_df["corpus_id"].to_numpy()
    num_major_corpora = major_df["corpus_id"].unique().shape[0]
    major_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_major_corpora)

    ax_full_gpr_model(ax=axs[0],
                      ax_title="ILD (major)",
                      m_outputs=major_WLD,
                      fmean_color=MAJOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=major_corpora_col,
                      scatter_colormap=major_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None)

    # minor wld:
    minor_corpora_col = minor_df["corpus_id"].to_numpy()
    num_minor_corpora = minor_df["corpus_id"].unique().shape[0]
    minor_scatter_color = sns.color_palette(maj_min_palette, n_colors=num_minor_corpora)

    ax_full_gpr_model(ax=axs[1],
                      ax_title="ILD (minor)",
                      m_outputs=minor_WLD,
                      fmean_color=MINOR_regline_COLOR,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=minor_corpora_col,
                      scatter_colormap=minor_scatter_color,
                      scatter_jitter=True,
                      ylim=ylim,
                      xlim=xlim,
                      legend_loc=None
                      )

    # add vertical-lines to era boundaries
    for ax in axs:
        add_period_text_to_ax(ax=ax, y0=.02)

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
        plt.savefig(f'{fig_path}gpr_diss_{prediction_stat}_{fname_anno}.pdf', dpi=200)

    plt.show()


# %% GPR models for chordleve indices r in piece

def plot_gpr_indices_r(df: pd.DataFrame,
                       mode: Literal["major", "minor"],
                       lengthscale: Optional[float],
                       repo_dir: str,
                       fname_anno: Optional[str],
                       ylim:Optional[Tuple[float, float]],
                       save: bool = False):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)

    # computing the models:

    m_outputs = gpr_model_outputs(df=mode_df, logY=False, model_name="r-ILC-ILD(major)", repo_dir=repo_dir,
                                  feature_index="r_WLC_WLD", lengthscale=lengthscale, sample=False)

    # unpack model outputs:
    _, (f_mean, f_var, y_mean, y_var), f_samples, lengthscale, modeled_feature = m_outputs

    X = m_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))[:, 0]
    f_mean_reshaped = f_mean.reshape((-1, 1))[:, 0]
    f_var_reshaped_low = f_mean_reshaped - 1.96 * np.sqrt(f_var).reshape((-1, 1))[:, 0]
    f_var_reshaped_up = f_mean_reshaped + 1.96 * np.sqrt(f_var).reshape((-1, 1))[:, 0]

    if mode == "major":
        scatter_color = '#E18791'  # RED
        regline_color = MAJOR_regline_COLOR
    else:
        scatter_color = '#83A0BE'  # BLUE
        regline_color = MINOR_regline_COLOR

    # plots
    g = sns.JointGrid(height=6, ylim=ylim)
    x, y = mode_df["piece_year"].to_numpy(), mode_df["r_WLC_WLD"].to_numpy()
    jitter_x = rand_jitter(arr=x)
    sns.scatterplot(x=jitter_x, y=y, ec=scatter_color, fc="none", s=20, linewidth=1.5,
                    ax=g.ax_joint, alpha=0.65)

    # gpr line
    sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_mean_reshaped, color=regline_color)

    # gpr ci region
    sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_var_reshaped_low, color=regline_color,
                 alpha=0.1)
    sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_var_reshaped_up, color=regline_color, alpha=0.1)
    line = g.ax_joint.get_lines()
    g.ax_joint.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(),
                            color=regline_color, alpha=0.1)

    sns.histplot(x=x, fill=True, linewidth=1, ax=g.ax_marg_x, color=scatter_color, alpha=0.6)
    sns.histplot(y=y, fill=True, linewidth=1, ax=g.ax_marg_y, color=scatter_color, alpha=0.6)

    # add era info
    add_period_text_to_ax(ax=g.ax_joint, y0=-0.96, fontsize="small")
    g.set_axis_labels(xlabel="Year", ylabel="r", fontweight="bold", fontsize="large")
    g.fig.set_figwidth(8)

    fig_path = f'{result_dir}figs/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if fname_anno:
        fname_anno = f'_{fname_anno}'
    else:
        fname_anno = ""

    plt.tight_layout()
    if save:
        plt.savefig(f'{fig_path}gpr_ilc_ild_corr_{mode}{fname_anno}.pdf', dpi=200)


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

    print(f'GPR for chromaticity ...')
    plot_gpr_chromaticities_by_mode(major_df=chromaticity_piece_major, minor_df=chromaticity_piece_minor,
                                    lengthscale=20,
                                    plot_samples=False,
                                    plot_y_uncertainty=False,
                                    plot_f_uncertainty=True,
                                    repo_dir=repo_dir,
                                    ylim=(0, 5),
                                    fname_anno="capped", save=True)

    print(f'GPR for dissonance ...')
    plot_gpr_dissonance(df=dissoance_piece_by_mode,
                        prediction_stat="mean",
                        lengthscale=20,
                        plot_samples=False,
                        plot_y_uncertainty=False,
                        plot_f_uncertainty=True,
                        repo_dir=repo_dir,
                        ylim=(0, 1),
                        fname_anno=None,
                        save=True)

    print(f'GPR for ILC-ILD correlation ...')
    r_vals_df = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/chord_indices_r_vals_by_piece.pickle")
    plot_gpr_indices_r(df=r_vals_df, lengthscale=20, repo_dir=repo_dir,
                       save=True, fname_anno=None, mode="major",
                       ylim=(-1, 1.25))
    plot_gpr_indices_r(df=r_vals_df, lengthscale=20, repo_dir=repo_dir,
                       save=True, fname_anno=None, mode="minor",
                       ylim=(-1, 1.25))
