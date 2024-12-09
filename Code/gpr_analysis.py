import functools
import os
import pickle
from contextlib import redirect_stdout
from typing import Literal, Optional, Tuple, List, Dict

import numpy as np
import gpflow as gf
import pandas as pd
from gpflow import mean_functions
from gpflow.utilities import print_summary
from tensorflow import Tensor

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from Code.analysis import get_piece_df_by_localkey_mode
from Code.utils.auxiliary import create_results_folder, map_array_to_colors, rand_jitter
from Code.utils.util import load_file_as_df
import plotly.express as px
import seaborn as sns

# MODEL_OUTPUT type: (model, (fmean, fvar, ymean, yvar), f_samples, lengthscale, feature_index)
MODEL_OUTPUT = Tuple[
    gf.models.gpr.GPR, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tensor | None, np.ndarray, str]

regline_COLOR = dict(major="#be0e23", minor="#073E7F")
ERA = {"P.Baroque": (1575, 1662),
       "Baroque": (1662, 1761),
       "Classical": (1761, 1820),
       "E.Rom": (1820, 1869),
       "L.Rom": (1869, 1950)}


# %% general GPR model
def gpr_model(X: np.ndarray, Y: np.ndarray,
              kernel: gf.kernels,
              optimize: bool) -> gf.models.gpr.GPR:
    m = gf.models.gpr.GPR(
        data=(X, Y),
        kernel=kernel,
        mean_function=mean_functions.Constant())
    gf.set_trainable(m.mean_function.parameters, True)

    if optimize:
        gf.optimizers.Scipy().minimize(
            closure=m.training_loss,
            variables=m.trainable_variables,
            track_loss_history=True
        )
    else:
        gf.set_trainable(kernel.variance, False)
        gf.set_trainable(kernel.lengthscales, False)

        opt = gf.optimizers.Scipy()
        opt.minimize(
            closure=m.training_loss,
            variables=m.trainable_variables
        )
    return m


def gpr_model_outputs(df: pd.DataFrame,
                      model_name: str,
                      log_transform: bool,
                      feature_index: Literal["ILC", "OLC", "DI", "r_ILC_DI"],
                      repo_dir: str,
                      lengthscale: Optional[float],
                      variance: Optional[float],
                      optimize: bool) -> MODEL_OUTPUT:
    X = df["piece_year"].to_numpy().squeeze().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X[:, 0]), max(X[:, 0]) + 1).reshape((-1, 1))  # for prediction
    Y = df[feature_index].to_numpy().squeeze().astype(float).reshape((-1, 1))

    # if take log-transformation
    if log_transform:
        epsilon = 1e-6  # add a small epsilon to avoid inf at 0
        Y_ = np.log(Y + epsilon)
    else:
        Y_ = Y

    # set kernel length scale
    if optimize:
        assert lengthscale is None and variance is None
        k = gf.kernels.SquaredExponential()
    else:
        k = gf.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance)
    m = gpr_model(X=X, Y=Y_, kernel=k, optimize=optimize)

    f_mean, f_var = np.array(m.predict_f(Xplot, full_cov=False))
    y_mean, y_var = np.array(m.predict_y(Xplot))

    ls = k.lengthscales.numpy()

    analysis_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    # save model outputs:
    model_outputs = (m, (f_mean, f_var, y_mean, y_var), None, ls, feature_index)
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
        ax.scatter(X, Y, c=color, s=13, alpha=0.3, label="Observations")

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
                      log_transform: bool,
                      fmean_color: Optional[str],
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool
                      ) -> Axes:
    """
    plot the regression mean line f and the std
    """
    # unpack model outputs:
    _, (f_mean, f_var, y_mean, y_var), f_samples, lengthscale, modeled_feature = m_outputs

    if log_transform:
        f_lower = np.exp(f_mean - 1.96 * np.sqrt(f_var))
        f_upper = np.exp(f_mean + 1.96 * np.sqrt(f_var))
        y_lower = np.exp(y_mean - 1.96 * np.sqrt(y_var))
        y_upper = np.exp(y_mean + 1.96 * np.sqrt(y_var))
        f_mean = np.exp(f_mean)
    else:
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

    fvar_color = 'gray'
    if plot_f_uncertainty:
        ax.plot(Xplot, f_lower, "--", color=fvar_color, label="f 95% confidence", alpha=0.25)
        ax.plot(Xplot, f_upper, "--", color=fvar_color, alpha=0.25)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fvar_color, alpha=0.25
        )
    if plot_y_uncertainty:
        ax.plot(Xplot, y_lower, "-", color=fvar_color, label="Y 95% confidence", linewidth=1, alpha=0.5)
        ax.plot(Xplot, y_upper, "-", color=fvar_color, linewidth=1, alpha=0.5)
        ax.fill_between(
            Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=fvar_color, alpha=0.1
        )
    ax.legend(loc="upper left")
    return ax


def ax_full_gpr_model(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
                      ax_title: str,
                      fmean_color: Optional[str],
                      log_transform: bool,
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool,
                      scatter_colormap: Optional[str | List[str]],
                      scatter_hue_by: Optional[np.ndarray],
                      scatter_jitter: bool,
                      ylim: Optional[Tuple[int, int]],
                      xlim: Optional[Tuple[int, int]] = None,
                      legend_loc: Optional[str] = None):
    X = np.array(m_outputs[0].data[0])
    if log_transform:
        Y = np.exp(np.array(m_outputs[0].data[1]))

    else:
        Y = np.array(m_outputs[0].data[1])
    ax.set_title(ax_title)
    ax_scatter_observations(ax=ax, X=X, Y=Y, hue_by=scatter_hue_by,
                            jitter=scatter_jitter,
                            scatter_colormap=scatter_colormap)

    ax_gpr_prediction(ax=ax, m_outputs=m_outputs, fmean_color=fmean_color,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      log_transform=log_transform)
    ax_regplot_observations(ax=ax, X=X, Y=Y)

    if ylim:
        ax.set_ylim([ylim[0], ylim[1]])
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    if legend_loc:
        ax.legend(title=r"$\lambda$={:.1f}".format(m_outputs[-2]), loc=legend_loc)
    else:
        ax.get_legend().remove()


def filter_df_by_mode(df: pd.DataFrame, mode: Literal["major", "minor"]) -> pd.DataFrame:
    return df.loc[df['localkey_mode'] == mode]


def corpora_col_df_mode(df: pd.DataFrame, mode: Literal["major", "minor"]) -> np.ndarray:
    return filter_df_by_mode(df, mode)["corpus_id"].to_numpy()


def era_col_df_mode(df: pd.DataFrame, mode: Literal["major", "minor"]) -> np.ndarray:
    return filter_df_by_mode(df, mode)["period"].to_numpy()


def scatter_color_df_mode(df: pd.DataFrame, mode: Literal["major", "minor"]):
    s = ["#61ad65",
         "#9350a1",
         "#b88a3a",
         "#697cd4",
         "#b8475f"]
    return s

    # return sns.color_palette('husl', n_colors=filter_df_by_mode(df, mode)["corpus_id"].unique().shape[0])


def add_era_text_to_ax(ax: Axes,
                       era_dict: Dict,
                       ylim: Tuple[float, float],
                       linelength: float,
                       text_pos: Literal["top", "bottom"]):
    if text_pos == "top":
        text_y_pos = (1 - linelength) * ylim[1]
        div_ymin = 1 - linelength
        div_ymax = 1
    else:
        text_y_pos = (ylim[0] + linelength)
        div_ymin = 0 + linelength
        div_ymax = 0

    for era, (start_year, end_year) in era_dict.items():
        ax.text(x=(start_year + end_year) / 2, s=era, fontsize="x-small", y=text_y_pos,
                horizontalalignment="center", verticalalignment="baseline")
        ax.axvline(x=start_year, ymin=div_ymin, ymax=div_ymax, c="black", lw=0.5)
        ax.axvline(x=end_year, ymin=div_ymin, ymax=div_ymax, c="black", lw=0.5)


# %% GPR models for chromaticity/dissonance plots

Metric = Literal["Chromaticity", "Dissonance"]


def plot_gpr(df: pd.DataFrame,
             metric: Metric,
             log_transform: bool,
             lengthscale: Optional[float],
             variance: Optional[float] | Literal["sample_var"],
             optimize: bool,
             ylim: Optional[Tuple[float, float]],
             fname_anno: Optional[str] = None,
             **plotting_params):
    # save the results to this folder:
    repo_dir = plotting_params.get('repo_dir')
    filtered_params = {k: v for k, v in plotting_params.items() if k not in ["repo_dir", "save"]}

    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)
    gpr_model_outputs_ = functools.partial(gpr_model_outputs, repo_dir=repo_dir,
                                           log_transform=log_transform,
                                           lengthscale=lengthscale,
                                           variance=variance,
                                           optimize=optimize)

    def ax_full_gpr_model_(ax: Axes, m: Literal["major", "minor"],
                           feature_index: Literal["ILC", "OLC", "DI"],
                           var: Optional[float] | Literal["sample_var"]) -> None:
        if var == "sample_var":
            var = compute_var(df=df, mode=m, feature_index=feature_index)
        else:
            var = variance

        ax_full_gpr_model(ax=ax,
                          ax_title=f"{feature_index} ({m})",
                          m_outputs=gpr_model_outputs_(df=filter_df_by_mode(df, m),
                                                       model_name=f"{feature_index}({m})",
                                                       feature_index=feature_index,
                                                       variance=var
                                                       ),
                          fmean_color=regline_COLOR[m],
                          scatter_hue_by=era_col_df_mode(df, m),
                          scatter_colormap=scatter_color_df_mode(df, m),
                          ylim=ylim,
                          log_transform=log_transform,
                          **filtered_params
                          )

    def format_axes(fig, axs: np.ndarray) -> None:
        # add vertical-lines to era boundaries
        for ax in axs.flatten():
            add_era_text_to_ax(ax=ax, era_dict=ERA, ylim=ylim, linelength=0.06, text_pos="top")
        fig.supylabel(metric, fontweight="bold")
        fig.supxlabel("Year", fontweight="bold")

    # computing the models:
    match metric:
        case "Chromaticity":
            # plotting:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(9.5, 6), sharex=True, sharey=True,
                                    layout="constrained")
            for (i, mo) in enumerate(["major", "minor"]):
                for (j, feature_index) in enumerate(["ILC", "OLC"]):
                    ax_full_gpr_model_(axs[j, i], mo, feature_index, variance)
            format_axes(fig, axs)
        case "Dissonance":
            # plotting:
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 3.5), sharex=True, sharey=True,
                                    layout="constrained")

            for (i, mo) in enumerate(["major", "minor"]):
                ax_full_gpr_model_(axs[i], mo, feature_index="DI", var=variance)
            format_axes(fig, axs)
    save = plotting_params.get('save', True)
    if save:
        fig_path = f'{result_dir}figs/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fname_anno = f'_{fname_anno}' if fname_anno else ""
        plt.savefig(f'{fig_path}gpr_{metric}{fname_anno}.pdf', dpi=200)


# %% GPR models for chordleve indices r in piece

def plot_gpr_indices_r(df: pd.DataFrame,
                       mode: Literal["major", "minor"],
                       ylim: Tuple[float, float],
                       lengthscale: Optional[float],
                       variance: Optional[float] | Literal["sample_var"],
                       optimize: bool,
                       fname_anno: Optional[str],
                       **plotting_params):
    # unpact params:
    repo_dir = plotting_params.get('repo_dir')
    xlim = plotting_params.get('xlim')
    scatter_jitter = plotting_params.get('scatter_jitter')
    plot_f_uncertainty = plotting_params.get('plot_f_uncertainty')
    save = plotting_params.get('save')

    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)
    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)

    if variance == "sample_var":
        var = compute_var(df=df, mode=mode, feature_index="r_ILC_DI")
    else:
        var = variance

    # computing the models:
    m_outputs = gpr_model_outputs(df=mode_df, model_name=f"r_ILC_DI({mode})",
                                  log_transform=False,
                                  repo_dir=repo_dir,
                                  feature_index="r_ILC_DI",
                                  lengthscale=lengthscale, variance=var, optimize=optimize)

    # unpack model outputs:
    _, (f_mean, f_var, y_mean, y_var), f_samples, lengthscale, modeled_feature = m_outputs

    X = m_outputs[0].data[0]
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))[:, 0]
    f_mean_reshaped = f_mean.reshape((-1, 1))[:, 0]
    f_var_reshaped_low = f_mean_reshaped - 1.96 * np.sqrt(f_var).reshape((-1, 1))[:, 0]
    f_var_reshaped_up = f_mean_reshaped + 1.96 * np.sqrt(f_var).reshape((-1, 1))[:, 0]

    if mode == "major":
        scatter_color = '#E18791'  # RED
        regline_color = regline_COLOR["major"]
    else:
        scatter_color = '#83A0BE'  # BLUE
        regline_color = regline_COLOR["minor"]

    # plots
    g = sns.JointGrid(height=6, ylim=ylim, xlim=xlim)
    x, y = mode_df["piece_year"].to_numpy(), mode_df["r_ILC_DI"].to_numpy()
    if scatter_jitter:
        x = rand_jitter(arr=x)
    sns.scatterplot(x=x, y=y, ec=scatter_color, fc="none", s=20, linewidth=1.5,
                    ax=g.ax_joint, alpha=0.65)

    # gpr line
    sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_mean_reshaped, color=regline_color)

    # gpr ci region
    if plot_f_uncertainty:
        sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_var_reshaped_low, color=regline_color,
                     alpha=0.1)
        sns.lineplot(ax=g.ax_joint, x=Xplot, y=f_var_reshaped_up, color=regline_color, alpha=0.1)
        line = g.ax_joint.get_lines()
        g.ax_joint.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(),
                                color=regline_color, alpha=0.1)

        sns.histplot(x=x, fill=True, linewidth=1, ax=g.ax_marg_x, color=scatter_color, alpha=0.6)
        sns.histplot(y=y, fill=True, linewidth=1, ax=g.ax_marg_y, color=scatter_color, alpha=0.6)

    # Add horizontal line at y=0
    g.ax_joint.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Conditionally modify y-axis tick labels if ylim is above 1
    if ylim[1] > 1:
        current_yticks = g.ax_joint.get_yticks()
        filtered_yticks = [tick for tick in current_yticks if tick <= 1]
        g.ax_joint.set_yticks(filtered_yticks)

    # add era info
    add_era_text_to_ax(ax=g.ax_joint, era_dict=ERA, ylim=ylim, linelength=0.04, text_pos="bottom")

    # add_period_text_to_ax(ax=g.ax_joint, y0=-0.96, fontsize="small")
    g.set_axis_labels(xlabel="Year", ylabel="r", fontweight="bold", fontsize="large")
    g.fig.set_figwidth(8)

    # annotate a specific point: Schumann
    if mode == "major":
        schumann_x = 1842
        schumann_y = -0.768
        g.ax_joint.text(schumann_x + 2.0, schumann_y, "Schumann", color="black", fontsize=8, va="center")  # Label

    # save
    fig_path = f'{result_dir}figs/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if fname_anno:
        fname_anno = f'_{fname_anno}'
    else:
        fname_anno = ""

    plt.tight_layout()
    if save:
        plt.savefig(f'{fig_path}gpr_ILC_DI_corr_{mode}{fname_anno}.pdf', dpi=200)


def plotly_scatter(df: pd.DataFrame, mode: Literal["major", "minor"],
                   metric: Literal["ILC", "OLC", "DI", "r_ILC_DI"]):
    # save the results to this folder:
    result_dir = create_results_folder(parent_folder="Results", analysis_name="GPR_analysis", repo_dir=repo_dir)

    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)

    fig = px.scatter(mode_df, x="piece_year", y=metric, color="corpus",
                     hover_data=['corpus', 'corpus_year', 'piece', 'piece_year', metric])

    fig_path = f'{result_dir}plotly/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig.write_html(f'{fig_path}GPR_{metric}_{mode}.html')


def compute_var(df: pd.DataFrame, mode: Literal["major", "minor"],
                feature_index: Literal["ILC", "OLC", "DI", "r_ILC_DI"]):
    mode_df = get_piece_df_by_localkey_mode(df=df, mode=mode)
    var = mode_df[feature_index].var(numeric_only=True)
    return var


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo_dir = f'{user}/Codes/chromaticism-codes/'
    print(f'Loading dfs ...')
    piece_indices = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/piece_level_indices_by_mode.pickle")
    r_vals_df = load_file_as_df(f"{repo_dir}Data/prep_data/for_analysis/chord_indices_r_vals_by_piece.pickle")

    # maj = piece_indices[piece_indices["localkey_mode"] == "major"]
    # minor = piece_indices[piece_indices["localkey_mode"] == "minor"]
    #
    # rg = np.log(maj["DI"].to_numpy().reshape(-1)+1e-6)
    # print(f'{rg.min()=}, {rg.max()=}')
    #
    # assert False

    plotting_common_params = dict(
        repo_dir=repo_dir,
        log_transform=False,
        lengthscale=25.0, variance="sample_var",
        plot_f_uncertainty=True, plot_y_uncertainty=False,
        scatter_jitter=True, xlim=(1575, 1950),
        legend_loc=None, save=True)

    # print(f'GPR for chromaticity ...')
    # plot_gpr(df=piece_indices,
    #          metric="Chromaticity",
    #          optimize=False,
    #          ylim=(-1, 5),
    #          fname_anno="capped",
    #          **plotting_common_params)
    #
    # print(f'GPR for dissonance ...')
    # plot_gpr(df=piece_indices,
    #          metric="Dissonance",
    #          optimize=False,
    #          ylim=(0, 1),
    #          fname_anno=None,
    #          **plotting_common_params)

    # assert False

    print(f'GPR for ILC-DI correlation ...')
    r_models_params = dict(optimize=False,
                           ylim=(-1, 1.25), fname_anno=None,
                           df=r_vals_df)



    for m in ["major", "minor"]:
        plot_gpr_indices_r(mode=m,
                           **r_models_params,
                           **plotting_common_params
                           )

    assert False

    print(f'Plotly figures ...')
    plotly_scatter(df=piece_indices, metric="ILC", mode="major")
    plotly_scatter(df=piece_indices, metric="ILC", mode="minor")
    plotly_scatter(df=piece_indices, metric="OLC", mode="major")
    plotly_scatter(df=piece_indices, metric="OLC", mode="minor")
    plotly_scatter(df=piece_indices, metric="DI", mode="major")
    plotly_scatter(df=piece_indices, metric="DI", mode="minor")
    plotly_scatter(df=r_vals_df, mode="major", metric="r_ILC_DI")
    plotly_scatter(df=r_vals_df, mode="minor", metric="r_ILC_DI")
