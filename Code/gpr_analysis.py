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

from Code.utils.auxiliary import create_results_folder, map_array_to_colors, rand_jitter, Fabian_periods, \
    Johannes_periods, mean_var_after_log, median_CI_after_log
from Code.utils.util import load_file_as_df

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
                      feature_index: Literal["WLC", "OLC", "WLD"],
                      lengthscale: Optional[int],
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
    pickle_file_path = f'{analysis_dir}{model_name}_ModelOutputs.pickle'
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(model_outputs, f)

    # save model params:
    params_file_path = f'{analysis_dir}{model_name}_Params.txt'
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
        ax.scatter(rand_jitter(X), Y, c=color, s=12, alpha=0.6
                   , label="Observations"
                   )
    else:
        ax.scatter(X, Y, c=color, s=12, alpha=0.6
                   , label="Observations"
                   )
    return ax


def ax_gpr_prediction(ax: Axes,
                      m_outputs: MODEL_OUTPUT,
                      prediction_stat: Literal["mean", "median"],
                      fmean_color: Optional[str],
                      # fvar_color: Optional[str],
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

    ax.plot(Xplot, exp_f, "-", color=fmean_color, label=f"f {prediction_stat}({modeled_feature})", linewidth=2)
    ax.set_ylim(0, 3)

    if plot_f_uncertainty:
        ax.plot(Xplot, f_lower, "--", color=fmean_color, label="f 95% confidence", alpha=0.3)
        ax.plot(Xplot, f_upper, "--", color=fmean_color, alpha=0.3)
        ax.fill_between(
            Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=fmean_color, alpha=0.3
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
                                   show_second_yticks: bool,
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
                      fvar_color: Optional[str],
                      plot_samples: int | None,
                      plot_f_uncertainty: bool,
                      plot_y_uncertainty: bool,
                      scatter_colormap: Optional[str | List[str]],
                      scatter_hue_by: Optional[np.ndarray],
                      scatter_jitter: bool):
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

    ax.set_ylim([0, 10])


# %% GPR models plots

def plot_gpr_chromaticities_by_mode(major_df: pd.DataFrame, minor_df: pd.DataFrame,
prediction_stat: Literal["mean", "median"],
                                    era_division: Literal["Fabian", "Johannes"],
                                    lengthscale: Optional[float],
                                    plot_samples: int | None,
                                    plot_f_uncertainty: bool,
                                    plot_y_uncertainty: bool,
                                    scatter_hue_by: bool,
                                    repo_dir: str
                                    ):
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
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(11, 6), sharex=True, sharey=True,
                            layout="constrained")
    fmean_color = "#3b3b3b"
    # color_palette4 = ['#D9BDC3', '#C4D0CC', '#76A0AD', '#597C8B']
    color_palette4 = ['#4f6980', '#849db1', '#638b66', '#bfbb60']

    color_palette5 = ['#4f6980', '#849db1', '#a2ceaa', '#638b66', '#bfbb60']

    if era_division == "Fabian":
        scatter_color = color_palette5
    else:
        scatter_color = color_palette4

    # major wlc:
    era_col_major = major_df[f'period_{era_division}'].to_numpy()

    ax_full_gpr_model(ax=axs[0, 0],
                      ax_title="WLC (major)",
                      m_outputs=major_wlc,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      fvar_color=None,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=era_col_major,
                      scatter_colormap=scatter_color,
                      scatter_jitter=True,
                      # show_second_yticks=False
                      )
    # major olc:
    ax_full_gpr_model(ax=axs[1, 0],
                      ax_title="OLC (major)",
                      m_outputs=major_olc,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      fvar_color=None,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=era_col_major,
                      scatter_colormap=scatter_color,
                      scatter_jitter=True,
                      # show_second_yticks=False
                      )
    # minor wlc:
    era_col_minor = minor_df[f'period_{era_division}'].to_numpy()
    ax_full_gpr_model(ax=axs[0, 1],
                      ax_title="WLC (minor)",
                      m_outputs=minor_wlc,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      fvar_color=None,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=era_col_minor,
                      scatter_colormap=scatter_color,
                      scatter_jitter=True,
                      # show_second_yticks=True
                      )
    # minor olc:
    ax_full_gpr_model(ax=axs[1, 1],
                      ax_title="OLC (minor)",
                      m_outputs=minor_olc,
                      prediction_stat=prediction_stat,
                      fmean_color=fmean_color,
                      fvar_color=None,
                      plot_f_uncertainty=plot_f_uncertainty,
                      plot_y_uncertainty=plot_y_uncertainty,
                      plot_samples=plot_samples,
                      scatter_hue_by=era_col_minor,
                      scatter_jitter=True,
                      scatter_colormap=scatter_color,
                      # show_second_yticks=True
                      )

    fig.supylabel("Chromaticity", fontweight="bold")
    # fig.text(x=0.97, y=0.5, s="f mean \n\n\n", size=13, fontweight='bold', rotation=270,
    #          ha='center', va='center')

    plt.show()


if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo_dir = f'{user}/Codes/chromaticism-codes/'

    major_df = load_file_as_df(f'{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_major.pickle')
    minor_df = load_file_as_df(f'{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_minor.pickle')
    plot_gpr_chromaticities_by_mode(major_df=major_df, minor_df=minor_df,
                                    prediction_stat="median",
                                    era_division="Fabian", lengthscale=10,
                                    plot_samples=False,
                                    plot_y_uncertainty=False,
                                    plot_f_uncertainty=True,
                                    scatter_hue_by=True,
                                    repo_dir=repo_dir)

    # plot_gpr_all_trendlines(major_df=major_df, minor_df=minor_df, lengthscale=10)
