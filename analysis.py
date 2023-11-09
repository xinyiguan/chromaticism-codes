from typing import Literal, List, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
import gpflow as gf
from tensorflow import Tensor

from gpflow.utilities import print_summary

MODEL_OUTPUT = Tuple[gf.models.gpr.GPR, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None, str]



def correlation(series1: pd.Series, series2: pd.Series):
    return pg.corr(x=series1, y=series2)


def get_period_df(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]) -> pd.DataFrame:
    late_renaissance = df[df["piece_year"] < 1662]
    baroque = df[(1662 <= df["piece_year"]) & (df["piece_year"] < 1761)]
    classical = df[(1761 <= df["piece_year"]) & (df["piece_year"] < 1820)]
    early_romantic = df[(1820 <= df["piece_year"]) & (df["piece_year"] < 1871)]
    late_romantic = df[df["piece_year"] >= 1871]

    if period == "renaissance":
        return late_renaissance
    elif period == "baroque":
        return baroque
    elif period == "classical":
        return classical
    elif period == "early_romantic":
        return early_romantic
    elif period == "late_romantic":
        return late_romantic
    else:
        raise ValueError


# GPR modeling
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

def gpr_model_outputs(df: pd.DataFrame, feature: str, sample: int = 5) -> MODEL_OUTPUT:
    X = df["piece_year"].to_numpy().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
    Y = df[feature].to_numpy().astype(float).reshape((-1, 1))

    k = gf.kernels.SquaredExponential()

    m = gpr_model(X=X, Y=Y, kernel=k, optimize=True)

    f_mean, f_var = m.predict_f(Xplot, full_cov=False)
    y_mean, y_var = m.predict_y(Xplot)
    if isinstance(sample, int):
        samples = m.predict_f_samples(Xplot, sample)
    else:
        raise TypeError

    return m, (f_mean, f_var, y_mean, y_var), samples, feature




#
# if __name__ == "__main__":
#     result_df = pd.read_csv("data/piece_chromaticities.tsv", sep="\t")
#     renaissance = get_period_df(result_df, period='renaissance')
#
#     a=correlation(series1=renaissance["mean_r_chromaticity"], series2=renaissance["mean_ct_chromaticity"])
#     print(a)
