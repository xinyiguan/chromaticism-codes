from typing import Tuple
import gpflow as gf
import numpy as np
import pandas as pd
from gpflow.utilities import print_summary

from tensorflow import Tensor

MODEL_OUTPUT = Tuple[gf.models.gpr.GPR, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None, str]


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


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    rc = gpr_model_outputs(df=result_df, feature="mean_r_chromaticity")
