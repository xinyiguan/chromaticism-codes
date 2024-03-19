from contextlib import redirect_stdout
from typing import Tuple, Literal, Optional

import gpflow as gf
import numpy as np
import pandas as pd
from gpflow.utilities import print_summary
from tensorflow import Tensor

MODEL_OUTPUT = Tuple[gf.models.gpr.GPR, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None, np.ndarray, str]


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
    # print_summary(m)

    return m


def gpr_model_outputs(df: pd.DataFrame,
                      df_type: Literal["Major", "Minor", "Combined"],
                      feature: str,
                      lengthscale: Optional[int] = None,
                      sample: int = 5) -> MODEL_OUTPUT:

    X = df["piece_year"].to_numpy().astype(float).reshape((-1, 1))
    Xplot = np.arange(min(X), max(X) + 1).reshape((-1, 1))
    Y = df[feature].to_numpy().astype(float).reshape((-1, 1))

    # try set kernel length scale, by default, we use the optimization.
    if isinstance(lengthscale, int):
        k = gf.kernels.SquaredExponential(lengthscales=lengthscale)
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=False)
    else:
        k = gf.kernels.SquaredExponential()
        m = gpr_model(X=X, Y=Y, kernel=k, optimize=True)

    save_gpr_params_to_table(m=m, df_name=df_type, m_name=feature)

    f_mean, f_var = m.predict_f(Xplot, full_cov=False)
    y_mean, y_var = m.predict_y(Xplot)
    if isinstance(sample, int):
        samples = m.predict_f_samples(Xplot, sample)
    else:
        raise TypeError

    ls = k.lengthscales.numpy()
    return m, (f_mean, f_var, y_mean, y_var), samples, ls, feature

def save_gpr_params_to_table(m: gf.models.gpr.GPR, m_name: str, df_name: str,
                             gpr_params_output_path: str = 'gpr_params.txt'):
    with open(gpr_params_output_path, 'a') as f:
        with redirect_stdout(f):
            f.write(f'{df_name} {m_name}\n')
            print_summary(m)
            f.write('\n')  # Add a newline for separation if needed


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    rc = gpr_model_outputs(df=result_df, feature="RC", df_type="Combined")
