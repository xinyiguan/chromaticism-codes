import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from Code.utils.util import load_file_as_df


def plot_lowess_model(df: pd.DataFrame,
                      model_name: str,
                      feature_index: Literal["WLC", "OLC", "WLD"],
                      sample_num: Optional[int],
                      repo_dir: str):
    fig, ax = plt.subplots()

    # bootstrap
    for _ in range(sample_num):
        X = df["piece_year"].to_numpy().squeeze().astype(float)
        Y = df[feature_index].to_numpy().squeeze().astype(float)
        l = lowess(endog=Y, exog=X, frac=.25)

        ax.plot(l[:, 0], l[:, 1], c="crimson", alpha=.3)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    user = os.path.expanduser("~")
    repo_dir = f'{user}/Codes/chromaticism-codes/'

    major_df = load_file_as_df(f'{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_major.pickle')
    minor_df = load_file_as_df(f'{repo_dir}Data/prep_data/for_analysis/chromaticity_piece_minor.pickle')

    plot_lowess_model(df=major_df, feature_index="WLC", model_name="WLC(major)",
                      repo_dir=repo_dir, sample_num=1000)
