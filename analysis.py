from typing import Literal, List
import pandas as pd
import statsmodels.api as sm
# correlation between indices


def correlation(series1: pd.Series, series2: pd.Series):
    return series1.corr(series2)


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


def chromaticity_pairwise_corr(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]):
    period_df = get_period_df(df=df, period=period)

    root = period_df["mean_r_chromaticity"]
    ct = period_df["mean_ct_chromaticity"]
    nct = period_df["mean_nct_chromaticity"]

    # correlation between ROOT and CT
    root_ct_corr = correlation(root, ct)
    root_nct_corr = correlation(root, nct)
    ct_nct_corr = correlation(ct, nct)

    return root_ct_corr, root_nct_corr, ct_nct_corr


def regression(df: pd.DataFrame,
               period: Literal["renaissance", "baroque", "classical", "early_romantic", "late_romantic"],
               X_type: Literal["r", "ct", "nct"], y_type: Literal["r", "ct", "nct"]):
    period_df = get_period_df(df=df, period=period)
    X_str = f'mean_{X_type}_chromaticity'
    y_str = f'mean_{y_type}_chromaticity'

    X = period_df[X_str]
    y = period_df[y_str]
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    print(results.summary())


if __name__ == "__main__":
    result_df = pd.read_csv("data/piece_chromaticities.tsv", sep="\t")
    a = chromaticity_pairwise_corr(result_df, period="renaissance")
    print(f'{a=}')

    regression(result_df, period="renaissance", X_type="r", y_type="ct")
