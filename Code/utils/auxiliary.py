# auxiliary functions for analysis
import os
from typing import Literal, Tuple, List, Optional
from matplotlib import colormaps
import numpy as np
import pandas as pd

# color palette for period division
Johannes_periods = ["pre-Baroque", "Baroque", "Classical", "Extended tonality"]
Fabian_periods = ["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]

color_palette4 = ['#4f6980', '#849db1', '#638b66', '#bfbb60']
color_palette5 = ['#4f6980', '#849db1', '#a2ceaa', '#638b66', '#bfbb60']
# color_palette5 = ['#2A5084', '#4C5F76', '#F7A851', '#F46A03', '#697217']

# preprocessing/computing metrics ___________ :

def determine_period(row: pd.Series, method: Literal["Fabian", "Johannes"]):
    y = row["piece_year"]
    if method == "Fabian":
        if y < 1662:
            p = "Renaissance"
        elif 1662 <= y < 1763:
            p = "Baroque"
        elif 1763 <= y < 1821:
            p = "Classical"
        elif 1821 <= y < 1869:
            p = "Early Romantic"
        elif y >= 1869:
            p = "Late Romantic"
        else:
            raise ValueError
    elif method == "Johannes":
        if y < 1650:
            p = "pre-Baroque"
        elif 1650 <= y < 1750:
            p = "Baroque"
        elif 1750 <= y < 1800:
            p = "Classical"
        elif y >= 1800:
            p = "Extended tonality"
        else:
            raise ValueError
    else:
        raise ValueError
    return p


def determine_period_id(row: pd.Series, method: Literal["Fabian", "Johannes"]):
    if method == "Johannes":
        JP = f'period_Johannes'
        assert JP in row.index
        if row[JP] == "pre-Baroque":
            ID = 1
        elif row[JP] == "Baroque":
            ID = 2
        elif row[JP] == "Classical":
            ID = 3
        elif row[JP] == "Extended tonality":
            ID = 4
        else:
            raise ValueError
    else:
        assert 'period_Fabian' in row.index
        FP = f'period_Fabian'
        if row[FP] == "Renaissance":
            ID = 1
        elif row[FP] == "Baroque":
            ID = 2
        elif row[FP] == "Classical":
            ID = 3
        elif row[FP] == "Early Romantic":
            ID = 4
        elif row[FP] == "Late Romantic":
            ID = 5
        else:
            raise ValueError
    return ID


def get_period_df(df: pd.DataFrame,
                  method: Literal["Johannes", "Fabian"],
                  period: Literal["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic",
                  "pre-Baroque", "Extended tonality"]):
    if method == "Johannes":
        assert period in Johannes_periods
        t1, t2, t3 = (1650, 1750, 1800)
        pre_Baroque = df[df["piece_year"] < t1]
        Baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
        Classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
        extended_tonality = df[df["piece_year"] >= t3]

        if period == "pre-Baroque":
            return pre_Baroque
        elif period == "Baroque":
            return Baroque
        elif period == "Classical":
            return Classical
        elif period == "Extended tonality":
            return extended_tonality
        else:
            raise ValueError


    elif method == "Fabian":
        assert period in Fabian_periods
        t1, t2, t3, t4 = (1662, 1763, 1821, 1869)

        late_renaissance = df[df["piece_year"] < t1]
        baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
        classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
        early_romantic = df[(t3 <= df["piece_year"]) & (df["piece_year"] < t4)]
        late_romantic = df[df["piece_year"] >= t4]

        if period == "Renaissance":
            return late_renaissance
        elif period == "Baroque":
            return baroque
        elif period == "Classical":
            return classical
        elif period == "Early Romantic":
            return early_romantic
        elif period == "Late Romantic":
            return late_romantic
        else:
            raise ValueError
    else:
        raise ValueError




def determine_group(row: pd.Series, interval: Literal[25, 50]):
    year = row["piece_year"]
    if year < 1600:
        return "<1600"
    elif interval == 50:
        return f"{((year - 1600) // 50) * 50 + 1600}-{((year - 1600) // 50) * 50 + 1650}"
    elif interval == 25:
        return f"{((year - 1600) // 25) * 25 + 1600}-{((year - 1600) // 25) * 25 + 1625}"
    else:
        raise ValueError("Interval should be either 25 or 50")

def create_results_folder(parent_folder: Literal["Data", "Results"], analysis_name: Optional[str], repo_dir: str):
    if parent_folder == "Data":
        directory = f"{repo_dir}{parent_folder}/prep_data/for_analysis/"
    elif parent_folder == "Results":
        assert analysis_name is not None
        directory = f"{repo_dir}{parent_folder}/{analysis_name}/"
    else:
        raise ValueError
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


# ___________

def pprint_p_text(p_val: float):
    if p_val < 0.001:
        p_val_txt = 'p < .001'
    elif 0.001 < p_val < 0.05:
        p_val_txt = 'p < .05'
    else:
        p_val_txt = f'p = {p_val:.2f}'
    return p_val_txt


# plotting related ____________

def map_array_to_colors(arr: np.ndarray, color_map: str | List[str]) -> List[str]:
    unique_values = np.unique(arr)
    num_unique_values = len(unique_values)

    # Define the colormap using the recommended method
    # cmap = mcolors.ListedColormap(cm.Dark2.colors[:num_unique_values])
    if isinstance(color_map, str):
        cmap = colormaps[color_map]
        # Create a dictionary to map unique values to colors
        value_to_color = dict(zip(unique_values, cmap.colors))

        # Map the values in the array to colors, using "gray" for 0 values
        color_list = [value_to_color[val] if val != 0 else "gray" for val in arr]
    else:

        assert len(color_map) >= num_unique_values
        # Create a dictionary to map unique values to colors
        value_to_color = dict(zip(unique_values, color_map))
        color_list = [value_to_color[val] for val in arr]

    return color_list


# jitter
def rand_jitter(arr, scale: float = .005):
    stdev = scale * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# GPR log-transform

def mean_var_after_log(mu: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given Y ~ N(mu, var) where Y=ln(Z), output mean and variance of Z
    """

    z_mu = np.exp(mu + var / 2)

    z_var = (np.exp(var) - 1) * (np.exp(2 * mu + var))

    return z_mu, z_var


def median_CI_after_log(mu: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
        Given Y ~ N(mu, var) where Y=ln(Z), output median and 95% CI of Z
    """
    z_med = np.exp(mu)

    z_CI_lower = np.exp(mu - 1.96 * np.sqrt(var))
    z_CI_upper = np.exp(mu + 1.96 * np.sqrt(var))

    return z_med, (z_CI_lower, z_CI_upper)


# dataframe filtering _________
def get_piece_df_by_localkey_mode(df: pd.DataFrame, mode: Literal["major", "minor"]) -> pd.DataFrame:
    if mode == "major":
        result_df = df[df['localkey_mode'].isin(['major'])]
    else:
        result_df = df[df['localkey_mode'].isin(['minor'])]

    return result_df
def exclude_piece_from_corpus(df: pd.DataFrame, corpus_piece_tups: List[Tuple[str, str]]) -> pd.DataFrame:
    corpus_list, piece_list = zip(*corpus_piece_tups)
    res_df = df[~(df['corpus'].isin(corpus_list) & df['piece'].isin(piece_list))]
    return res_df