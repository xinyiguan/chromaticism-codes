# auxiliary functions for analysis
import os
from typing import Literal, Tuple, List, Optional
from matplotlib import colormaps
import numpy as np
import pandas as pd


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
    if method=="Johannes":
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


# ___________
Johannes_periods = ["pre-Baroque", "Baroque", "Classical", "Extended tonality"]
Fabian_periods = ["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]


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

def map_array_to_colors(arr, color_map: str | None) -> List[str]:
    unique_values = np.unique(arr)
    # num_unique_values = len(unique_values)

    # Define the colormap using the recommended method
    # cmap = mcolors.ListedColormap(cm.Dark2.colors[:num_unique_values])
    cmap = colormaps[color_map]

    # Create a dictionary to map unique values to colors
    value_to_color = dict(zip(unique_values, cmap.colors))

    # Map the values in the array to colors, using "gray" for 0 values
    color_list = [value_to_color[val] if val != 0 else "gray" for val in arr]

    return color_list


# jitter
def rand_jitter(arr, scale: float = .01):
    stdev = scale * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev
