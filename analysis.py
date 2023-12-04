import os
from fractions import Fraction
from typing import Literal, Tuple

import pandas as pd
import pingouin as pg

from utils.htypes import Key

pd.set_option('display.max_columns', None)


def correlation(series1: pd.Series, series2: pd.Series):
    return pg.corr(x=series1, y=series2)


def get_period_df(df: pd.DataFrame, period: Literal[
    "renaissance", "baroque", "classical", "early_romantic", "late_romantic"]) -> pd.DataFrame:
    t1, t2, t3, t4 = (1662, 1763, 1821, 1869)

    late_renaissance = df[df["piece_year"] < t1]
    baroque = df[(t1 <= df["piece_year"]) & (df["piece_year"] < t2)]
    classical = df[(t2 <= df["piece_year"]) & (df["piece_year"] < t3)]
    early_romantic = df[(t3 <= df["piece_year"]) & (df["piece_year"] < t4)]
    late_romantic = df[df["piece_year"] >= t4]

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

def get_corpus_df(df:pd.DataFrame, corpus: str)->pd.DataFrame:

    result = df[df["corpus"]==corpus]
    return result


def corpora_in_periods_dfs(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df[["corpus", "piece", "piece_year"]]
    df = df.reset_index(drop=True)

    renaissance = get_period_df(df=df, period="renaissance")
    renaissance.loc[:, "period"] = "Renaissance"
    # renaissance["period"] = "Renaissance"

    baroque = get_period_df(df=df, period="baroque")
    baroque.loc[:, "period"] = "Baroque"

    classical = get_period_df(df=df, period="classical")
    classical.loc[:, "period"] = "Classical"

    early_romantic = get_period_df(df=df, period="early_romantic")
    early_romantic.loc[:, "period"] = "Early Romantic"

    late_romantic = get_period_df(df=df, period="late_romantic")
    late_romantic.loc[:, "period"] = "Late Romantic"

    # corpora_by_period_df = pd.concat([renaissance, baroque, classical, early_romantic, late_romantic])
    # corpora_by_period_df = corpora_by_period_df.set_index("period")

    return renaissance, baroque, classical, early_romantic, late_romantic


def get_copora_list_in_period_table(df: pd.DataFrame):
    renaissance, baroque, classical, early_romantic, late_romantic = corpora_in_periods_dfs(df=df)

    renaissance_corpus = [x.replace("_", " ") for x in renaissance['corpus'].unique()]
    baroque_corpus = [x.replace("_", " ") for x in baroque['corpus'].unique()]
    classical_corpus = [x.replace("_", " ") for x in classical['corpus'].unique()]
    early_romantic_corpus = [x.replace("_", " ") for x in early_romantic['corpus'].unique()]
    late_romantic_corpus = [x.replace("_", " ") for x in late_romantic['corpus'].unique()]

    data = {
        'Renaissance': renaissance_corpus,
        'Baroque': baroque_corpus,
        'Classical': classical_corpus,
        'Early Romantic': early_romantic_corpus,
        'Late Romantic': late_romantic_corpus
    }

    # Start building the LaTeX table
    latex_table = "\\begin{tabular}{ll}\n\\toprule\nPeriod & Corpus \\\\\n\\midrule\n"
    total_periods = len(data)
    current_period_count = 0

    for period, corpus_list in data.items():
        current_period_count += 1
        for i, corpus in enumerate(corpus_list):
            if i == 0:
                latex_table += f"{period} & {corpus} \\\\\n"
            else:
                latex_table += f" & {corpus} \\\\\n"

        if current_period_count < total_periods:  # add horizontal line except for the last period
            latex_table += "\\midrule\n"

    # end the LaTeX table
    latex_table += "\\bottomrule\n\\end{tabular}"

    latex_table_path = os.path.join("results_latex", 'corpora_by_period.txt')
    with open(latex_table_path, 'w') as file:
        file.write(latex_table)

    return latex_table


# EXAMPLES _________________________________________________________________________________________

def get_rachmaninoff_CI(tsv_path: str = "data/chord_indices.tsv") -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    rach = df[(df["corpus"] == "rachmaninoff_piano") & (df["piece"] == "op42_01a")]
    print(rach)

def get_bwv808_example_CI(version: Literal["original", "agrements"],
                          tsv_path: str = "data/chord_indices.tsv") -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["corpus", "piece", "quarterbeats", "globalkey", "localkey", "chord",
                              "root", "RC", "ct", "CTC", "nct", "NCTC"])
    df['quarterbeats'] = df['quarterbeats'].apply(lambda x: float(Fraction(x)) if '/' in x else float(x))

    if version == "original":
        bwv808 = df[(df['corpus'] == 'bach_en_fr_suites') & (df['piece'] == 'BWV808_04_Sarabande') & (
            df['quarterbeats'].between(0, 22))]

    else:
        bwv808 = df[(df['corpus'] == 'bach_en_fr_suites') & (df['piece'] == 'BWV808_04a_Agrements_de_la_Sarabande') & (
            df['quarterbeats'].between(0, 22))]

    data = [bwv808["RC"].mean(), bwv808["CTC"].mean(), bwv808["NCTC"].mean()]

    cols = ["RC", "CTC", "NCTC"]

    results = pd.DataFrame([data], columns=cols)

    results.index = [f'{version}']
    return results


def get_k331_1_variations_CI(version: Literal["thema", "var1", "var2", "var3", "var4", "var5", "var6"],
                             tsv_path: str = "data/chord_indices.tsv") -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["corpus", "piece", "quarterbeats", "globalkey", "localkey", "chord",
                              "root", "RC", "ct", "CTC", "nct", "NCTC"])
    df['quarterbeats'] = df['quarterbeats'].apply(lambda x: float(Fraction(x)) if '/' in x else float(x))

    mozart = df[(df['corpus'] == 'mozart_piano_sonatas') & (df['piece'] == 'K331-1')]

    if version == "thema":
        k331 = mozart[(mozart["quarterbeats"].between(0, 45 / 2))]

    elif version == "var1":
        k331 = mozart[(mozart["quarterbeats"].between(111 / 2, 153 / 2))]

    elif version == "var2":
        k331 = mozart[(mozart["quarterbeats"].between(109, 261 / 2))]

    elif version == "var3":
        k331 = mozart[(mozart["quarterbeats"].between(162, 369 / 2))]

    elif version == "var4":
        k331 = mozart[(mozart["quarterbeats"].between(216, 477 / 2))]

    elif version == "var5":
        k331 = mozart[(mozart["quarterbeats"].between(1085 / 4, 293))]

    elif version == "var6":
        k331 = mozart[(mozart["quarterbeats"].between(325, 354))]

    else:
        raise ValueError

    k331["version"] = f'{version}'
    k331 = k331[
        ["version", "globalkey", "localkey", "chord", "root", "RC", "ct", "CTC", "nct", "NCTC"]]

    with pd.option_context('display.max_columns', None):
        k331.to_csv(f"data/k331/tsv/k331_{version}.tsv", sep="\t")

    data = [k331["RC"].mean(), k331["CTC"].mean(), k331["NCTC"].mean()]

    cols = ["RC", "CTC", "NCTC"]

    results = pd.DataFrame([data], columns=cols)

    results.index = [f'{version}']
    return results


def get_k331_CI_table():
    thema = get_k331_1_variations_CI(version="thema")
    var1 = get_k331_1_variations_CI(version="var1")
    var2 = get_k331_1_variations_CI(version="var2")
    var3 = get_k331_1_variations_CI(version="var3")
    var4 = get_k331_1_variations_CI(version="var4")
    var5 = get_k331_1_variations_CI(version="var5")
    var6 = get_k331_1_variations_CI(version="var6")

    df = pd.concat([thema, var1, var2, var3, var4, var5, var6])
    df = df.round(3)

    latex_table = df.to_latex(float_format=lambda x: '%.3f' % x)

    latex_table_path = os.path.join("results_latex", 'K331_CI_table.txt')
    with open(latex_table_path, 'w') as file:
        file.write(latex_table)

    return latex_table


# MAJOR/MINOR MODE GROUP __________________________________________________________________________
def get_major_minor_pieces_df(mode: Literal["major", "minor"],
                              df: pd.DataFrame,
                              save_df: bool = False) -> pd.DataFrame:
    # df = pd.read_csv(tsv_path, sep="\t")
    df["mode"] = df.apply(lambda row: Key.from_string(row["globalkey"]).mode, axis=1)


    if mode == "major":
        result = df[df["mode"] == "major"]
    elif mode == "minor":
        result = df[df["mode"] == "minor"]
    else:
        raise ValueError
    if save_df:
        result.to_csv(f"data/{mode}key_piece_indices.tsv", sep="\t")

    return result

def get_major_minor_group_stats(df: pd.DataFrame):
    major = get_major_minor_pieces_df(mode="major", df=df)
    minor = get_major_minor_pieces_df(mode="minor", df=df)

    num_total = len(df)
    num_major = len(major)
    num_minor = len(minor)
    major_percentage = num_major/num_total
    minor_percentage = num_minor/num_total

    print(f'{num_major=}')
    print(f'{num_minor=}')
    print(f'{major_percentage=}')
    print(f'{minor_percentage=}')



def get_MajorMinor_MeanCI_ttest_table(major: pd.DataFrame, minor: pd.DataFrame):
    major_ctc_mean = major["CTC"].mean()
    minor_ctc_mean = minor["CTC"].mean()

    major_rc_mean = major["RC"].mean()
    minor_rc_mean = minor["RC"].mean()

    major_nctc_mean = major["NCTC"].mean()
    minor_nctc_mean = minor["NCTC"].mean()

    CI_dict = {
        "RC(major)": major_rc_mean,
        "RC(minor)": minor_rc_mean,
        "CTC(major)": major_ctc_mean,
        "CTC(minor)": minor_ctc_mean,
        "NCTC(major)": major_nctc_mean,
        "NCTC(minor)": minor_nctc_mean,
    }

    # Convert dictionary to DataFrame
    CI_df = pd.DataFrame(CI_dict.items(), columns=['CI Type', 'Value'])

    # Generate LaTeX table from the modified DataFrame
    MeanCI_table_latex = CI_df.to_latex(index=False, float_format=lambda x: '%.3f' % x)

    MeanCI_results_path = os.path.join("results_latex", 'MajorMinor_MeanCI.txt')

    # Write the LaTeX string to a text file
    with open(MeanCI_results_path, 'w') as file:
        file.write(MeanCI_table_latex)

    print(MeanCI_table_latex)
    print("==============================================")

    # perform two-sample t-test

    major_rc = major["RC"].to_numpy()
    minor_rc = minor["RC"].to_numpy()

    major_ctc = major["CTC"].to_numpy()
    minor_ctc = minor["CTC"].to_numpy()

    major_nctc = major["NCTC"].to_numpy()
    minor_nctc = minor["NCTC"].to_numpy()

    cols2use = ["T", "p-val", "cohen-d"]
    rc_ttest = pg.ttest(x=major_rc, y=minor_rc, correction='auto')[cols2use]
    ctc_ttest = pg.ttest(x=major_ctc, y=minor_ctc, correction='auto')[cols2use]
    nctc_ttest = pg.ttest(x=major_nctc, y=minor_nctc, correction='auto')[cols2use]

    ttest_results = pd.concat([rc_ttest, ctc_ttest, nctc_ttest])
    new_index = ["RC", "CTC", "NCTC"]
    new_cols = {"T": "t-val", "p-val": "p-val", "cohen-d": "cohen d"}
    ttest_results.index = new_index
    ttest_results = ttest_results.rename(columns=new_cols).to_latex(float_format=lambda x: '%.3f' % x)

    ttest_result_path = os.path.join("results_latex", 'TwoSample_T_Test.txt')
    with open(ttest_result_path, 'w') as file:
        file.write(ttest_results)

    print(ttest_results)


def get_corpus_CIs(tsv_file: str = "data/piece_indices.tsv") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """compute the corpus CI in each period
    (note that for some corpora, there may be overlap between periods, but no overlapping pieces between periods)"""
    df = pd.read_csv(tsv_file, sep="\t")

    renaissance = get_period_df(df=df, period="renaissance")
    baroque = get_period_df(df=df, period="baroque")
    classical = get_period_df(df=df, period="classical")
    early_romantic = get_period_df(df=df, period="early_romantic")
    late_romantic = get_period_df(df=df, period="late_romantic")

    periods_dfs = [renaissance, baroque, classical, early_romantic, late_romantic]
    periods = ["Renaissance", "Baroque", "Classical", "Early Romantic", "Late Romantic"]

    period_CIs = []
    for i, period in enumerate(periods_dfs):
        corpus = period.groupby(["corpus"]).agg(
            RC=("RC", "mean"),
            CTC=("CTC", "mean"),
            NCTC=("NCTC", "mean")
        )
        corpus["period"] = periods[i]
        corpus = corpus.reset_index()
        corpus = corpus[["period", "corpus", "RC", "CTC", "NCTC"]]
        period_CIs.append(corpus)

    return tuple(period_CIs)


def get_corpus_MajorMinor_CIs_table():
    renaissance_M, baroque_M, classical_M, early_romantic_M, late_romantic_M = get_corpus_CIs(
        tsv_file="data/majorkey_piece_indices.tsv")
    renaissance_m, baroque_m, classical_m, early_romantic_m, late_romantic_m = get_corpus_CIs(
        tsv_file="data/minorkey_piece_indices.tsv")

    combined_Major = pd.concat([renaissance_M, baroque_M, classical_M, early_romantic_M, late_romantic_M])
    combined_Major = combined_Major.round({'RC': 3, 'CTC': 3, 'NCTC': 3})
    combined_Major = combined_Major.rename(columns={'RC': 'RC(Major)', 'CTC': 'CTC(Major)', 'NCTC': 'NCTC(Major)'})

    combined_minor = pd.concat([renaissance_m, baroque_m, classical_m, early_romantic_m, late_romantic_m])
    combined_minor = combined_minor.round({'RC': 3, 'CTC': 3, 'NCTC': 3})
    combined_minor = combined_minor.rename(columns={'RC': 'RC(Minor)', 'CTC': 'CTC(Minor)', 'NCTC': 'NCTC(Minor)'})

    combined_df = pd.merge(combined_Major, combined_minor, how='outer')

    combined_df = combined_df[['period', 'corpus', 'RC(Major)', 'RC(Minor)', 'CTC(Major)','CTC(Minor)', 'NCTC(Major)','NCTC(Minor)']]

    custom_order = ['Renaissance', 'Baroque', 'Classical', 'Early Romantic', 'Late Romantic']
    combined_df['period'] = pd.Categorical(combined_df['period'], categories=custom_order, ordered=True)
    combined_df = combined_df.sort_values('period')

    # Generate LaTeX table string
    latex_table = "\\begin{tabular}{llllllll}\n\\toprule\nPeriod & Corpus & RC(Major) & RC(Minor) & CTC(Major) & CTC(Minor) & NCTC(Major) & NCTC(Minor) \\\\\n\\midrule\n"

    prev_period = None  # track previous period
    for index, row in combined_df.iterrows():
        if row['period'] != prev_period:
            if prev_period is not None:
                latex_table += "\\midrule\n"  # # add horizontal line except for the last period
            latex_table += f"{row['period']} & {row['corpus']} & {row['RC(Major)']} & {row['RC(Minor)']} & {row['CTC(Major)']} & {row['CTC(Minor)']} & {row['NCTC(Major)']} & {row['NCTC(Minor)']} \\\\\n\n"
            prev_period = row['period']
        else:
            latex_table += f" & {row['corpus']} & {row['RC(Major)']} & {row['RC(Minor)']} & {row['CTC(Major)']} & {row['CTC(Minor)']} & {row['NCTC(Major)']} & {row['NCTC(Minor)']} \\\\\n\n"

    latex_table += "\\bottomrule\n\\end{tabular}"

    latex_table = latex_table.replace("_", " ").replace("nan", "$ -$")

    latex_table_path = os.path.join("results_latex", 'corpora_CI_table.txt')
    with open(latex_table_path, 'w') as file:
        file.write(latex_table)

    return latex_table


###

def basic_corpus_summary_stats(
        metadata: str = "data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.metadata.tsv",
        original_chord: str = "data/distant_listening_corpus_no_missing_keys/distant_listening_corpus.expanded.tsv",
        chord: str = "data/chord_indices.tsv",
        piece: str = "data/piece_indices.tsv"):

    m = pd.read_csv(metadata, sep="\t")
    original_c = pd.read_csv(original_chord, sep="\t")
    post_p = pd.read_csv(piece, sep="\t")
    post_c = pd.read_csv(chord, sep="\t")

    # original data:
    metadata_len = m.shape[0]
    original_corpora = len(m["corpus"].unique())
    original_composers =m["composer"].unique()
    original_c["chord"] = original_c["chord"].astype(str)
    original_chord_num =len(original_c[(~original_c["chord"].isna()) & (original_c["chord"] != '') & (original_c["chord"] != '@none') & (original_c["chord"] != 'nan')])

    min_year = m["composed_end"].unique().min()
    max_year  = m["composed_end"].unique().max()

    print(f'{min_year=}')
    print(f'{max_year=}')

    # after preprocessing:
    discarded_pieces = m[~m['piece'].isin(post_p['piece'])]["piece"].tolist()
    discarded_corpus =  m[~m['piece'].isin(post_p['piece'])]["corpus"].tolist()

    discarded_chords = original_c[~original_c['chord'].isin(post_c['chord'])][["corpus", "piece", "chord"]]

    subcorpora_num = post_p["corpus_id"].max()
    piece_num = post_p["piece_id"].max()
    chord_labels = len(post_c)
    chord_token = len(post_c["chord"].unique())


    # chord level:


    print(f'{metadata_len=}')
    print(f'{original_corpora=}')
    print(f'{original_chord_num=}')
    print(f'{subcorpora_num=}')
    print(f'{piece_num=}')
    # print(f'{year_range=}')
    print(f'{chord_labels=}')
    print(f'{chord_token=}')
    # print(f'{discarded_pieces=}')
    # print(f'{discarded_corpus=}')
    # print(f'{discarded_chords=}')

### fifths range stats

def get_fifths_range_stats(df:pd.DataFrame, df_type: Literal["CombinedMode", "MajorMode", "MinorMode"]):
    total_pieces = len(df)
    diatonic_r = len(df[df["r_fifths_range"]<7])
    diatonic_ct = len(df[df["ct_fifths_range"]<7])
    diatonic_nct = len(df[df["nct_fifths_range"]<7])

    diatonic_r_percentage = diatonic_r/total_pieces
    diatonic_ct_percentage = diatonic_ct/total_pieces
    diatonic_nct_percentage = diatonic_nct/total_pieces

    data = {
        'diatonic_r': [diatonic_r],
        'diatonic_r_percentage': [diatonic_r_percentage],

        'diatonic_ct': [diatonic_ct],
        'diatonic_ct_percentage': [diatonic_ct_percentage],

        'diatonic_nct': [diatonic_nct],
        'diatonic_nct_percentage': [diatonic_nct_percentage]
    }

    result = pd.DataFrame(data)
    print(result)


def get_corpuswise_fifth_range(piece:pd.DataFrame):

    corpus = piece.groupby(["corpus"]).agg(
        year=("corpus_year", "first"),
        max_r_5thRange=("r_fifths_range", "max"),
        max_ct_5thRange=("ct_fifths_range", "max"),
        max_nct_5thRange=("nct_fifths_range", "max"),
        # min_r_5thRange=("r_fifths_range", "min"),
        # min_ct_5thRange=("ct_fifths_range", "min"),
        # min_nct_5thRange=("nct_fifths_range", "min")
    )

    corpus["year"] = corpus["year"].round(0).astype(int)
    corpus = corpus.sort_values(by='year', ascending=True).reset_index()
    long_df = corpus.melt(id_vars=['corpus', 'year'], var_name='FR_type', value_name='FR_value')

    return long_df




if __name__ == "__main__":
    pieces_df = pd.read_csv("data/piece_indices.tsv", sep="\t")
    get_major_minor_pieces_df(df=pieces_df, mode="major", save_df=True)
    get_major_minor_pieces_df(df=pieces_df, mode="minor", save_df=True)

    # major = pd.read_csv("data/majorkey_piece_indices.tsv", sep="\t")
    # minor = pd.read_csv("data/minorkey_piece_indices.tsv", sep="\t")
    #
    # # get_MajorMinor_MeanCI_ttest_table(major, minor)
    # # get_corpus_MajorMinor_CIs_table()
    # # get_major_minor_group_stats(df = pieces_df)
    #
    # # get_fifths_range_stats(pieces_df, df_type="CombinedMode")
    # get_rachmaninoff_CI()