from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def corpus_3features_bar_plot(df: pd.DataFrame):
    longform_df = pd.melt(df, id_vars=['corpus', "corpus_year"], value_vars=['mean_r_chromaticity',
                                                                             'mean_ct_chromaticity',
                                                                             'mean_nct_chromaticity'],
                          var_name="chromaticity_type", value_name="value")

    sns.barplot(longform_df, x="corpus_year", y="value", hue="chromaticity_type")
    plt.show()


class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0 / self.n)

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 7), angle=angle, labels=label)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 6)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


def piece_radar_plot(df: pd.DataFrame, corpus: str, piece: str):
    condition_corpus = df["corpus"] == corpus
    condition_piece = df["piece"] == piece

    piece_df = df[condition_corpus & condition_piece]

    chromaticities = [piece_df["scaled_r_chromaticity"].values[0],
                      piece_df["scaled_ct_chromaticity"].values[0],
                      piece_df["scaled_nct_chromaticity"].values[0]]

    print(f'{chromaticities=}')

    categories = ["RC", "CTC", "NCTC"]
    full_range = range(1, 37)
    labels = [
        full_range[0::6],
        full_range[0::6],
        full_range[0::6],
    ]
    matplotlib.style.use('ggplot')  # interesting: 'bmh' / 'ggplot' / 'dark_background'

    fig = plt.figure(figsize=(8, 8))

    radar = Radar(fig, categories, labels)
    radar.plot(chromaticities, '-', lw=2, color='b', alpha=0.4, label='first')
    radar.ax.legend()

    fig.show()


def barplot_corpus_stacked_features(df: pd.DataFrame):
    fig = plt.figure(figsize=(12, 8))

    chromaticities_df = df[
        ["mean_r_chromaticity", "mean_ct_chromaticity", "mean_nct_chromaticity", "corpus_id", "corpus"]]
    chromaticities_df.plot(x="corpus_id", kind="bar", stacked=True, color=['#CE2227', '#F47B5D', '#E9A29E'])

    fig.tight_layout()
    fig.show()


def featuers2_scatter_text(df: pd.DataFrame):
    fig = px.scatter(df, x="mean_ct_chromaticity", y="mean_r_chromaticity", text="corpus")
    # fig.update_traces(textposition='top center')
    fig.update_traces(texttemplate='%{text}', textfont_size=12)  # Adjust textfont_size as needed

    fig.update_layout(title_text='Chromaticism', title_x=0.5)
    fig.show()


def corpus_histogram(corpus_path: str = "data/piecelevel_chromaticities.tsv"):
    df = pd.read_csv(corpus_path, sep="\t")

    sns.histplot(data=df, x="piece_year", bins=20, stat="density")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    plt.savefig("figs/piece_distribution.pdf")


def corpus_features_histogram(features: List, corpus_path: str = "data/core_cl_chromaticities.tsv"):
    df = pd.read_csv(corpus_path, sep="\t")

    # fig = plt.figure(figsize=(12, 8))

    fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    _corpus = df["corpus"].tolist()
    corpus = [x.split("_")[0] for x in _corpus]

    for i, f in enumerate(features):
        feature_chromaticity = df[f].tolist()
        ax[i].bar(corpus, feature_chromaticity)

        ax[i].set_title(f'{f}')
        ax[i].set_xticklabels(ax[i].get_xticks(), rotation=90)

    fig.tight_layout()
    fig.show()


def corpus_summary_table(piece_level_df_path: str = "data/all_subcorpora/all_subcorpora.metadata.tsv"):
    df = pd.read_csv(piece_level_df_path, sep="\t")

    result_df = df.groupby(by=["corpus"]).agg(
        Corpus=("corpus", "first"),
        Pieces=("piece", "count"),
        MinYear=("composed_end", "min"),
        MaxYear=("composed_end", "max")

    )
    print(result_df)
    # latex_table = result_df.to_latex(index=False)
    # print(latex_table)


def barplot_corpus_variance_5thsRange(corpuslevel_df_path: str = "data/core_cl_chromaticities.tsv"):
    df = pd.read_csv(corpuslevel_df_path, sep="\t")
    X = df["corpus_id"].tolist()
    Y_r = df["root_fifths_range"].tolist()
    Y_ct = df["ct_fifths_range"].tolist()
    Y_nct = df["nct_fifths_range"].tolist()
    plt.plot(X, Y_r, "r-", X, Y_ct, "b-", X, Y_nct, "g-")
    plt.show()


def violin_plot_corpus(df: pd.DataFrame, feature: str, selected_corpus: List[str]):
    subdf = df[df["corpus"].isin(selected_corpus)]

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(data=subdf, x="corpus", y=feature, palette="Set3")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.show()


def grieg_corpus(df: pd.DataFrame):
    greig_df = df[df["corpus"] == "grieg_lyric_pieces"]
    greig_df_long_5thrange = pd.melt(greig_df, id_vars=["piece"],
                                     value_vars=["root_fifths_range", "ct_fifths_range", "nct_fifths_range"],
                                     var_name="fifths_range_type", value_name="fifths_range_value")

    greig_df_long_chromaticity = pd.melt(greig_df, id_vars=["piece"],
                                         value_vars=["mean_r_chromaticity", "mean_ct_chromaticity",
                                                     "mean_nct_chromaticity"],
                                         var_name="chromaticity_type", value_name="chromaticity_value")

    sns.violinplot(data=greig_df_long_chromaticity, x='chromaticity_type', y='chromaticity_value',
                   bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
    plt.show()


def corpus_chromaticity_scatter(corpus_path: str = "old_data/core_pl_chromaticities.tsv"):
    df = pd.read_csv(corpus_path, sep="\t")

    ax1 = plt.subplot(221)
    ax1.scatter(x=df["mean_r_chromaticity"].tolist(),
                y=df["mean_ct_chromaticity"].tolist(),
                s=5, c=df["corpus_id"], cmap="Purples")
    ax1.set_xlabel('RC')
    ax1.set_ylabel('CTC')
    # ax1.set_title("r-ct")

    ax2 = plt.subplot(222)
    ax2.scatter(x=df["mean_r_chromaticity"].tolist(),
                y=df["mean_nct_chromaticity"].tolist(),
                s=5, c=df["corpus_id"], cmap="Blues")
    ax2.set_xlabel("RC")
    ax2.set_ylabel('NCTC')
    # ax2.set_title("r-nct")

    ax3 = plt.subplot(212)
    ax3.scatter(x=df["mean_ct_chromaticity"].tolist(),
                y=df["mean_nct_chromaticity"].tolist(),
                s=5, c=df["corpus_id"], cmap="Oranges")
    ax3.set_xlabel('CTC')
    ax3.set_ylabel('NCTC')
    # ax3.set_title("ct-nct")
    plt.show()


def plotly_corpus_3features_scatter(df: pd.DataFrame, features: List[str], hue_by: List):
    # color = df["piece_year"].tolist()
    color = hue_by

    fig = go.Figure(data=[go.Scatter3d(
        x=df[features[0]],
        y=df[features[1]],
        z=df[features[2]],
        mode='markers',
        marker_symbol='diamond',
        marker=dict(
            size=4,
            color=color,  # set color to an array/list of desired values
            colorscale='darkmint',  # choose a colorscale
            opacity=0.6
        ),
        text=df["corpus"],
        customdata=df[['corpus_year']],
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "year: %{customdata[0]}<br>" +
        "root_chromaticity: %{x}<br>" +
        "ct_chromaticity: %{y}<br>" +
        "nct_chromaticity: %{z}<br>" +
        "<extra></extra>",
    )])

    fig.update_layout(title=f'{features} chromaticity 3D plot', title_x=0.5)

    fig.update_layout(scene=dict(
        xaxis_title=f'{features[0]}',
        xaxis=dict(
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ),
        yaxis_title=f'{features[1]}',
        yaxis=dict(
            backgroundcolor="rgb(230, 200,230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white"),
        zaxis_title=f'{features[2]}',
        zaxis=dict(
            backgroundcolor="rgb(230, 230,200)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white")))
    fig.show()
    return fig


def histogram_composition_distribution(df:pd.DataFrame):
    # global variables
    DPI = 300
    fs = 30
    lines = sns.histplot(
        data=df,
        x=df["piece_year"].tolist(),
        kde=True,
        stat="probability",
        bins=30
    ).get_lines()
    xs, ys = lines[0].get_data()

    print(f'{xs=}')

    mininds = []
    a, b = -1, -1
    for i, c in enumerate(ys):
        if a > b and b < c:
            mininds.append(i)
        a, b = b, c
    _, t1, t2, t3, t4, _ = xs[mininds]

    for b in [t1, t2, t3, t4]:
        plt.axvline(b, c="gray", ls="--", lw=3, zorder=-2)

    plt.xlabel("year", fontsize=fs)
    plt.ylabel("probability", fontsize=fs)
    # plt.savefig("img/Figure1.pdf", dpi=DPI)
    plt.show()



if __name__ == "__main__":
    corpus_chromaticity_scatter()
    result_df = pd.read_csv("old_data/piecelevel_chromaticities.tsv", sep="\t")

    # histogram_composition_distribution(result_df)



    # corpus_df = pd.read_csv("data/corpuslevel_chromaticities.tsv", sep="\t")
    # hue_by = corpus_df["corpus_id"].to_numpy()
    # p=plotly_corpus_3features_scatter(corpus_df, features=["mean_r_chromaticity",
    #                                                      "mean_ct_chromaticity",
    #                                                      "mean_nct_chromaticity"], hue_by=hue_by)
    # p.write_html("figs/corpus_level_3D_scatter.html")
    # grieg_corpus(result_df)
    #
    # corpus_df = pd.read_csv("data/corpuslevel_chromaticities.tsv", sep="\t")
    # piece_radar_plot(result_df, corpus="bartok_bagatelles", piece="op06n12")
    # featuers2_scatter_text(corpus_df)

    # corpus_features_histogram(features=["mean_r_chromaticity", "mean_ct_chromaticity", "mean_nct_chromaticity"])

    # violin_plot_corpus(result_df, feature="mean_r_chromaticity",
    #                    selected_corpus=["corelli",
    #                                     # "couperin_clavecin", "handel_keyboard",
    #                                     "bach_solo",
    #                                     "bach_en_fr_suites",
    #                                     # "couperin_concerts",
    #                                     # "pergolesi_stabat_mater","scarlatti_sonatas"
    #                                     ])
