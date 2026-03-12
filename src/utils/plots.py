import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_distribution_over_time(data: pd.DataFrame, targets: list[str], by: str, title: str) -> None:
    """
    Plot that compares the evolution of a given distribution overtime
    :param data: dataframe where each column is a variable and each row is an observation
    :param targets: list of strings containing the targets distributions, if more than 1 target, the plots will be side by side
    :param by: Column used to group distributions
    :return: A plot
    """
    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):

        data_long = data[targets + [by]].melt(
            id_vars=by,
            value_vars=targets,
            var_name="target",
            value_name="price"
        )
        pal = sns.cubehelix_palette(n_colors=data[by].nunique(), rot=-.25, light=.7)
        g = sns.FacetGrid(data=data_long,
                          row=by,
                          hue=by,
                          col="target",
                          aspect=15,
                          height=0.5,
                          palette=pal)

        g.map(sns.kdeplot, "price",
              clip=(-100, 200),
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)

        g.map(sns.kdeplot, "price", clip_on=False, clip=(-100, 200), color="w", lw=2, bw_adjust=.5)

        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "price")
        g.figure.subplots_adjust(hspace=-.25)

        for ax, col_name in zip(g.axes[-1], g.col_names):
            ax.set_xlabel(col_name.replace("_", " "))

        g.set_titles("")
        g.figure.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        plt.show()


def plot_top_correlated_features(data: pd.DataFrame, target: str, nb_features: int, subset: list[str]|None = None) -> None:
    """
    Scatter plots top n features with target, with a regression line
    :param data: dataframe where each column is a variable and each row is an observation
    :param target: name of the target column with which the correlation is computed
    :param nb_features: number of scatter plots to show
    :return: a plot
    """
    if subset is None:
        num_cols = list(set(data.select_dtypes(["int", "float"]).columns.tolist()) - {"Long_Imbalance_Price",
                                                                                      "Short_Imbalance_Price"})
    else:
        num_cols = subset

    corr_with_target = (data[num_cols]
                        .corrwith(data[target])
                        .abs()
                        .sort_values(ascending=False))

    top_features = corr_with_target.head(nb_features)

    nrows = max(1, math.ceil(nb_features / 5))

    fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20, 5 * nrows))
    fig.suptitle(f"Top {nb_features} Numerical Feature vs {target}", fontsize=16, fontweight="bold")

    for ax, feat in zip(axes.flatten(), top_features.index):
        sns.regplot(data=data,
                        x=feat,
                        y=target,
                        scatter_kws={"alpha":0.7, "s":15},
                        line_kws={"color":"red", "lw":2},
                        ci=None,
                        ax=ax)
        ax.set_title(f"{feat}\nr={corr_with_target[feat]:.2f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes.flatten()[nb_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def consecutive_nan(data: pd.DataFrame, feature: str):
    """
    diff().ne(0) <=> diff() != 0, diff calculates the difference of consecutive numbers
    so if null_mask.astype(int).diff() != 0 that means there's a change, ne(0) detects the changes in groups

    :param data:
    :param feature:
    :return:
    """
    null_mask = data[feature].isna()
    null_group = null_mask.astype(int).ne(0).cumsum()
    return null_mask.groupby(null_group).sum()

def missing_summary(data, label='Train'):
    miss = data.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_pct = (miss / len(data) * 100).round(2)
    return pd.DataFrame({'Missing': miss, 'Missing (%)': miss_pct}).rename_axis(f'{label} — Column')


def plot_acf_pacf(data: pd.DataFrame, features: list[str], nb_of_lags: int):
    fig, axes = plt.subplots(nrows=len(features), ncols=2, figsize=(20, 5 * len(features)), squeeze=False)

    for ax, feat in zip(axes, features):
        plot_acf(x=data[feat],
                 ax=ax[0],
                 lags=nb_of_lags,
                 alpha=.05)
        feat_name = feat.replace("_", " ")
        ax[0].set_title(f"{feat_name} ACF")
        plot_pacf(x=data[feat],
                  ax=ax[1],
                  lags=nb_of_lags,
                  alpha=.05)
        ax[1].set_title(f"{feat_name} PACF")

    plt.tight_layout()
    plt.show()


