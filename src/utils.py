import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from policy import BasePolicy
from matplotlib.ticker import FormatStrFormatter
from typing import Union
import seaborn as sns


# Data ################################################################


def read_data():
    """Liest die Daten aus einer CSV-Datei und gibt ein DataFrame mit 2 Spalten zurück: movieId (Aktionen) und rating (Belohnung).

    Returns:
        pandas.DataFrame: Ein DataFrame mit den Spalten movieId und rating.
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "MovieLens-25M-Dataset", "ratings.csv")
    data = pd.read_csv(file_path, header=0, usecols=["movieId", "rating"])
    return data


def preprocess_data(data: pd.DataFrame, num_ratings: int, num_movies: Union[None, int]) -> pd.DataFrame:
    """Bereitet die Daten vor, indem für jede movieId/Aktion eine gleichmäßig verteilte Stichprobe erstellt wird.

    Args:
        data (pandas.DataFrame): Ein DataFrame mit den Spalten movieId und rating.
        num_ratings (int): Die Anzahl der Bewertungen, die für jede movieId/Aktion in der Stichprobe enthalten sein sollen.
        num_movies (int or None, optional): Die Anzahl der Filme, die in der Stichprobe enthalten sein sollen. Standardwert ist None.

    Returns:
        pandas.DataFrame: Das vorverarbeitete DataFrame mit den gleichmäßig verteilten movieId/Aktionen und Bewertungen.
    """
    # Filme filtern, die weniger als num_ratings Bewertungen haben
    movies = data.groupby("movieId").agg({"rating": "count"})
    if num_movies is not None:
        movies_to_keep = (
            movies[(movies["rating"] >= num_ratings)].sample(n=num_movies, random_state=12).index
        )
    else:
        movies_to_keep = movies[(movies["rating"] >= num_ratings)].index
    data = data[data["movieId"].isin(movies_to_keep)]
    # Für jeden Film eine zufällige Stichprobe der Größe num_ratings nehmen
    data = data.groupby("movieId").sample(n=num_ratings, random_state=42)
    # Zeilen mischen, um den Datenstrom zu randomisieren
    data = data.sample(frac=1, random_state=42)
    # Index zurücksetzen, um einen pseudo zeitlichen Index zu erstellen
    data = data.reset_index(drop=True)
    return data



def get_data(num_ratings, num_movies=None):
    data = read_data()
    data = preprocess_data(data, num_ratings, num_movies)
    return data


# Plot ################################################################


def plot_rewards(*policies: BasePolicy, title=None):
    """Plottet die kumulativen Belohnungen und durchschnittlichen Belohnungen für verschiedene Richtlinien.

    Args:
        *policies (BasePolicy): Eine Liste von Richtlinien (Policy)-Objekten.
        title (str, optional): Der Titel für die Plots. Standardwert ist None.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    fig.suptitle(title)
    for policy in policies:
        # Kumulative Belohnungen erhalten
        cumsum_rewards = policy.history.reward.cumsum()
        # Durchschnittliche Belohnungen erhalten
        timesteps = np.arange(len(cumsum_rewards)) + 1
        avg_rewards = cumsum_rewards / timesteps
        # Plots
        ax1.plot(timesteps, avg_rewards, label=policy.name)
        ax2.plot(timesteps, cumsum_rewards, label=policy.name)
    
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.set_xlabel("Zeitschritt")
    ax1.set_ylabel("Durchschnittliche Belohnung")
    ax1.legend(loc="lower right")
    
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.set_xlabel("Zeitschritt")
    ax2.set_ylabel("Kumulative Belohnung")
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()


def plot_action_values(*policies: BasePolicy):
    """Plottet die Aktionsscores für verschiedene Richtlinien.

    Args:
        *policies (BasePolicy): Eine Liste von Richtlinien (Policy)-Objekten.
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(policies), figsize=(15, 5), squeeze=False)
    fig.suptitle("Aktionsscores")
    axs = axs.ravel()
    for i, policy in enumerate(policies):
        cbar = True if i == len(axs) - 1 else False
        sns.heatmap(
            policy.scores_log.T,
            ax=axs[i],
            vmin=2.5,
            vmax=5,
            cmap="hot",
            cbar=cbar,
            xticklabels=1000,
            yticklabels=False,
        )
        axs[i].set_xlabel("Zeitschritt")
        axs[i].title.set_text(policy.name)
    axs[0].set_ylabel("movieId")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = get_data(10000)
    # Visualisierung der Bewertungsstatistiken
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # Histogramm der Bewertungserwartung pro Film
    mean_ = data.groupby("movieId").agg({"rating": "mean"})
    mean_.hist(grid=False, bins=24, ax=axs[0])
    axs[0].set_title("Bewertungserwartung")
    # Histogramm der Bewertungsvarianz pro Film
    var_ = data.groupby("movieId").agg({"rating": "var"})
    var_.hist(grid=False, bins=24, ax=axs[1])
    axs[1].set_title("Bewertungsvarianz")
    # Streudiagramm von var_ gegen mean_
    axs[2].scatter(mean_, var_, alpha=0.5)
    axs[2].set_title("Bewertungsvarianz vs. Bewertungserwartung")
    plt.tight_layout()
    plt.show()

