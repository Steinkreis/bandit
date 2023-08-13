from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import beta


class BasePolicy(ABC):
    """Abstrakte Basisklasse als Interface für eine Richtlinie (Policy) in einem Bandit-Problem"""

    def __init__(self, bandit, slate_size, scores_logging):
        self.name = None
        self.slate_size = slate_size
        self.history = pd.DataFrame(data=None, columns=["movieId", "reward"])
        if scores_logging is True:
            self.scores_log = pd.DataFrame(data=None, columns=bandit.actions)
        else:
            self.scores_log = None

    @abstractmethod
    def get_recommendations(self):
        """
        Gibt Empfehlungen für Aktionen (movieIds) basierend auf der Richtlinie zurück.

        Returns:
            list: Eine Liste von Empfehlungen für Aktionen (movieIds).
        """
        ...

    def update(self, rewards):
        """
        Aktualisiert die Richtlinie (Policy) basierend auf den erhaltenen Belohnungen (Rewards).

        Args:
            rewards (pandas.DataFrame): Ein DataFrame mit den Belohnungen (Rewards) und den dazugehörigen
                Aktionen (movieIds).
        """
        # Neue Ereignisse zur Historie hinzufügen
        self.history = self.history.append(rewards, ignore_index=True)

    def _sort_actions(self, scores):
        """Sortiert Aktionen nach Score und mischt Aktionen mit dem gleichen Score.

        Args:
            scores (pandas.Series): Eine Series mit Aktionen (movieIds) als Index und den entsprechenden
                Scores als Werte.

        Returns:
            list: Eine sortierte Liste von Aktionen (movieIds).
        """
        sorted_actions = sorted(
            scores.sample(frac=1).index, key=lambda idx: scores.loc[idx], reverse=True
        )
        return sorted_actions

    def _update_scores_history(self, scores):
        if self.scores_log is not None:
            self.scores_log = self.scores_log.append(
                pd.DataFrame(
                    data=scores.to_numpy().reshape((1, -1)), columns=self.scores_log.columns
                ),
                ignore_index=True,
            )
            self.scores_log = self.scores_log.astype("float")



class EpsilonGreedy(BasePolicy):
    """Epsilon-Greedy Richtlinie für ein Bandit-Problem"""

    def __init__(self, bandit, epsilon, slate_size=1, scores_logging=False):
        super(EpsilonGreedy, self).__init__(bandit, slate_size, scores_logging)
        self.name = f"{epsilon}-Greedy"
        self.epsilon = epsilon
        self.action_values = pd.DataFrame(data=0, columns=["value", "count"], index=bandit.actions)

    def get_recommendations(self):
        """Gibt Empfehlungen für Aktionen (movieIds) basierend auf der Epsilon-Greedy Richtlinie zurück.

        Returns:
            list: Eine Liste von Empfehlungen für Aktionen (movieIds).
        """
        # Aktionen nach Wert sortieren und Aktionen mit dem gleichen Wert mischen
        sorted_actions = self._sort_actions(self.action_values["value"])
        # Empfehlungen auswählen
        if np.random.random() < self.epsilon:
            recs = np.random.choice(
                sorted_actions[self.slate_size :], size=self.slate_size, replace=False
            )
        else:
            recs = sorted_actions[: self.slate_size]
        # Historie der Aktionsscores aktualisieren
        self._update_scores_history(self.action_values["value"])
        return recs

    def update(self, rewards):
        super(EpsilonGreedy, self).update(rewards)
        # Aktionsscores aktualisieren
        for _, (movieId, reward) in rewards.iterrows():
            value = self.action_values.loc[movieId, "value"]
            N = self.action_values.loc[movieId, "count"]
            self.action_values.loc[movieId, "value"] = (value * N + reward) / (N + 1)
            self.action_values.loc[movieId, "count"] += 1



class UCB1(BasePolicy):
    """Upper Confidence Bound (UCB1) Richtlinie für ein Bandit-Problem"""

    def __init__(self, bandit, slate_size=1, scores_logging=False):
        super(UCB1, self).__init__(bandit, slate_size, scores_logging)
        self.name = "UCB1"
        self.action_values = pd.DataFrame(data=0, columns=["value", "count"], index=bandit.actions)

    def get_recommendations(self):
        """Gibt Empfehlungen für Aktionen (movieIds) basierend auf der UCB1 Richtlinie zurück.

        Returns:
            list: Eine Liste von Empfehlungen für Aktionen (movieIds).
        """
        # UCB für jede Aktion berechnen
        current_step = len(self.history)
        if current_step > 0:
            scores = self.action_values["count"].apply(
                lambda N: np.sqrt(2 * np.log(current_step) / N) if N > 0 else np.Inf
            )
            scores = scores + self.action_values["value"]
        else:
            scores = pd.Series(data=np.Inf, index=self.action_values.index)
        # Aktionen nach Score sortieren und Aktionen mit dem gleichen Score mischen
        sorted_actions = self._sort_actions(scores)
        # Empfehlungen auswählen
        recs = sorted_actions[: self.slate_size]
        # Historie der Aktionsscores aktualisieren
        self._update_scores_history(scores)
        return recs

    def update(self, rewards):
        super(UCB1, self).update(rewards)
        # Aktionsscores aktualisieren
        for _, (movieId, reward) in rewards.iterrows():
            value = self.action_values.loc[movieId, "value"]
            N = self.action_values.loc[movieId, "count"]
            self.action_values.loc[movieId, "value"] = (value * N + reward) / (N + 1)
            self.action_values.loc[movieId, "count"] += 1



class ThompsonSampling(BasePolicy):
    """Thompson Sampling Richtlinie für ein Bandit-Problem"""

    def __init__(self, bandit, slate_size=1, scores_logging=False):
        super(ThompsonSampling, self).__init__(bandit, slate_size, scores_logging)
        self.name = "Thompson Sampling"
        self.beta_params = pd.DataFrame(data=1, columns=["alpha", "beta"], index=bandit.actions)

    def get_recommendations(self):
        """Gibt Empfehlungen für Aktionen (movieIds) basierend auf der Thompson
        Sampling Richtlinie zurück.

        Returns:
            list: Eine Liste von Empfehlungen für Aktionen (movieIds).
        """
        # Erwartete Werte für jede Aktion sampeln
        expected_values = pd.Series(
            data=4.5 * beta.rvs(self.beta_params["alpha"], self.beta_params["beta"]) + 0.5,
            index=self.beta_params.index,
        )
        # Aktionen nach Wert sortieren und Aktionen mit dem gleichen Wert mischen
        sorted_actions = self._sort_actions(expected_values)
        # Empfehlungen auswählen
        recs = sorted_actions[: self.slate_size]
        # Historie der Aktionsscores aktualisieren
        self._update_scores_history(expected_values)
        return recs

    def update(self, rewards):
        """Aktualisiert die Thompson Sampling Richtlinie basierend auf den erhaltenen Belohnungen (Rewards).

        Args:
            rewards (pandas.DataFrame): Ein DataFrame mit den Belohnungen (Rewards) und den dazugehörigen
                Aktionen (movieIds).
        """
        super(ThompsonSampling, self).update(rewards)
        # Prior der Aktionsscore-Verteilung aktualisieren
        for _, (movieId, reward) in rewards.iterrows():
            self.beta_params.loc[movieId, "alpha"] += (reward - 0.5) / 4.5
            self.beta_params.loc[movieId, "beta"] += (5.0 - reward) / 4.5
