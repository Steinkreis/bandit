import numpy as np
import pandas as pd


class ReplayBandit:
    """Implementierung eines Bandit-Problems mit Replay-Auswertung
    Diese Klasse stellt eine Implementierung eines Bandit-Problems mit Replay-Auswertung dar.
    Sie verwendet eine Liste von aufgezeichneten Ereignissen (logged_events) und erlaubt es,
    Rewards basierend auf Empfehlungen (recommendations) zu erhalten.

    Parameters:
        logged_events (pandas.DataFrame): Ein DataFrame mit aufgezeichneten Ereignissen, das die Spalten
            "movieId" für die Aktionen (Actions) und "rating" für die Rewards enthält.
        batch_size (int, optional): Die Batch-Größe für die Replay-Auswertung. Standardwert ist 1.

    Attributes:
        events (pandas.DataFrame): Ein DataFrame mit den aufgezeichneten Ereignissen, wobei die
            Spalte "rating" in "reward" umbenannt wurde.
        actions (numpy.ndarray): Ein sortiertes Array mit den verfügbaren Aktionen (movieIds) aus den
            aufgezeichneten Ereignissen.
        batch_size (int): Die Batch-Größe für die Replay-Auswertung.
        stream_length (int): Die Länge des Datenstroms, berechnet als die Anzahl der aufgezeichneten
            Ereignisse geteilt durch die Batch-Größe.

    Methods:
        get_rewards(recommendations, n_event):
            Gibt die Rewards basierend auf den gegebenen Empfehlungen zurück.

            Args:
                recommendations (list): Eine Liste von Aktionen (movieIds) als Empfehlungen.
                n_event (int): Der Index des Ereignisses im Datenstrom.

            Returns:
                pandas.DataFrame: Ein DataFrame mit den Rewards für die Empfehlungen, die mit den
                aufgezeichneten Ereignissen des gegebenen Ereignisindexes (n_event) übereinstimmen.

    """
    def __init__(self, logged_events, batch_size=1):
        self.events = logged_events.rename(columns={"rating": "reward"})
        self.actions = np.sort(logged_events["movieId"].unique())
        self.batch_size = batch_size
        self.stream_length = len(self.events) // batch_size

    def get_rewards(self, recommendations, n_event):
        # generate events
        idx = n_event * self.batch_size
        events = self.events.iloc[idx : idx + self.batch_size]
        # keep only events that match with the recommendation slate
        rewards = events[events["movieId"].isin(recommendations)]
        return rewards
