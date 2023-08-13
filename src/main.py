from tqdm import tqdm
from bandit import ReplayBandit
from policy import EpsilonGreedy, UCB1, ThompsonSampling
from utils import get_data, plot_rewards, plot_action_values



def main():
    ######################################################################

    NUM_RATINGS = 10000
    NUM_MOVIES = None
    SLATE_SIZE = 5
    BATCH_SIZE = 100
    STREAM_LENGTH = 50000
    SCORES_LOG = False  # Loggen der Filmbewertungen

    ######################################################################


    # Daten abrufen
    logged_events = get_data(NUM_RATINGS, NUM_MOVIES)

    # Bandit-Problem instanziieren
    bandit = ReplayBandit(logged_events, BATCH_SIZE)
    STREAM_LENGTH = bandit.stream_length
    title = "Belohnungen für Bandit-Problem mit Replay-Auswertung"

    print(f"ANZAHL DER FILME/AKTIONEN: {len(bandit.actions)}")
    print()

    # Richtlinien (Policies) instanziieren
    policies = [
        EpsilonGreedy(bandit, epsilon=0.1, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
        UCB1(bandit, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
        ThompsonSampling(bandit, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
    ]

    # Richtlinien (Policies) auswerten
    for policy in policies:
        print(f"RICHTLINIE (POLICY): {policy.name}")
        for i in tqdm(range(STREAM_LENGTH), ascii=True):
            recs = policy.get_recommendations()
            rewards = bandit.get_rewards(recs, i)
            policy.update(rewards)
        print(f"LÄNGE DER HISTORIE: {len(policy.history)}")
        print()

    # Ergebnisse plotten
    plot_rewards(*policies, title=title)
    if SCORES_LOG is True:
        plot_action_values(*policies)
