**Filmempfehlungen mithilfe von Multi-Armed-Bandit Algorithmen**

In diesem Projekt wird ein Experiment mit den MovieLens 25M Daten durchgeführt. Hierbei wird das Potenzial von  MAB-Algorithmen als Recommender Systeme für Filmempfehlungen untersucht.

---

**Voraussetzungen**

Bevor mit dem Projekt begonnen werden kann, muss sichergestellt werden, dass die folgenden Schritte befolgt werden:

1. Der MovieLens 25M Datensatz muss als .zip von hier https://grouplens.org/datasets/movielens/25m/ heruntergeladen werden.
2. Im Root-Verzeichnis des Projekts muss ein Ordner namens "MovieLens-25M-Dataset" angelegt werden.
3. Die heruntergeladenen Daten (.csv Dateien) müssen im erstellten Ordner "MovieLens-25M-Dataset" gespeichert werden ohne sie umzubenennen. Die Ordnerstruktur des Projektes sollte nun folgendermaßen aussehen:

```bash
Projektordner/
│   imgs
│   ├── ...
│
├── MovieLens-25M-Dataset/
│   ├── (hier befinden sich die MovieLens 25M Daten)
│
│   src
│   ├── ...
│
│   Ausarbeitung.ipynb
│   README.MD
│   requirements.txt
```

4. Die erforderlichen Python-Pakete müssen in eine virtuelle Umgebung installiert werden, indem folgender Befehl vom Root-Verzeichnis ausgeführt wird:

```bash
pip install -r requirements.txt
```