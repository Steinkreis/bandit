import os
import glob
import pandas as pd
from typing import Callable
from functools import reduce

# Typalias für die Pipeline Schritte -> (pd.DataFrame) Input -> (pd.DataFrame) Output
Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: Preprocessor) -> Preprocessor:
    """Die Funktion compose akzeptiert eine beliebige Anzahl von Funktionen (functions),
    die alle vom Typ Preprocessor sind. Das Sternchen (*) vor functions erlaubt es,
    mehrere Funktionen als einzelne Argumente oder als Tupel von Funktionen zu übergeben.

    Args:
        *functions (Preprocessor): Variable Anzahl von Funktionen,
        die jeweils als Eingabe ein DataFrame (pd.DataFrame) erhalten und ebenfalls eine DataFrame zurückgeben.

    Returns:
        Preprocessor: Gibt eine einzige Funktion zurück,
        die alle übergebenen Funktionen in der Reihenfolge ihrer Übergabe ausführt.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def load_movielens_data(folder_path: str) -> dict[str, pd.DataFrame]:
    """Lädt alle CSV Files eines Ordners in seperate Dataframes und
    gibt diese als dict zurück.

    Args:
        folder_path (str): Ordner in dem die CSV Daten liegen die
        geladen werden sollen.

    Returns:
        dict[str, pd.DataFrame]: CSV Filename als Schlüssel und Dataframe
        als Wert.
    """
    csv_files = glob.glob(folder_path + "*.csv")
    ml_data = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).split(".")[0].replace("-", "_")
        df = pd.read_csv(csv_file)
        ml_data[filename] = df
    return ml_data


def select_movielens_dataset(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Die Funktion erhält alle verfügbaren Dataframes des MovieLens Datensatzes und
    vereint die benötigten Tabellen für das Experiment.

    Args:
        dataframes (dict[str, pd.DataFrame]): CSV Filename als Schlüssel und Dataframe
        als Wert.

    Returns:
        pd.DataFrame: Ein einziges Dataframe mit den benötigten Daten.
    """
    movies = dataframes["movies"]
    ratings = dataframes["ratings"]
    df = ratings.merge(movies, on="movieId", how="left")
    return df


def ohe_genres(df: pd.DataFrame) -> pd.DataFrame:
    """One Hot Encoding für die Genres

    Args:
        df (pd.DataFrame): Dataframe mit Genre Spalte

    Returns:
        pd.DataFrame: Fügt die OHE Spalten hinzu und gibt das
        neue Dataframe zurück.
    """
    return df.join(df["genres"].str.get_dummies().astype(int).add_suffix("_ohe"))


def extract_year(df: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert die Information für das Jahr aus der Titel Spalte, weil dort
    idr. das Produktionsjahr vermerkt ist. Ist dort kein Jahr zu finden erhält man
    für diesen Film ein NaN

    Args:
        df (pd.DataFrame): Dataframe mit Titel Spalte

    Returns:
        pd.DataFrame: Fügt die Spalte für das Produktionsjahr hinzu und gibt
        das neue Dataframe zurück.
    """
    df["year"] = df["title"].str.findall("\((\d{4})\)").str.get(0)
    return df


def like_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Erstellt einen binären Indikator der Aussagt ob ein User
    den Film gemocht hat oder nicht. Alles unter einer Bewertung
    von 4 Sternen gilt als nicht gemocht.

    Args:
        df (pd.DataFrame): Dataframe mit der Rating Spalte.

    Returns:
        pd.DataFrame: Erstellt die binäre Spalte liked und
            gibt das neue Dataframe zurück
    """
    df["liked"] = (df["rating"] >= 4).astype(int)
    return df


def preprocessing(df: pd.DataFrame, processing_steps: list[Preprocessor]) -> pd.DataFrame:
    """Führt die Preprocesing Pipeline aus und gibt das bearbeitete
    Dataframe zurück.

    Args:
        df (pd.DataFrame): Dataframe das bearbeitet werden soll.
        processing_steps (list[Preprocessor]): Liste mit den processing steps,
        die ausgeführt werden sollen.

    Returns:
        pd.DataFrame: Bearbeitetes Dataframe, was durch die Pipeline gelaufen ist.
    """
    pipeline = compose(*processing_steps)
    processed_df = pipeline(df)
    return processed_df


def main():
    # Setup um Daten zu laden
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_folder_path = os.path.join(current_directory, "../MovieLens-25M-Dataset/")
    ml_data = load_movielens_data(folder_path=csv_folder_path)
    df = select_movielens_dataset(dataframes=ml_data)

    # Preprocessing
    processing_steps = [
        extract_year,
        like_indicator,
        ohe_genres,
    ]
    df = preprocessing(df, processing_steps)
    df.to_csv(csv_folder_path + "processed-data.csv", index=False)


if __name__ == "__main__":
    main()
