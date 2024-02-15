import pandas as pd

def load_data(file_path):
    """
    Chargement des données à partir d'un fichier CSV.

    Args:
        file_path (str): Chemin du fichier CSV.

    Returns:
        pandas.DataFrame: Le DataFrame contenant les données du fichier CSV.
    """
    df = pd.read_csv(file_path, sep=';', header=0, index_col=False)
    return df
