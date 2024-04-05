import pandas as pd

def load_data(filepath, date_column=None, date_format=None):
    """
    Charge des données depuis un fichier dans un DataFrame pandas.

    Args:
        filepath (str): Chemin du fichier à charger.
        date_column (str, optional): Nom de la colonne contenant les dates à convertir. Default est None.
        date_format (str, optional): Format des dates dans la colonne spécifiée. Nécessaire si date_column est spécifié. Default est None.

    Returns:
        pandas.DataFrame: DataFrame contenant les données chargées.
    """
    # Charger les données
    df = pd.read_csv(filepath,nrows =2000,sep=';', header=0, index_col=False)
    
    # Convertir la colonne de date si spécifié
    if date_column and date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    
    return df
