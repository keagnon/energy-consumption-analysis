import pandas as pd
from sklearn.impute import SimpleImputer

# Analyse des données
def explore_data(data):
    nombre_lignes = len(data)
    print("Nombre de lignes dans le DataFrame :", nombre_lignes)

    # Afficher les premières lignes
    print("Premières lignes du DataFrame :")
    print(data.head())

    # Statistiques descriptives
    print("\nStatistiques descriptives :")
    print(data.describe())
    
    # Vérifier les valeurs manquantes
    print("\nValeurs manquantes :")
    print(data.isnull().sum())
    
    # Identifier les valeurs uniques dans chaque colonne
    print("\nValeurs uniques :")
    for col in data.columns:
        print(f"{col}: {data[col].nunique()} unique values")



def calculate_missing_percentage(df):
    """
    Calcule le pourcentage de valeurs manquantes dans le DataFrame.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.

    Returns:
        float: Le pourcentage de valeurs manquantes.
    """

    total_missing = df.isnull().sum().sum()
    total_values = df.size
    percentage_missing = (total_missing / total_values) * 100

    return percentage_missing


def impute_missing_values(df, strategy='mean'):
    """
    Impute les valeurs manquantes dans le DataFrame en utilisant scikit-learn.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        strategy (str, optional): La stratégie d'imputation à utiliser ('mean', 'median', 'most_frequent', or 'constant'). Par défaut, 'mean' est utilisée.

    Returns:
        pandas.DataFrame: Le DataFrame avec les valeurs manquantes imputées.
    """

    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed

