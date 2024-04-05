import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class PreprocessingData:
    """
    Class for preprocessing any data.
    """

    def __init__(self):
        pass

    def remove_outliers(self, dataframe, column):
        """
        Remove outliers from a specific column of a DataFrame using Isolation Forest.
        -------------------------------------------------------------
        params:
            dataframe: Pandas DataFrame containing the data.
            column (str): The name of the column to process.

        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        isolation_forest = IsolationForest(contamination='auto', random_state=42)
        isolation_forest.fit(dataframe[[column]])
        anomalies = isolation_forest.predict(dataframe[[column]])
        dataframe_filtered = dataframe[anomalies == 1]

        return dataframe_filtered

        def remove_nan_rows(df, column_name):
            """
            Supprime les lignes avec des valeurs NaN dans une colonne spécifique.

            Args:
                df (pandas.DataFrame): Le DataFrame contenant les données.
                column_name (str): Le nom de la colonne à vérifier.

            Returns:
                pandas.DataFrame: Le DataFrame filtré.
            """
            df = df.dropna(subset=[column_name])
            return df

        def drop_column(dataframe, column_name):
            """
            Supprime une colonne spécifiée d'un DataFrame pandas.

            Parameters:
            - dataframe: DataFrame pandas à partir duquel la colonne sera supprimée.
            - column_name: chaîne de caractères représentant le nom de la colonne à supprimer.

            Returns:
            Le DataFrame après la suppression de la colonne.
            """
            if column_name in dataframe.columns:
                dataframe.drop(columns=[column_name], inplace=True)
            else:
                print(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")

        def convert_to_datetime(dataframe, date_column, date_format):
            """
            Convertit une colonne de dates dans un DataFrame pandas au format datetime.

            Parameters:
            - dataframe: DataFrame pandas contenant la colonne de dates.
            - date_column: chaîne de caractères représentant le nom de la colonne de dates à convertir.
            - date_format: chaîne de caractères représentant le format des dates dans la colonne à convertir.

            Returns:
            Affiche les 5 premières lignes du DataFrame après conversion.
            """
            if date_column in dataframe.columns:
                dataframe[date_column] = pd.to_datetime(dataframe[date_column], format=date_format)
                return dataframe.head()
            else:
                print(f"La colonne '{date_column}' n'existe pas dans le DataFrame.")
                return None

