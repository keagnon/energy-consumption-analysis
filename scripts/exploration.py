import pandas as pd

class DataExploration:
    """
    Class to perform exploration of any dataset.
    """

    def __init__(self):
        pass

    def display_columns_by_type(data):
        """
        Display all columns and the first 5 lines of the dataset
        Display columns and their types
        -------------------------------------------------------------
        params:
            data (DataFrame): DataFrame.
        """
        print(data.head())

        # Nombre de lignes
        nombre_lignes = len(data)

        # Nombre de colonnes
        nombre_colonnes = len(data.columns)

        print("Nombre de lignes :", nombre_lignes)
        print("Nombre de colonnes :", nombre_colonnes)
        print("--------------------------------------------------")

        for dtype in data.dtypes.unique():
            print(f"Columns of type {dtype}:")
            print(list(data.select_dtypes(include=[dtype]).columns))
            print()

    def explore_data(data):
        """
        Perform data exploration.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns_to_drop: List of columns to drop from the DataFrame (default is None).
        """
        print(data.head(5))
        print("data shape")
        print(data.shape)
        print("Data columns")
        print(data.columns)
        print("Data information:")
        print(data.info())
        print("\nDescriptive statistics:")
        print(data.describe())

        def print_column_types(df):
            """
            Affiche le type de chaque colonne dans le DataFrame.

            Args:
                df (pandas.DataFrame): Le DataFrame contenant les données.

            Returns:
                None
            """
            print("Type de chaque colonne :")
            print(df.dtypes)

    def display_unique_values(data):
        """
        Display unique values of each column in a DataFrame.
        -------------------------------------------------------------
        params:
            data : DataFrame containing the data.
        """
        for column in data.columns:
            print("Column:", column)
            print(data[column].unique())
            print()

    def display_missing_values(data):
        """
        Display the number of missing values per column in the DataFrame.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
        """
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found.")
        else:
            print("Missing values per column:")
            print(missing_values[missing_values > 0])

    def handle_missing_values(df):
        """
        Gère les valeurs manquantes dans un DataFrame.

        Si le pourcentage de valeurs manquantes dans une colonne dépasse 55 %,
        les lignes contenant des valeurs manquantes dans cette colonne seront supprimées.
        Sinon, le pourcentage de valeurs manquantes dans cette colonne sera simplement imprimé.

        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données.

        Returns:
            pandas.DataFrame: Le DataFrame après traitement des valeurs manquantes.
        """
        total_rows = len(df)
        for column in df.columns:
            missing_values = df[column].isnull().sum()
            missing_percentage = (missing_values / total_rows) * 100
            print(f"Percentage of missing values in column '{column}': {missing_percentage:.1f}%")

            if missing_percentage > 55:
                print(f"Removing rows with missing values in column '{column}'...")
                df = df.dropna(subset=[column])
                print(f"{len(df)} rows remaining after removing missing values in column '{column}'.")
        return df



