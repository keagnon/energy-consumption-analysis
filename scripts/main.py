import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, widgets

df = pd.read_csv('dataset/consommation-quotidienne-brute.csv',sep=';', header=0, index_col=False)
print(df)
print(df.shape)
print("---------------------------------------------------")
df.drop(columns=['Date - Heure'],inplace=True)
print(df)
print(df.shape)

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


# Utilisation des fonctions pour chaque colonne
df=remove_nan_rows(df, 'Statut - GRTgaz')
df=remove_nan_rows(df, 'Statut - Teréga')

print(df)

df['Date'] = pd.to_datetime(data['Date'])

# Fonction pour créer différentes visualisations
def create_visualization(visualization_type):
    if visualization_type == 'Consommation brute de gaz au fil du temps':
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='Consommation brute gaz totale (MW PCS 0°C)', data=data)
        plt.title('Consommation brute de gaz au fil du temps')
        plt.xlabel('Date')
        plt.ylabel('Consommation brute de gaz (MW PCS 0°C)')
        plt.xticks(rotation=45)
        plt.show()

    elif visualization_type == 'Distribution de la consommation brute de gaz':
        plt.figure(figsize=(8, 6))
        sns.histplot(data['Consommation brute gaz totale (MW PCS 0°C)'], bins=20, kde=True)
        plt.title('Distribution de la consommation brute de gaz')
        plt.xlabel('Consommation brute de gaz (MW PCS 0°C)')
        plt.ylabel('Nombre d\'occurrences')
        plt.show()

    elif visualization_type == 'Consommation brute de gaz par heure':
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Heure', y='Consommation brute gaz totale (MW PCS 0°C)', data=data)
        plt.title('Consommation brute de gaz par heure')
        plt.xlabel('Heure')
        plt.ylabel('Consommation brute de gaz (MW PCS 0°C)')
        plt.xticks(rotation=45)
        plt.show()

    elif visualization_type == 'Relation entre la consommation brute de gaz et d\'électricité':
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Consommation brute gaz totale (MW PCS 0°C)', y='Consommation brute électricité (MW) - RTE', data=data)
        plt.title('Relation entre la consommation brute de gaz et d\'électricité')
        plt.xlabel('Consommation brute de gaz (MW PCS 0°C)')
        plt.ylabel('Consommation brute d\'électricité (MW)')
        plt.show()

    elif visualization_type == 'Consommation brute de gaz totale par statut de GRTgaz':
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Statut - GRTgaz', y='Consommation brute gaz totale (MW PCS 0°C)', data=data)
        plt.title('Consommation brute de gaz totale par statut de GRTgaz')
        plt.xlabel('Statut - GRTgaz')
        plt.ylabel('Consommation brute de gaz (MW PCS 0°C)')
        plt.show()

    elif visualization_type == 'Consommation brute totale par mois':
        data['Mois'] = data['Date'].dt.month
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Mois', y='Consommation brute totale (MW)', data=data, estimator=sum)
        plt.title('Consommation brute totale par mois')
        plt.xlabel('Mois')
        plt.ylabel('Consommation brute totale (MW)')
        plt.show()

# Liste des types de visualisations disponibles
visualization_types = [
    'Consommation brute de gaz au fil du temps',
    'Distribution de la consommation brute de gaz',
    'Consommation brute de gaz par heure',
    'Relation entre la consommation brute de gaz et d\'électricité',
    'Consommation brute de gaz totale par statut de GRTgaz',
    'Consommation brute totale par mois'
]

# Créer un menu interactif
interact(create_visualization, visualization_type=widgets.Dropdown(options=visualization_types, value=visualization_types[0]))
