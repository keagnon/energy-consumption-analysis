import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from scripts.load_data import load_data
from scripts.explore_data import explore_data,calculate_missing_percentage,impute_missing_values


def main():

    # Chemin du fichier CSV
    file_path = 'dataset/consommation-quotidienne-brute.csv'

    # Chargement des données
    data = load_data(file_path)

    # Analyse des données
    explore_data(data)
    percentagege=calculate_missing_percentage(data)
    print(percentagege)
    #impute_missing_values(data, strategy='mean')

    # Visualisation des données
    #visualize_data(data)

    # Prétraitement des données
    #data = preprocess_data(data)

if __name__ == "__main__":
    main()
