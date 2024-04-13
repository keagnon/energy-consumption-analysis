import pandas as pd
import numpy as np

def enrich_dataset_with_regions(file_path):
    # Charger les données
    data = pd.read_csv(file_path)

    # Liste des régions françaises et leurs coordonnées
    regions = [
        "Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire",
        "Corse", "Grand Est", "Hauts-de-France", "Île-de-France", "Normandie", "Nouvelle-Aquitaine",
        "Occitanie", "Pays de la Loire", "Provence-Alpes-Côte dAzur"
    ]

    # Dictionnaire des coordonnées par région
    region_to_coords = {
        'Auvergne-Rhône-Alpes': {'lat': 45.4473, 'lon': 4.3859},
        'Bourgogne-Franche-Comté': {'lat': 47.2805, 'lon': 4.9994},
        'Bretagne': {'lat': 48.2020, 'lon': -2.9326},
        'Centre-Val de Loire': {'lat': 47.7516, 'lon': 1.6751},
        'Corse': {'lat': 42.0396, 'lon': 9.0129},
        'Grand Est': {'lat': 48.6998, 'lon': 6.1878},
        'Hauts-de-France': {'lat': 50.4801, 'lon': 2.7937},
        'Île-de-France': {'lat': 48.8566, 'lon': 2.3522},
        'Normandie': {'lat': 49.1829, 'lon': 0.3707},
        'Nouvelle-Aquitaine': {'lat': 45.7074, 'lon': 0.1532},
        'Occitanie': {'lat': 43.8927, 'lon': 3.2828},
        'Pays de la Loire': {'lat': 47.7633, 'lon': -0.3296},
        'Provence-Alpes-Côte dAzur': {'lat': 43.9352, 'lon': 6.0679}
    }

    # Attribuer aléatoirement une région à chaque ligne
    np.random.seed(42)  # Pour la reproductibilité
    data['Région'] = np.random.choice(regions, size=len(data))

    # Ajouter les colonnes pour la latitude et la longitude
    data['Latitude'] = data['Région'].map(lambda x: region_to_coords[x]['lat'])
    data['Longitude'] = data['Région'].map(lambda x: region_to_coords[x]['lon'])

    # Sauvegarder le dataset enrichi
    enriched_file_path = file_path.replace('.csv', '_enriched.csv')
    data.to_csv(enriched_file_path, index=False)

    print(f"Le dataset enrichi a été sauvegardé sous : {enriched_file_path}")
    return data

# Utilisation du script
file_path = 'data_clean.csv'
enriched_data = enrich_dataset_with_regions(file_path)
print(enriched_data.head())  # Afficher les premières lignes pour vérifier
