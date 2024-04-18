import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration de la page
st.set_page_config(layout="wide")
st.title('Analyse de la Consommation Énergétique')

# Chargement des données
data_path = 'dataset/data_clean.csv'
data_clean = pd.read_csv(data_path,nrows=25000)

data_path2 = 'dataset/Consomation&Mouvement.csv'
data_no_clean = pd.read_csv(data_path2, sep=';', header=0, index_col=False, nrows=25000)

# Tabs
tab1, tab2, tab3 = st.tabs(["Visualisation", "Statistique", "Modèle"])

# Fonctions de visualisation
def plot_histogram(data, columns):
    fig, ax = plt.subplots()
    data[columns].hist(bins=20, ax=ax, alpha=0.7, figsize=(10, 5))
    plt.title('Histogramme de la Consommation')
    return fig

# Contenu de l'onglet Visualisation
with tab1:
    st.header("Visualisation de la Consommation Énergétique")
    col1, col2 = st.columns(2)
    with col1:

        numeric_columns = data_clean[data_clean.columns].select_dtypes(include=['float64', 'int64']).columns
        selected_columns = st.multiselect('Choisissez une ou plusieurs colonnes', numeric_columns)

    with col2:
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ['Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot']
        )

    if selected_columns:
        if plot_type == 'Histogramme':
            numeric_columns = data_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) > 0:
                # Calcul du nombre de lignes et de colonnes en fonction du nombre de colonnes sélectionnées
                rows = (len(selected_columns) - 1) // 3 + 1
                cols = min(3, len(selected_columns))  # Au plus trois colonnes par ligne

                # Création de la figure
                fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_columns)

                # Ajout des histogrammes aux positions appropriées dans la figure
                for i, col in enumerate(selected_columns, start=1):
                    row = (i - 1) // 3 + 1
                    col_pos = (i - 1) % 3 + 1
                    fig.add_trace(go.Histogram(x=data_clean[col]), row=row, col=col_pos)

                # Remplissage des graphiques vides avec des traces vides
                for i in range(len(selected_columns) + 1, rows * cols + 1):
                    row = (i - 1) // 3 + 1
                    col_pos = (i - 1) % 3 + 1
                    fig.add_trace(go.Scatter(), row=row, col=col_pos)

                # Mise à jour de la mise en page pour ajuster la taille de la figure
                fig.update_layout(width=1600, height=400 * rows)

                # Afficher la figure
                st.plotly_chart(fig)
            else:
                st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

        elif plot_type == 'Graphique à barres':
                fig = px.bar(data_clean, x=selected_columns[0], y=selected_columns[1:])
                fig.update_layout(width=1600)
                st.plotly_chart(fig)
        elif plot_type == 'Lineplot':
                fig = px.line(data_clean, x=selected_columns[0], y=selected_columns[1:])
                fig.update_layout(width=1600)
                st.plotly_chart(fig)
        elif plot_type == 'Boxplot':

                numeric_columns = data_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0:
                    fig = px.box(data_clean, y=numeric_columns)
                    fig.update_layout(width=1600)
                    st.plotly_chart(fig)
                else:
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation")


    data_clean_enriched=data_clean
    data_clean_enriched['Date'] = pd.to_datetime(data_clean_enriched['Date'])

    # Obtention de la liste des régions uniques pour le selectbox
    regions = data_clean_enriched['Région'].unique()
    selected_region = st.selectbox('Choisissez une région', regions)

    # Filtre pour la région spécifique choisie
    region_data = data_clean_enriched[data_clean_enriched['Région'] == selected_region]

    # Création de graphique de série temporelle
    def plot_time_series(region_data):
        plt.figure(figsize=(15, 7))
        plt.plot(region_data['Date'], region_data['Consommation brute totale (MW)'], label='Consommation de gaz')

        # Ajout des marqueurs pour les événements
        if 'mouvement_social' in region_data.columns:
            for date in region_data[region_data['mouvement_social']]['Date']:
                plt.axvline(x=date, color='red', linestyle='--', lw=2)

        # Format de la date sur l'axe x
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        # Légende et étiquettes
        plt.legend()
        plt.title(f'Consommation de gaz pour la région {selected_region}')
        plt.xlabel('Date')
        plt.ylabel('Consommation de gaz (MW)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Affichage de la figure
        st.pyplot(plt)


    if not region_data.empty:
        plot_time_series(region_data)
    else:
        st.write(f"Aucune donnée disponible pour la région sélectionnée: {selected_region}")

    # Affichage des données
    with st.expander("Voir les données clean"):
        data_clean_enriched = pd.read_csv('dataset/data_clean.csv')
        st.dataframe(data_clean_enriched)

    with st.expander("Voir les données brutes"):
        st.dataframe(data_no_clean)

# Chargement des données
data_clean_enriched = pd.read_csv('dataset/data_clean.csv',)
data_clean_enriched['Date'] = pd.to_datetime(data_clean_enriched['Date'])

with tab2:

    # Aggrégation des données par région pour la carte
    aggregated_data = data_clean_enriched.groupby(['Région', 'Latitude', 'Longitude'])\
                                        .agg({'Consommation brute totale (MW)': 'sum'})\
                                        .reset_index()

    # Fonction pour créer une carte avec Folium
    def create_folium_map(data):
        m = folium.Map(location=[46.2276, 2.2137], zoom_start=6)
        for idx, row in data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"{row['Région']}: {row['Consommation brute totale (MW)']} MW",
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)
        return m

    # Création et affichage de la carte
    st.header("Carte de consommation par région")
    folium_map = create_folium_map(aggregated_data)
    st_folium(folium_map, width=2725, height=500)

    # Statistiques descriptives
    st.header("Statistiques descriptives")
    numerical_columns = data_clean_enriched.select_dtypes(include=[np.number]).columns
    st.write(data_clean_enriched[numerical_columns].describe())

    # Matrice de corrélation
    st.header("Matrice de corrélation")
    corr_matrix = data_clean_enriched[numerical_columns].corr()
    st.write(corr_matrix)

# Contenu de l'onglet Modèle
with tab3:
    st.header("Modèle")
    st.write("Ici, vous pouvez intégrer des modèles prédictifs, afficher les résultats de modélisation.")


