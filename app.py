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

import mlflow.pyfunc

# Configuration de la page
st.set_page_config(layout="wide")
st.title('Analyse de la Consommation Énergétique')

# Chargement des données
data_path = 'dataset/data_clean.csv'
data_clean = pd.read_csv(data_path,nrows=68641)

data_path2 = 'dataset/Consomation&Mouvement.csv'
data_no_clean = pd.read_csv(data_path2, sep=';', header=0, index_col=False, nrows=68641)

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
                rows = (len(selected_columns) - 1) // 3 + 1  # Nombre de lignes nécessaire
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
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

    # Affichage des données
    with st.expander("Voir les données clean"):
        data_clean_enriched = pd.read_csv('dataset/data_clean.csv')
        st.dataframe(data_clean_enriched)

    with st.expander("Voir les données brutes"):
        st.dataframe(data_no_clean)

# Chargement des données
data_clean_enriched = pd.read_csv('dataset/data_clean.csv')
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
    st.header("Prédiction de consommation énergétique")
    # Chargement du modèle MLflow
    logged_model = 'runs:/93693469433c47788a34ee6f3c37644f/model_v2'
    model = mlflow.pyfunc.load_model(logged_model)

    # Création de l'interface utilisateur

    # Saisie des entrées utilisateur
    heure = st.number_input('Heure', min_value=0, max_value=23, value=12)
    consommation_gaz_grtgaz = st.number_input('Consommation brute gaz (MW PCS 0°C) - GRTgaz', min_value=0.0, format="%.2f")
    consommation_gaz_terega = st.number_input('Consommation brute gaz (MW PCS 0°C) - Teréga', min_value=0.0, format="%.2f")
    consommation_electricite = st.number_input('Consommation brute électricité (MW) - RTE', min_value=0.0, format="%.2f")
    region = st.selectbox('Région', ['Île-de-France', 'Occitanie', 'Nouvelle-Aquitaine', 'Auvergne-Rhône-Alpes'])

    features_supplementaires = np.random.rand(1, 8)

    # Préparation des données pour le modèle
    input_data = {
        'Heure': [heure],
        'Consommation brute gaz (MW PCS 0°C) - GRTgaz': [consommation_gaz_grtgaz],
        'Consommation brute gaz (MW PCS 0°C) - Teréga': [consommation_gaz_terega],
        'Consommation brute électricité (MW) - RTE': [consommation_electricite],
        'Région_Île-de-France': [1 if region == 'Île-de-France' else 0],
        'Région_Occitanie': [1 if region == 'Occitanie' else 0],
        'Région_Nouvelle-Aquitaine': [1 if region == 'Nouvelle-Aquitaine' else 0],
        'Région_Auvergne-Rhône-Alpes': [1 if region == 'Auvergne-Rhône-Alpes' else 0],
    }
    input_df = pd.DataFrame(input_data)
    input_df = pd.concat([input_df, pd.DataFrame(features_supplementaires, columns=['Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15'])], axis=1)

    if st.button('Prédire'):
        predictions = model.predict(input_df)
        predicted_value = predictions.iloc[0, 0]
        st.write(f'Prédiction de la consommation : {predicted_value:.2f}kwh')


