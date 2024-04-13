import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns

# Configuration de la page
st.set_page_config(layout="wide")
st.title('Analyse de la Consommation Énergétique')

# Chargement des données
data_path = 'data_clean.csv'
data_clean = pd.read_csv(data_path,nrows =500)

data_path2 = 'data_no_clean.csv'
data_no_clean = pd.read_csv(data_path2, sep=';', header=0, index_col=False,nrows =500)

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
        selected_columns = st.multiselect('Choisissez une ou plusieurs colonnes', data_clean.columns)

    with col2:
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ['Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot']
        )


    col3, col4 = st.columns(2)

    with col3:
        if selected_columns:
            if plot_type == 'Histogramme':
                numeric_columns = data_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0:
                    rows = len(selected_columns) // 2 + (1 if len(selected_columns) % 2 > 0 else 0)
                    cols = 2  # Nous voulons 2 colonnes par ligne

                    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_columns)
                    for i, col in enumerate(selected_columns, start=1):
                        row = (i-1) // 2 + 1  # Calculer la ligne actuelle
                        col_pos = (i-1) % 2 + 1  # Calculer la position de la colonne dans la ligne
                        fig.add_trace(go.Histogram(x=data_clean[col]), row=row, col=col_pos)

                    fig.update_layout(height=400*rows, showlegend=False)
                    st.plotly_chart(fig)
                else:
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

            elif plot_type == 'Graphique à barres':
                fig = px.bar(data_clean, x=selected_columns[0], y=selected_columns[1:])
                st.plotly_chart(fig)
            elif plot_type == 'Lineplot':
                fig = px.line(data_clean, x=selected_columns[0], y=selected_columns[1:])
                st.plotly_chart(fig)
            elif plot_type == 'Boxplot':
                # Filtrer pour garder uniquement les colonnes numériques
                numeric_columns = data_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0:
                    fig = px.box(data_clean, y=numeric_columns)
                    st.plotly_chart(fig)
                else:
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

    with col4:
        if selected_columns:
            if plot_type == 'Histogramme':
                numeric_columns = data_no_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0:
                   # Déterminer le nombre de lignes pour les subplots
                    rows = len(selected_columns) // 2 + (1 if len(selected_columns) % 2 > 0 else 0)
                    cols = 2  # Nous voulons 2 colonnes par ligne

                    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_columns)
                    for i, col in enumerate(selected_columns, start=1):
                        row = (i-1) // 2 + 1  # Calculer la ligne actuelle
                        col_pos = (i-1) % 2 + 1  # Calculer la position de la colonne dans la ligne
                        fig.add_trace(go.Histogram(x=data_no_clean[col]), row=row, col=col_pos)

                    fig.update_layout(height=400*rows, showlegend=False)
                    st.plotly_chart(fig)
                else:
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

            elif plot_type == 'Graphique à barres':
                fig = px.bar(data_no_clean, x=selected_columns[0], y=selected_columns[1:])
                st.plotly_chart(fig)
            elif plot_type == 'Lineplot':
                fig = px.line(data_no_clean, x=selected_columns[0], y=selected_columns[1:])
                st.plotly_chart(fig)
            elif plot_type == 'Boxplot':
                # Filtrer pour garder uniquement les colonnes numériques
                numeric_columns = data_no_clean[selected_columns].select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0:
                    fig = px.box(data_no_clean, y=numeric_columns)
                    st.plotly_chart(fig)
                else:
                    st.error("Veuillez sélectionner des colonnes numériques pour ce type de visualisation.")

        # Affichage des données
    with st.expander("Voir les données"):
        st.dataframe(data_clean)
        # Affichage des données
    with st.expander("Voir les données"):
        st.dataframe(data_no_clean)
# Contenu de l'onglet Statistique
with tab2:
    st.header("Statistiques descriptives et Matrice de Corrélation")
    col1, col2 = st.columns(2)

    # Exclure les colonnes non numériques pour la matrice de corrélation
    numerical_columns = data_clean.select_dtypes(include=[np.number]).columns
    corr_data = data_clean[numerical_columns]

    # Matrice de corrélation
    st.write("Matrice de corrélation :")
    corr_matrix = corr_data.corr()
    st.write(corr_matrix)

    # Statistiques descriptives
    st.write("Statistiques descriptives :")
    st.write(data_clean.describe())


# Contenu de l'onglet Modèle
with tab3:
    st.header("Modèle")
    st.write("Ici, vous pouvez intégrer des modèles prédictifs, afficher les résultats de modélisation, etc.")


