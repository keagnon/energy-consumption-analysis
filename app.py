import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# Configure la page pour utiliser la mise en page large
st.set_page_config(layout="wide")

df = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.randint(1, 100, 100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100)
})

df1 = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), columns=list('EFGH'))

# Insérez du CSS personnalisé pour styliser les DataFrames
st.markdown("""
<style>
table {
    border-collapse: collapse;
    border-spacing: 0;
    width: 100%;
    border: 1px solid #ddd;
}
th, td {
    text-align: left;
    padding: 8px;
}
tr:nth-child(even){background-color: #f2f2f2}
th {
    background-color: #f2f2f2;
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.title('Analyse de la Consommation Énergétique')

tab1, tab2, tab3 = st.tabs(["Visualisation", "Statistique", "Modèle"])

# Ajout d'une nouvelle ligne pour les histogrammes
def plot_histogram(data, columns):
    fig, ax = plt.subplots()
    data[columns].hist(bins=20, ax=ax, alpha=0.7, figsize=(10, 5))
    return fig

# Contenu de l'onglet Visualisation
with tab1:
    st.header("Visualisation de la Consommation Énergétique")
    # Première ligne pour les histogrammes
    col1, col2 = st.columns(2)
    with col1:
        # Multiselect pour permettre la sélection de plusieurs colonnes
        selected_columns = st.multiselect('Choisissez une ou plusieurs colonnes', df.columns)

    with col2:
        # Menu déroulant pour sélectionner un type de visualisation
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ('Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot')
        )

    # Générez la visualisation basée sur la sélection de l'utilisateur
    if plot_type == 'Histogramme':
        pass
    elif plot_type == 'Graphique à barres':
        pass
    elif plot_type == 'Lineplot':
        pass
    elif plot_type == 'Boxplot':
        pass
    elif plot_type == 'Scatterplot':
       pass
    elif plot_type == 'Barplot':
        pass
    else:
        st.error("Ce type de visualisation n'est pas pris en charge.")

    # Deuxième ligne pour les histogrammes
    col3, col4 = st.columns(2)
    selected_columns_df1=['A']
    selected_columns_df2=['E']
    with col3:
        if selected_columns_df1:
            st.pyplot(plot_histogram(df1, selected_columns_df1))

    with col4:
        if selected_columns_df2:
            st.pyplot(plot_histogram(df2, selected_columns_df2))

    # Utilisation des expanders pour afficher les DataFrames
    with st.expander("Données brutes"):
        # Afficher le DataFrame df1 avec style
        st.dataframe(df1.style.set_table_styles(
            [{'selector': 'th', 'props': [('font-size', '15pt'), ('text-align', 'center')]}]
        ))

    with st.expander("Données nettoyer"):
        # Afficher le DataFrame df2 avec style
        st.dataframe(df2.style.set_table_styles(
            [{'selector': 'th', 'props': [('font-size', '15pt'), ('text-align', 'center')]}]
        ))


# Contenu de l'onglet Statistique
df_geo = pd.DataFrame({
    'City': ['Paris', 'Madrid', 'Berlin', 'Rome', 'Lisbon'],
    'lat': [48.8566, 40.4168, 52.5200, 41.9028, 38.7223],
    'lon': [2.3522, -3.7038, 13.4050, 12.4964, -9.1393],
    'Value': [4, 2, 5, 1, 3]  # Cette colonne pourrait représenter n'importe quelle donnée que vous voulez montrer
})

# Contenu de l'onglet Statistique
with tab2:
    st.header("Statistique")

    col1, col2 = st.columns(2)

    with col1:
        # Trois selectbox pour différentes sélections
        select1 = st.selectbox('Sélection 1', df_geo['City'], key='1')
        select2 = st.selectbox('Sélection 2', df_geo['City'], key='2')
        select3 = st.selectbox('Sélection 3', df_geo['City'], key='3')

    with col2:
        df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

        st.map(df)

# Contenu de l'onglet Modèle
with tab3:
    st.header("Modèle")
    # Ajoutez ici le contenu de votre onglet modèle
    st.write("Ici, vous pouvez intégrer des modèles prédictifs, afficher les résultats de modélisation, etc.")
