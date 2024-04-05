import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
with tab2:
    st.header("Statistique")
    # Ajoutez ici le contenu de votre onglet statistique
    st.write("Ici, vous pouvez afficher des statistiques descriptives, des tableaux de données, etc.")


    with col1:
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ('Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot')
        )
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ('Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot')
        )
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ('Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot')
        )
        plot_type = st.selectbox(
            'Choisissez un type de visualisation',
            ('Histogramme', 'Graphique à barres', 'Lineplot', 'Boxplot')
        )

    with col2:
        """Création d'un DataFrame associatif pour les échanges """
        echanges_geo = pd.DataFrame({
            'Colonne': ['ech_comm_espagne', 'ech_comm_allemagne', 'ech_comm_italie'],  # Ajoutez plus selon vos colonnes
            'Pays': ['Espagne', 'Allemagne', 'Italie'],
            'Latitude': [40.463667, 51.165691, 41.87194],
            'Longitude': [-3.74922, 10.451526, 12.56738]
        })

        # Exemple d'utilisation : afficher ce DataFrame
        print(echanges_geo)

        fig = go.Figure(data=go.Scattergeo(
            lon = echanges_geo['Longitude'],
            lat = echanges_geo['Latitude'],
            text = echanges_geo['Pays'],
            mode = 'markers',
            marker = dict(size = 8, color = 'red'),
        ))

        fig.update_layout(title = 'Échanges d\'énergie', geo_scope='europe')
        fig.show()



# Contenu de l'onglet Modèle
with tab3:
    st.header("Modèle")
    # Ajoutez ici le contenu de votre onglet modèle
    st.write("Ici, vous pouvez intégrer des modèles prédictifs, afficher les résultats de modélisation, etc.")
