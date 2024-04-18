# ENERGY CONSUMPTION ANALYSIS

## Description

Energy Consumption Analysis est une application Streamlit conçue pour visualiser et analyser les données de consommation énergétique. Elle permet aux utilisateurs d'explorer des séries temporelles de consommation, d'effectuer des analyses statistiques et d'interagir avec des modèles prédictifs liés à la consommation d'énergie.

## Fonctionnalités

- **Visualisation :** Des graphiques interactifs qui permettent aux utilisateurs d'examiner les tendances de la consommation énergétique.
- **Statistique :** Une analyse descriptive et exploratoire des données.
- **Modélisation :** Un onglet où les modèles prédictifs peuvent être entraînés et évalués.

## Commencer

Pour exécuter cette application en local, suivez les étapes ci-dessous :

### Prérequis

- Python 3.10+
- pip

### Installation

1. Clonez le dépôt :
   ```bash
   git clone git@github.com:keagnon/energy-consumption-analysis.git
   ```
2. Accédez au répertoire du projet :
   ```bash
   cd energy-consumption-analysis
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Exécutez l'application :
   ```bash
   streamlit run app.py
   ```

## Déploiement

Pour déployer cette application dans un conteneur Docker, veuillez suivre les instructions contenues dans le `Dockerfile`.

## Structure du Projet

```
energy-consumption-analysis/
│
├── dataset/                # Contient les datasets utilisés.
├── notebook/               # Jupyter notebooks pour l'exploration de données.
├── app.py                  # Point d'entrée de l'application Streamlit.
├── Dockerfile              # Définitions pour la construction de l'image Docker.
├── .gitignore              # Spécifie les fichiers intentionnellement non suivis à ignorer.
├── requirements.txt        # Les dépendances nécessaires pour l'application.
└── README.md               # Documentation du projet.
```


## Licence

Ce projet est distribué sous la Licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## Contact

- **Créateur :** [@keagnon](https://github.com/keagnon)
- **Projet GitHub :** [Energy Consumption Analysis](https://github.com/keagnon/energy-consumption-analysis)
