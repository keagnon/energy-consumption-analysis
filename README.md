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

## Déploiement avec Docker

Si vous avez déjà Docker installé et fonctionnel sur votre machine, vous pouvez déployer cette application en construisant et en exécutant un conteneur Docker. Suivez ces instructions :

1. Ouvrez un terminal et naviguez vers le répertoire racine du projet où se trouve le `Dockerfile`.

2. Construisez l'image Docker à l'aide de la commande suivante :
   ```bash
   docker build -t energy-consumption-analysis .
   ```
   Cette commande construit une image Docker nommée `energy-consumption-analysis` en utilisant le `Dockerfile` situé dans le répertoire courant (représenté par le `.`).

3. Une fois l'image construite, lancez un conteneur en utilisant :
   ```bash
   docker run -p 8501:8501 energy-consumption-analysis
   ```
   Cette commande exécute le conteneur Docker et mappe le port 8501 de votre machine au port 8501 du conteneur (ce qui est le port par défaut utilisé par Streamlit). Vous pouvez alors accéder à l'application Streamlit via votre navigateur en allant à l'adresse `http://localhost:8501`.

Assurez-vous de consulter le `Dockerfile` pour comprendre la configuration du conteneur et pour faire d'autres personnalisations si nécessaire.

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
