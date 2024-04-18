# Utiliser l'image Miniconda3 comme base
FROM continuumio/miniconda3:latest

# Créer le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier environment.yml dans le répertoire de travail
COPY environment.yml .

# Installer les dépendances depuis le fichier environment.yml
RUN conda env create -f environment.yml

# Activer l'environnement Conda spécifié
SHELL ["conda", "run", "-n", "nom_de_votre_environnement", "/bin/bash", "-c"]

# Copier le script de l'application Streamlit dans le répertoire de travail
COPY app.py .

# Exposer le port 8501 pour l'application Streamlit
EXPOSE 8501

# Définir la commande pour démarrer l'application Streamlit
CMD ["conda", "run", "-n", "nom_de_votre_environnement", "streamlit", "run", "--server.port", "8501", "app.py"]
