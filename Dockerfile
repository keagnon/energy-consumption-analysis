FROM python:3.9

# Créez le répertoire de travail
WORKDIR /app

# Copiez les fichiers nécessaires dans le conteneur
COPY requirements.txt .
COPY app.py .

# Installez les dépendances
RUN conda install --file requirements.txt -c conda-forge

# Exposez le port utilisé par votre application Streamlit
EXPOSE 8501

# Commande pour démarrer l'application
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
