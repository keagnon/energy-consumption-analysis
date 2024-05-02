#!/bin/bash
# Démarrage du serveur MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
# Démarrage de l'application Streamlit
streamlit run app.py
