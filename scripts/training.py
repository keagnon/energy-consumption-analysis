import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('dataset/data_clean.csv', delimiter=',')

# Nettoyage et préparation des données
# Conversion des colonnes temporelles en heures (si applicable)
if 'Heure' in data.columns:
    data['Heure'] = pd.to_datetime(data['Heure'], format='%H:%M', errors='coerce').dt.hour

# Suppression des valeurs aberrantes pour chaque colonne numérique
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

numerical_cols = ['Consommation brute gaz (MW PCS 0°C) - GRTgaz', 'Consommation brute gaz (MW PCS 0°C) - Teréga', 'Consommation brute électricité (MW) - RTE']
for col in numerical_cols:
    if data[col].dtype in ['float64', 'int64']:
        data = remove_outliers(data, col)

# Remplacer les valeurs manquantes par la moyenne (pour les colonnes numériques uniquement)
for col in numerical_cols:
    if data[col].dtype in ['float64', 'int64']:
        data[col].fillna(data[col].mean(), inplace=True)

# Sélection et normalisation des caractéristiques
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()

X = data[numerical_cols + ['Heure']]
y = data['Consommation brute gaz totale (MW PCS 0°C)']

X_scaled = scaler_X.fit_transform(X.dropna())  # Assurer que X ne contient pas de NaN
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Construction du modèle
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Augmentation de la taille des couches
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),  # Ajout d'une couche supplémentaire
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mean_absolute_error'])  # Réduction du taux d'apprentissage

# Entraînement du modèle
history = model.fit(X_train, y_train_scaled, epochs=10, validation_split=0.1)  # Augmentation du nombre d'époques

# Évaluation du modèle sur les données de test
test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
print("Perte sur les données de test:", test_loss)
print("Erreur absolue moyenne sur les données de test:", test_mae)

# Calcul de MAE sur les données d'entraînement
train_predictions_scaled = model.predict(X_train)
train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
train_mae = mean_absolute_error(scaler_y.inverse_transform(y_train_scaled), train_predictions)
print("Erreur absolue moyenne sur les données d'entraînement:", train_mae)

# Affichage de la distribution de la variable cible 'Consommation'
# plt.figure(figsize=(10, 5))
# plt.hist(data['Consommation brute gaz totale (MW PCS 0°C)'], bins=30, color='blue', alpha=0.7)
# plt.title('Distribution de la consommation totale de gaz après traitement')
# plt.xlabel('Consommation')
# plt.ylabel('Nombre d\'observations')
# plt.show()
