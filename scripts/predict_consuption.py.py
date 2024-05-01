import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.keras

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def build_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mean_absolute_error'])
    return model

data = pd.read_csv('dataset/data_clean.csv', delimiter=',')

# Encodage One-Hot de la colonne 'Région'
encoder = OneHotEncoder()
region_encoded = encoder.fit_transform(data[['Région']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['Région']))

data = pd.concat([data, region_encoded_df], axis=1)

# Nettoyage et préparation des données
data['Heure'] = pd.to_datetime(data['Heure'], format='%H:%M', errors='coerce').dt.hour

numerical_cols = ['Consommation brute gaz (MW PCS 0°C) - GRTgaz', 'Consommation brute gaz (MW PCS 0°C) - Teréga', 'Consommation brute électricité (MW) - RTE'] + list(region_encoded_df.columns)

for col in numerical_cols:
    if col in data.columns:
        data = remove_outliers(data, col)
        data[col].fillna(data[col].mean(), inplace=True)

# Sélection et normalisation des caractéristiques
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
X = data[numerical_cols]
y = data['Consommation brute gaz totale (MW PCS 0°C)']

X_scaled = scaler_X.fit_transform(X.dropna())
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Utilisation de MLflow
# mlflow.set_experiment('energy_consumption_prediction')

with mlflow.start_run(run_name="log_keras_model_with_social_mvt_hyperparametre_turning", experiment_id=120055952280529084):
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train_scaled, epochs=10, validation_split=0.1)
    mlflow.log_params({"epochs": 50, "layer1_units": 256, "layer2_units": 128, "layer3_units": 64, "learning_rate": 0.0001})
    test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
    mlflow.log_metrics({"test_loss": test_loss, "test_mae": test_mae})

    # Enregistrer le modèle
    mlflow.keras.log_model(model, "model")


    # Évaluation du modèle sur les données de test et calcul de MAE sur les données d'entraînement
    train_predictions_scaled = model.predict(X_train)
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    train_mae = mean_absolute_error(scaler_y.inverse_transform(y_train_scaled), train_predictions)
    mlflow.log_metrics({"train_mae": train_mae})


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Calcul du RMSE
test_rmse = mean_squared_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)), squared=False)
print("Root Mean Squared Error on Test Data:", test_rmse)

# Calcul du MAPE
test_mape = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)))
print("Mean Absolute Percentage Error on Test Data:", test_mape)



from tensorflow.keras.layers import Dropout

def build_model_v2(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),  # Ajout une couche Dropout
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mean_absolute_error'])
    return model

# Utilisation de MLflow
with mlflow.start_run(run_name="logging_models_keras_v2", experiment_id=120055952280529084):
    model = build_model_v2(X_train.shape[1])
    history = model.fit(X_train, y_train_scaled, epochs=10, validation_split=0.1)
    mlflow.log_params({"epochs": 10, "layer1_units": 256, "layer2_units": 128, "layer3_units": 64, "learning_rate": 0.0001})
    test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
    mlflow.log_metrics({"test_loss": test_loss, "test_mae": test_mae})

    # Enregistrer le modèle
    mlflow.keras.log_model(model, "model_v2")

    # Enregistrer le meilleur modèle avec MLflow
    mlflow.keras.save_model(model, "best_model")

    # Évaluation du modèle sur les données de test et calcul de MAE sur les données d'entraînement
    train_predictions_scaled = model.predict(X_train)
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    train_mae = mean_absolute_error(scaler_y.inverse_transform(y_train_scaled), train_predictions)
    mlflow.log_metrics({"train_mae": train_mae})

# Affichage des résultats
test_rmse = mean_squared_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)), squared=False)
print("Root Mean Squared Error on Test Data:", test_rmse)
test_mape = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)))
print("Mean Absolute Percentage Error on Test Data:", test_mape)
