import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.keras
import mlflow.sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch

# Fonction pour supprimer les valeurs aberrantes
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Fonction pour construire le modèle avec Keras Tuner
def build_model_tuned(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=256, step=32),
                    activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_hidden', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mse', metrics=['mean_absolute_error'])
    return model

# Charger les données
data = pd.read_csv('dataset/data_clean.csv', delimiter=',')

# One-hot encoding de la colonne 'Région'
encoder = OneHotEncoder()
region_encoded = encoder.fit_transform(data[['Région']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['Région']))

# Préparation des données
data = pd.concat([data, region_encoded_df], axis=1)
data['Heure'] = pd.to_datetime(data['Heure'], format='%H:%M', errors='coerce').dt.hour
data['mouvement_social'] = data['mouvement_social'].astype(int)

# Colonnes numériques y compris mouvement social
numerical_cols = [
    'Consommation brute gaz (MW PCS 0°C) - GRTgaz',
    'Consommation brute gaz (MW PCS 0°C) - Teréga',
    'Consommation brute électricité (MW) - RTE',
    'mouvement_social'
] + list(region_encoded_df.columns)

for col in numerical_cols:
    data = remove_outliers(data, col)
    data[col].fillna(data[col].mean(), inplace=True)

# Sélection et normalisation des caractéristiques
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
X = data[numerical_cols]
y = data['Consommation brute gaz totale (MW PCS 0°C)']

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Lancement de l'expérience avec Keras Tuner
tuner = RandomSearch(
    build_model_tuned,
    objective='val_mean_absolute_error',
    max_trials=10,
    executions_per_trial=2,
    directory='model_tuning',
    project_name='EnergyConsumption')

tuner.search(X_train, y_train_scaled, epochs=10, validation_split=0.1)
best_model = tuner.get_best_models(num_models=1)[0]

# Enregistrement avec MLflow
with mlflow.start_run(run_name="keras_model_tuning"):
    mlflow.log_params(tuner.get_best_hyperparameters(num_models=1)[0].values)
    mlflow.keras.log_model(best_model, "best_keras_model")

# Comparaison avec un modèle de régression linéaire
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train_scaled)
lin_reg_predictions = lin_reg_model.predict(X_test)
lin_reg_rmse = mean_squared_error(y_test_scaled, lin_reg_predictions, squared=False)
lin_reg_mape = mean_absolute_percentage_error(y_test_scaled, lin_reg_predictions)

# Enregistrement des résultats de la régression linéaire
with mlflow.start_run(run_name="linear_regression_model"):
    mlflow.log_metric("rmse", lin_reg_rmse)
    mlflow.log_metric("mape", lin_reg_mape)
    mlflow.sklearn.log_model(lin_reg_model, "linear_regression_model")

print(f"Linear Regression RMSE: {lin_reg_rmse}")
print(f"Linear Regression MAPE: {lin_reg_mape}")
