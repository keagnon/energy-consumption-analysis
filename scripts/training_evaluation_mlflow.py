import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
import mlflow
from mlflow import MlflowClient
from mlflow_utils import create_mlflow_experiment,get_mlflow_experiment


if __name__=="__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name="Hyperparameter_Tuning",
        artifact_location="deep_learning_mlflow_artifact",
        tags={"env": "dev", "version": "1.0.0"},
    )


client = MlflowClient()
experiment = get_mlflow_experiment(experiment_name="Hyperparameter_Tuning")

# print(f"Experiment ID: {experiment_id}")

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
def build_model(optimizer='adam', activation='relu', neurons=128):
    model = Sequential([
        Dense(neurons, activation=activation, input_shape=(X_train.shape[1],)),
        Dense(neurons, activation=activation),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
    return model

# Définir la grille d'hyperparamètres à rechercher
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [64, 128, 256]
}

# Initialiser RandomizedSearchCV
random_search = RandomizedSearchCV(build_model(), param_distributions=param_grid, cv=3, n_iter=10, scoring='neg_mean_absolute_error', verbose=2)

# Suivre l'exécution de l'entraînement avec MLflow
with mlflow.start_run(experiment_id=experiment_id):
    # Effectuer la recherche aléatoire des hyperparamètres
    random_search.fit(X_train, y_train_scaled)

    # Enregistrer les meilleurs hyperparamètres et les métriques
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metrics({"best_mean_absolute_error": -random_search.best_score_})

    # Évaluation du modèle sur les données de test avec les meilleurs hyperparamètres
    test_loss, test_mae = random_search.best_estimator_.evaluate(X_test, y_test_scaled)
    mlflow.log_metrics({"test_loss": test_loss, "test_mean_absolute_error": test_mae})


