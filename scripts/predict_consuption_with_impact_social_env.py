import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mean_absolute_error'])
    return model

# Load data
data = pd.read_csv('dataset/data_clean.csv', delimiter=',')

# One-hot encoding of the 'Region' column
encoder = OneHotEncoder()
region_encoded = encoder.fit_transform(data[['Région']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['Région']))

# Append encoded data back to the original dataframe
data = pd.concat([data, region_encoded_df], axis=1)

# Preparing data
data['Heure'] = pd.to_datetime(data['Heure'], format='%H:%M', errors='coerce').dt.hour
data['mouvement_social'] = data['mouvement_social'].astype(int)  # Assuming mouvement_social is initially a boolean

# Defining numerical columns including movement_social
numerical_cols = [
    'Consommation brute gaz (MW PCS 0°C) - GRTgaz',
    'Consommation brute gaz (MW PCS 0°C) - Teréga',
    'Consommation brute électricité (MW) - RTE',
    'mouvement_social'
] + list(region_encoded_df.columns)

for col in numerical_cols:
    data = remove_outliers(data, col)
    data[col].fillna(data[col].mean(), inplace=True)

# Feature selection and scaling
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
X = data[numerical_cols]
y = data['Consommation brute gaz totale (MW PCS 0°C)']

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# MLflow tracking
with mlflow.start_run(run_name="model_training_with_social_movement",experiment_id=120055952280529084):
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train_scaled, epochs=10, validation_split=0.1)
    test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
    mlflow.log_params({"epochs": 10, "layer1_units": 256, "layer2_units": 128, "layer3_units": 64, "learning_rate": 0.0001})
    mlflow.log_metrics({"test_loss": test_loss, "test_mae": test_mae})
    mlflow.keras.log_model(model, "model")

    # Evaluating the model on test data
    train_predictions_scaled = model.predict(X_train)
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    train_mae = mean_absolute_error(scaler_y.inverse_transform(y_train_scaled), train_predictions)
    mlflow.log_metrics({"train_mae": train_mae})

# Calculate RMSE and MAPE
test_rmse = mean_squared_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)), squared=False)
test_mape = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(model.predict(X_test)))
print(f"Root Mean Squared Error on Test Data: {test_rmse}")
print(f"Mean Absolute Percentage Error on Test Data: {test_mape}")
