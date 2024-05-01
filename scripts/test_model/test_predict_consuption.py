import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Hypothetical full feature names from the training
features = [
    'Heure', 'Consommation brute gaz (MW PCS 0°C) - GRTgaz',
    'Consommation brute gaz (MW PCS 0°C) - Teréga', 'Consommation brute électricité (MW) - RTE',
    'Région_Île-de-France', 'Région_Occitanie', 'Région_Nouvelle-Aquitaine',
    'Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature12',
    'Feature13', 'Feature14', 'Feature15', 'Feature16'
]

# Example to create a DataFrame with the correct number of dummy features
data = {
    'Heure': [12, 14, 9, 22, 18],
    'Consommation brute gaz (MW PCS 0°C) - GRTgaz': [120.5, 150.3, 170.2, 145.5, 130.0],
    'Consommation brute gaz (MW PCS 0°C) - Teréga': [60.2, 75.1, 65.0, 70.3, 80.4],
    'Consommation brute électricité (MW) - RTE': [300.5, 280.0, 310.2, 290.3, 305.4],
    'Région_Île-de-France': [1.0, 0.0, 1.0, 0.0, 0.0],
    'Région_Occitanie': [0.0, 1.0, 0.0, 0.0, 0.0],
    'Région_Nouvelle-Aquitaine': [0.0, 0.0, 0.0, 0.0, 1.0]
}

# Adding dummy data for missing features (Assuming numerical data for simplicity)
for feature in features[7:]:
    data[feature] = np.random.rand(5)

df = pd.DataFrame(data, columns=features)

# Load the model and predict
logged_model = 'runs:/93693469433c47788a34ee6f3c37644f/model_v2'
loaded_model = mlflow.pyfunc.load_model(logged_model)
predictions = loaded_model.predict(df)
print("Predictions:", predictions)
