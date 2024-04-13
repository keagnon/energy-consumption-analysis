import numpy as np
import pandas as pd

data = pd.read_csv("stationnement_en_ouvrage_cleaned.csv", sep=",")

# Define a function to generate random geographic coordinates within a plausible range
def generate_random_coordinates(center_lat, center_lon, max_diff=0.05, size=1):
    latitudes = center_lat + np.random.uniform(-max_diff, max_diff, size)
    longitudes = center_lon + np.random.uniform(-max_diff, max_diff, size)
    return list(zip(latitudes, longitudes))

# Helper function to generate random data based on the type
def generate_data_based_on_type(series, num_entries):
    if series.dtype == 'float64' or series.dtype == 'int64':
        # Assuming normal distribution for simplicity, but can be adjusted based on actual data analysis
        return np.random.normal(loc=series.mean(), scale=series.std(), size=num_entries)
    elif series.dtype == 'object':
        # Check if it looks like a coordinate
        if 'coordinates' in series.name or 'geo' in series.name:
            if 'geo_point' in series.name:
                return generate_random_coordinates(48.8566, 2.3522, size=num_entries)
            else:
                # Returning fictional JSON-like strings for geo_shape
                return [f'{{"coordinates": [{np.random.uniform(2.25, 2.40)}, {np.random.uniform(48.8, 48.9)}]}}'
                        for _ in range(num_entries)]
        else:
            # Assuming categorical with a mix of observed categories or as simple as generic categories
            unique_categories = series.dropna().unique()
            if len(unique_categories) < 10:  # Reasonably small number of unique categories
                return np.random.choice(unique_categories, num_entries)
            else:
                return ['Category ' + str(i) for i in np.random.randint(1, 10, num_entries)]
    else:
        return ['Unknown Type' for _ in range(num_entries)]

# Set the number of entries you want in your fictional dataset
num_entries = 100  # Adjust this value as needed

# Apply the function to each column to create a full fictional dataset
fictional_full_data = pd.DataFrame({col: generate_data_based_on_type(data[col], num_entries) for col in data.columns})

# Adjust the format of geographic coordinates to match string format
if 'geo_point_2d' in fictional_full_data.columns:
    fictional_full_data['geo_point_2d'] = fictional_full_data['geo_point_2d'].apply(lambda x: f"{x[0]}, {x[1]}")

# Show a snippet of the fictional full dataset
print(fictional_full_data.head())
fictional_full_data.to_csv("testdonneefictive.csv")