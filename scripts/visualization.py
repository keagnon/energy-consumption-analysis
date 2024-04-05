import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class visualization:
    def __init__(self, df):
        self.df = df

    def visualize_distribution(self):
        """
        Visualise la distribution des valeurs pour chaque colonne numérique du DataFrame.
        """
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.df[column].dropna(), kde=True)
                plt.title(f'Distribution de {column}')
                plt.show()

