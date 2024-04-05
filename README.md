# sEnergy-consumption-analysis
Analyse de la consommation d'énergie avec Scala, Spark et BigQuery

chardet ---- utiliser pour resoudre le probleme https://stackoverflow.com/questions/74535380/importerror-cannot-import-name-common-safe-ascii-characters-from-charset-nor


def visualize_gas_consumption_over_time(df):
        """
        Trace la consommation brute de gaz au fil du temps.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y='Consommation brute gaz totale (MW PCS 0°C)', data=df)
        plt.title('Consommation brute de gaz au fil du temps')
        plt.xticks(rotation=45)
        plt.show()

    def visualize_gas_consumption_distribution(df):
        """
        Affiche la distribution de la consommation brute de gaz.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Consommation brute gaz totale (MW PCS 0°C)'], kde=True)
        plt.title('Distribution de la consommation brute de gaz')
        plt.show()

    def visualize_gas_consumption_by_hour(df):
        """
        Visualise la consommation brute de gaz par heure.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Heure', y='Consommation brute gaz totale (MW PCS 0°C)', data=df)
        plt.title('Consommation brute de gaz par heure')
        plt.xticks(rotation=45)
        plt.show()

    def visualize_gas_electricity_consumption_relationship(df):
        """
        Montre la relation entre la consommation brute de gaz et d'électricité.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Consommation brute gaz totale (MW PCS 0°C)', y='Consommation brute électricité (MW) - RTE', data=df)
        plt.title("Relation entre la consommation brute de gaz et d'électricité")
        plt.show()

    def visualize_gas_consumption_by_gaz_status(df):
        """
        Visualise la consommation brute de gaz totale par statut de GRTgaz.
        """
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Statut - GRTgaz', y='Consommation brute gaz totale (MW PCS 0°C)', data=df)
        plt.title('Consommation brute de gaz totale par statut de GRTgaz')
        plt.show()

    def visualize_total_consumption_by_month(df):
        """
        Visualise la consommation brute totale par mois.
        """
        df['Mois'] = df['Date'].dt.month
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Mois', y='Consommation brute totale (MW)', data=df, estimator=sum)
        plt.title('Consommation brute totale par mois')
        plt.show()

    def visualize_gas_consumption_by_weekday(df):
        """
        Visualise la consommation brute de gaz par jour de la semaine.
        """
        df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Day_of_Week', y='Consommation brute gaz (MW PCS 0°C) - GRTgaz', data=df, palette='Set3')
        plt.xticks(ticks=range(7), labels=['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
        plt.title('Consommation brute de gaz par jour de la semaine')
        plt.show()
