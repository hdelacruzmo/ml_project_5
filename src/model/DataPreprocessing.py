import numpy as np
import pandas as pd
import os

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.target_feature = "tipo_punto"
        self.feature_path = "resources/models/features_maxent.txt"

    def transform(self, df):
        print("DataPreprocessing.transform ->")
        df = df.copy()

        # Eliminar columnas innecesarias si existen
        columnas_a_desechar = ['fid', 'FID_Mina']
        df = df.drop(columns=[col for col in columnas_a_desechar if col in df.columns])

        # Codificar variables categóricas (sin drop_first para que no falten columnas)
        variables_categoricas = ['TipoCultivo', 'Tipo_Cobertura', 'Tipo_Relieve', 'Tipo_Via']
        df = pd.get_dummies(df, columns=variables_categoricas, drop_first=False)

        # Separar X e y
        X = df.drop(columns=[self.target_feature])
        y = df[self.target_feature]

        # Reindexar columnas para que coincidan con las del entrenamiento
        if os.path.exists(self.feature_path):
            with open(self.feature_path, "r", encoding="utf-8") as f:
                columnas_entrenamiento = f.read().splitlines()
            X = X.reindex(columns=columnas_entrenamiento, fill_value=0)

        return X, y

    def get_categories(self):
        return [0, 1]

    def get_cat_name(self, index):
        if index == 0:
            return "Fondo"
        elif index == 1:
            return "Presencia"
        else:
            return "Desconocido"

    def get_columns(self):
        return {
            'fid', 'tipo_punto', 'DistMinas', 'FID_Mina', 'Minas1000m',
            'TipoCultivo', 'Dist_NoComb', 'Dens_NoComb', 'Num_PrediosURT',
            'Tipo_Cobertura', 'Tipo_Relieve', 'Pendiente', 'Aspecto',
            'Dist_Via', 'Tipo_Via', 'Dist_EventoCombatiente', 'Dens_EventoCombatiente'
        }

