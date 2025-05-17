import numpy as np
import pandas as pd

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.target_feature = "tipo_punto"

    def transform(self, df):
        print("DataPreprocessing.transform ->")

        # Asegurar consistencia de tipos
        df = df.copy()

        # Eliminar columnas innecesarias
        columnas_a_desechar = ['fid', 'FID_Mina']
        for col in columnas_a_desechar:
            if col in df.columns:
                df = df.drop(columns=col)

        # Codificar variables categóricas con one-hot encoding
        variables_categoricas = ['TipoCultivo', 'Tipo_Cobertura', 'Tipo_Relieve', 'Tipo_Via']
        df = pd.get_dummies(df, columns=variables_categoricas, drop_first=True)

        X, y = df.drop(self.target_feature, axis=1), df[self.target_feature]
        return X, y

    def get_categories(self):
        return [0, 1]  # tipo_punto: 0 = fondo, 1 = sitio confirmado

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
