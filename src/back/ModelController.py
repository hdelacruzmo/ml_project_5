import Definitions
import numpy as np
import os.path as osp
import pandas as pd
from io import StringIO
import joblib

class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        self.model_path_file = osp.join(self.model_path, "maxent_model.joblib")

        self.model = joblib.load(self.model_path_file)
        self.input_df = ""

    def validate_data(self, df):
        return "tipo_punto" in df.columns

    def load_input_data(self, input_data):
        try:
            input_data_str = StringIO(input_data.decode("utf-8"))
            self.input_df = pd.read_csv(input_data_str)
            is_valid = self.validate_data(self.input_df)
            return self.input_df, is_valid
        except Exception as e:
            raise Exception(f"Error al leer el CSV: {e}")

    def predict(self):
        X = self.input_df.drop(columns=["tipo_punto"])
        y = self.input_df["tipo_punto"]

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        df_resultado = X.copy()
        df_resultado["Real"] = y.values
        df_resultado["Predicción"] = y_pred
        df_resultado["Probabilidad"] = np.round(y_proba, 4)

        return df_resultado
