import Definitions
import numpy as np
import os.path as osp
import pandas as pd
from io import StringIO
import joblib
from src.model.DataPreprocessing import DataPreprocessing

class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")

        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        # self.rf_model_path = osp.join(self.model_path, "rf_model.joblib")
        self.maxent_model_path = osp.join(self.model_path, "maxent_model.joblib")

        # self.rf_model = joblib.load(self.rf_model_path)
        self.maxent_model = joblib.load(self.maxent_model_path)

        self.input_df = ""
        self.d_processing = DataPreprocessing()

    def validate_data(self, df):
        return self.d_processing.get_columns().issubset(df.columns)

    def load_input_data(self, input_data):
        try:
            input_data_str = StringIO(input_data.decode("utf-8"))
            self.input_df = pd.read_csv(input_data_str)
            is_valid = self.validate_data(self.input_df)
            return self.input_df, is_valid
        except Exception as e:
            raise Exception(f"Error al leer el CSV: {e}")

    def predict(self):
        X_test, Y_test = self.d_processing.transform(self.input_df)

        # --- RF DESACTIVADO ---
        # y_pred_rf = self.rf_model.predict(X_test)
        # y_pred_proba_rf = self.rf_model.predict_proba(X_test)
        # y_pred_proba_rf = np.round(y_pred_proba_rf * 100, 2)

        # rf_df = X_test.copy()
        # rf_df["Real"] = Y_test.values
        # rf_df["Predicción"] = y_pred_rf
        # rf_df[f"Prob. Clase 0 RF (%)"] = y_pred_proba_rf[:, 0]
        # rf_df[f"Prob. Clase 1 RF (%)"] = y_pred_proba_rf[:, 1]

        # --- MaxEnt ---
        y_pred_maxent = self.maxent_model.predict(X_test)
        y_pred_proba_maxent = self.maxent_model.predict_proba(X_test)
        y_pred_proba_maxent = np.round(y_pred_proba_maxent * 100, 2)

        maxent_df = X_test.copy()
        maxent_df["Real"] = Y_test.values
        maxent_df["Predicción"] = y_pred_maxent
        maxent_df[f"Prob. Clase 0 MaxEnt (%)"] = y_pred_proba_maxent[:, 0]
        maxent_df[f"Prob. Clase 1 MaxEnt (%)"] = y_pred_proba_maxent[:, 1]

        # Full view (solo MaxEnt)
        full_df = X_test.copy()
        full_df["Real"] = Y_test.values
        full_df["Pred. MaxEnt"] = y_pred_maxent
        full_df["Prob. MaxEnt (1)"] = y_pred_proba_maxent[:, 1]

        return maxent_df, full_df

