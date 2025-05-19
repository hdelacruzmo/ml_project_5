import Definitions
import os.path as osp
import joblib
import geopandas as gpd
import numpy as np

class ModelController:

    def __init__(self):
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        self.model = joblib.load(osp.join(self.model_path, "maxent_model.joblib"))

    def predict_from_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        atributos = gdf.drop(columns="geometry")
        probabilidades = self.model.predict_proba(atributos)[:, 1]
        gdf = gdf.copy()
        gdf["probabilidad"] = np.round(probabilidades, 4)
        return gdf
