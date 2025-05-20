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

def predict_with_second_model(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    modelo_alt = joblib.load(osp.join(self.model_path, "segundo_modelo.joblib"))
    atributos = gdf.drop(columns="geometry")
    gdf = gdf.copy()
    gdf["probabilidad_2"] = np.round(modelo_alt.predict_proba(atributos)[:, 1], 4)
    return gdf

