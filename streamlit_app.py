import Definitions
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import Polygon
from src.back.ModelController import ModelController

st.set_page_config(layout="wide", page_title="MaxEnt desde GPKG", page_icon="ðﾟﾧﾠ")
st.title("ðﾟﾧﾠ Visualizador y Predicción desde archivo GPKG")

# -------------------------------
#Carga y visualización básica del GPKG
# -------------------------------

with st.expander("ðﾟﾓﾂ Cargar archivo GPKG"):
    uploaded_gpkg = st.file_uploader("Sube tu archivo .gpkg con geometría y atributos", type=["gpkg"])

    if uploaded_gpkg is not None:
        try:
            gdf_vista = gpd.read_file(uploaded_gpkg)

            if gdf_vista.crs and gdf_vista.crs.to_epsg() != 4326:
                gdf_vista = gdf_vista.to_crs(epsg=4326)

            st.write("Vista previa de los primeros registros:")
            st.dataframe(gdf_vista.drop(columns="geometry").head())

            # Mostrar polígono de cobertura (bounding box)
            bounds = gdf_vista.total_bounds
            centro = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            mapa = folium.Map(location=centro, zoom_start=8, tiles="CartoDB positron")

            folium.GeoJson(gdf_vista, name="Límites").add_to(mapa)
            st_folium(mapa, width=900, height=500)

        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")

# -------------------------------
#  PREDICCIÓN DESDE GPKG
# -------------------------------

st.markdown("---")
st.header("ðﾟﾓﾊ Predicción y visualización del resultado")

ctrl = ModelController()

if uploaded_gpkg is not None:
    try:
        gdf_input = gpd.read_file(uploaded_gpkg)

        gdf_resultado = ctrl.predict_from_gdf(gdf_input)

        st.subheader("ðﾟﾓﾄ Resultados del modelo")
        st.dataframe(gdf_resultado.drop(columns="geometry").head())

        # Visualizar en mapa
        st.subheader("ðﾟﾗﾺ️ Mapa con probabilidades")

        if gdf_resultado.crs and gdf_resultado.crs.to_epsg() != 4326:
            gdf_resultado = gdf_resultado.to_crs(epsg=4326)

        bounds = gdf_resultado.total_bounds
        centro = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        mapa = folium.Map(location=centro, zoom_start=8, tiles="CartoDB positron")

        for _, row in gdf_resultado.iterrows():
            prob = row["probabilidad"]
            color = "red" if prob > 0.5 else "blue"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Prob: {prob:.2f}"
            ).add_to(mapa)

        st_folium(mapa, width=900, height=600)

        st.markdown("### ðﾟﾓﾥ Descargar archivo con resultados")
        output_path = "/tmp/resultados_probabilidad.gpkg"
        gdf_resultado.to_file(output_path, driver="GPKG")
        with open(output_path, "rb") as f:
            st.download_button(
                label="Descargar GPKG con columna de probabilidad",
                data=f,
                file_name="resultados_probabilidad.gpkg",
                mime="application/octet-stream"
            )

    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")

