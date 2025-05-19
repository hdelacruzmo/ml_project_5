import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from src.back.ModelController import ModelController

st.set_page_config(layout="wide", page_title="MaxEnt - Visualizador", page_icon="ðﾟﾧﾠ")
st.title("ðﾟﾧﾠ Visualizador y Predicción con Modelo MaxEnt")

# -------------------------------
# ðﾟﾗﾺ️ VISUALIZACIÓN DEL GPKG
# -------------------------------

with st.expander("ðﾟﾗﾺ️ Ver área geográfica cubierta (.gpkg)"):
    uploaded_gpkg = st.file_uploader(
        "ðﾟﾓﾂ Sube tu archivo GPKG", accept_multiple_files=False, type=["gpkg"], key="gpkg"
    )

    if uploaded_gpkg is not None:
        try:
            gdf = gpd.read_file(uploaded_gpkg)

            if gdf.crs and gdf.crs.to_epsg() != 4326:
                st.info(f"ðﾟﾓﾐ Reproyectando desde {gdf.crs} a EPSG:4326 para visualización.")
                gdf = gdf.to_crs(epsg=4326)

            st.write("Vista previa del archivo:")
            st.dataframe(gdf.head())

            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            bounds = gdf.total_bounds
            minx, miny, maxx, maxy = bounds

            rectangle = Polygon([
                (minx, miny), (minx, maxy),
                (maxx, maxy), (maxx, miny),
                (minx, miny)
            ])
            gdf_rect = gpd.GeoDataFrame(geometry=[rectangle], crs=gdf.crs)

            mapa = folium.Map(location=[(miny + maxy)/2, (minx + maxx)/2], zoom_start=8, tiles=None)

            folium.TileLayer("OpenStreetMap", name="Mapa Base").add_to(mapa)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Satélite (Esri)",
                overlay=False,
                control=True
            ).add_to(mapa)
            folium.TileLayer(
                tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Nombres del Territorio",
                overlay=True,
                control=True
            ).add_to(mapa)

            folium.GeoJson(gdf_rect, name="Área de Estudio", tooltip="Área cubierta").add_to(mapa)
            mapa.fit_bounds([[miny, minx], [maxy, maxx]])
            folium.LayerControl(collapsed=False).add_to(mapa)

            st.markdown("ðﾟﾓﾍ Área aproximada del archivo")
            st_folium(mapa, width='100%', height=600)

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")

# -------------------------------
# 🔍 PREDICCIÓN DESDE GPKG
# -------------------------------

st.markdown("---")
st.header("📊 Predicción y visualización del resultado")

if uploaded_gpkg is not None:
    try:
        # Volvemos a leer el archivo .gpkg completo para mantener geometría
        gdf_input = gpd.read_file(uploaded_gpkg)

        # Aplicar predicción con pipeline entrenado
        gdf_resultado = ctrl.predict_from_gdf(gdf_input)

        # Mostrar tabla con probabilidades
        st.subheader(" Resultados del modelo")
        st.dataframe(gdf_resultado.drop(columns="geometry"))

        # Visualización en mapa
        st.subheader("Mapa con probabilidades")

        # Reproyectar si es necesario
        if gdf_resultado.crs and gdf_resultado.crs.to_epsg() != 4326:
            gdf_resultado = gdf_resultado.to_crs(epsg=4326)

        # Crear mapa centrado
        bounds = gdf_resultado.total_bounds
        centro = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        mapa = folium.Map(location=centro, zoom_start=8, tiles="CartoDB positron")

        # Añadir puntos con color por probabilidad
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

        st_data = st_folium(mapa, width=900, height=600)

        # Botón de descarga
        st.markdown("### Descargar archivo con resultados")
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

