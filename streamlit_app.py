import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from src.back.ModelController import ModelController

st.set_page_config(layout="wide", page_title="MaxEnt Bounding Box", page_icon="🧠")

st.title("🧠 Visualizador y Predicción - Modelo MaxEnt")

with st.expander("🗺️ Ver área geográfica cubierta (.gpkg)"):
    uploaded_file = st.file_uploader(
        "📂 Sube tu archivo GPKG", accept_multiple_files=False, type=["gpkg"], key="gpkg"
    )

    if uploaded_file is not None:
        try:
            gdf = gpd.read_file(uploaded_file)

            if gdf.crs and gdf.crs.to_epsg() != 4326:
                st.info(f"📐 Reproyectando desde {gdf.crs} a EPSG:4326 para visualización.")
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

            # Basemaps
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

            st.markdown("📍 Área aproximada del archivo")
            st_folium(mapa, width='100%', height=600)

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")

# ----------------------------------------------------
# SECCIÓN DE PREDICCIÓN
# ----------------------------------------------------

st.markdown("---")
st.header("📊 Predicciones con Modelos Entrenados")

ctrl = ModelController()

uploaded_csv = st.file_uploader(
    "📂 Sube tu archivo CSV con datos de entrada", type=["csv"], key="csv_prediccion"
)

if uploaded_csv is not None:
    try:
        bytes_data = uploaded_csv.getvalue()
        input_df, is_valid = ctrl.load_input_data(bytes_data)

        if not is_valid:
            st.error("⚠️ Las columnas del CSV no coinciden con lo esperado.")
        else:
            rf_df, maxent_df, full_df = ctrl.predict()

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Input Data", "Stats", "Random Forest", "MaxEnt", "Full View"
            ])

            with tab1:
                st.subheader("🧩 Datos de entrada")
                st.dataframe(input_df)

            with tab2:
                st.subheader("📊 Estadísticas del dataset")
                st.dataframe(input_df.describe())

            with tab3:
                st.subheader("🌲 Random Forest")
                st.dataframe(rf_df)

            with tab4:
                st.subheader("🧠 MaxEnt (Regresión Logística)")
                st.dataframe(maxent_df)

            with tab5:
                st.subheader("🔍 Comparación entre modelos")
                st.dataframe(full_df)

    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {e}")

