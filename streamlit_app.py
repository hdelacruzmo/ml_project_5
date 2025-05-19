import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from src.back.ModelController import ModelController

st.set_page_config(layout="wide", page_title="MaxEnt - Visualizador", page_icon="Sitios Búsquedaﾠ")
st.title(" Visualizador y Predicción con Modelo Regresión logística")

# -------------------------------
#  VISUALIZACIÓN DEL GPKG
# -------------------------------

with st.expander(" Ver área geográfica cubierta (.gpkg)"):
    uploaded_gpkg = st.file_uploader(
        "Sube tu archivo GPKG", accept_multiple_files=False, type=["gpkg"], key="gpkg"
    )

    if uploaded_gpkg is not None:
        try:
            gdf = gpd.read_file(uploaded_gpkg)

            if gdf.crs and gdf.crs.to_epsg() != 4326:
                st.info(f" Reproyectando desde {gdf.crs} a EPSG:4326 para visualización.")
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

            st.markdown("Área aproximada del archivo")
            st_folium(mapa, width='100%', height=600)

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")



# -------------------------------
#  PREDICCIÓN DESDE GPKG
# -------------------------------

st.markdown("---")
st.header("Predicción y resumen estadístico")

ctrl = ModelController()

if uploaded_gpkg is not None:
    try:
        gdf_input = gpd.read_file(uploaded_gpkg)
        gdf_resultado = ctrl.predict_from_gdf(gdf_input)

        st.subheader(" Resultados del modelo")
        st.dataframe(gdf_resultado.drop(columns="geometry").head())

        # Solo estadísticas, visualización comentada
        st.subheader("Estadísticas por rangos de probabilidad")
        import matplotlib.pyplot as plt
        import seaborn as sns

        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
        gdf_resultado["rango_probabilidad"] = pd.cut(
            gdf_resultado["probabilidad"], bins=bins, labels=labels, include_lowest=True
        )
        conteo = gdf_resultado["rango_probabilidad"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=conteo.index, y=conteo.values, palette="coolwarm", ax=ax)
        ax.set_title("Cantidad de puntos por rango de probabilidad")
        ax.set_xlabel("Rango de probabilidad")
        ax.set_ylabel("Número de puntos")
        st.pyplot(fig)

        st.markdown("### Estadísticas adicionales:")
        st.markdown(f"- Número total de puntos: **{len(gdf_resultado)}**")
        st.markdown(f"- Probabilidad promedio: **{gdf_resultado['probabilidad'].mean():.3f}**")
        st.markdown(f"- Máxima: **{gdf_resultado['probabilidad'].max():.3f}** | Mínima: **{gdf_resultado['probabilidad'].min():.3f}**")
        st.markdown(f"- Puntos con probabilidad ≥ 0.8: **{(gdf_resultado['probabilidad'] >= 0.8).sum()}**")

        # Exportación .gpkg
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
