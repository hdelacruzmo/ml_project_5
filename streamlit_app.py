import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
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
            st.dataframe(gdf, height=500, use_container_width=True, disabled=True)

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

            st_folium(mapa, width=900, height=500)


        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")


# -------------------------------
#  PREDICCIÓN DESDE GPKG - TRES MODELOS
# -------------------------------
st.markdown("---")
st.header("Predicción y resumen estadístico")

ctrl = ModelController()

if uploaded_gpkg is not None:
    try:
        gdf_input = gpd.read_file(uploaded_gpkg)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Modelo 1: MaxEnt",
            "Modelo 2: Alternativo",
            "Modelo 3: Random Forest",
            "Comparación por umbral"
        ])

        modelos = [
            ("MaxEnt", ctrl.predict_from_gdf(gdf_input)),
            ("Alternativo", ctrl.predict_with_second_model(gdf_input)),
            ("Random Forest", ctrl.predict_with_third_model(gdf_input))
        ]

        for nombre_modelo, gdf_resultado in modelos:
            tab = tab1 if nombre_modelo == "MaxEnt" else tab2 if nombre_modelo == "Alternativo" else tab3

            with tab:
                st.subheader(f"Resultados del modelo {nombre_modelo}")
                st.dataframe(gdf_resultado.drop(columns="geometry"), height=400, use_container_width=True)

                st.subheader("Estadísticas por rangos de probabilidad")
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
                gdf_resultado["rango_probabilidad"] = pd.cut(
                    gdf_resultado["probabilidad"], bins=bins, labels=labels, include_lowest=True
                )
                conteo = gdf_resultado["rango_probabilidad"].value_counts().sort_index()

                conteo_df = conteo.reset_index()
                conteo_df.columns = ["Rango", "Cantidad"]

                fig = px.bar(
                    conteo_df,
                    x="Rango",
                    y="Cantidad",
                    color="Rango",
                    title="Cantidad de puntos por rango de probabilidad",
                    color_discrete_sequence=px.colors.sequential.YlGnBu,
                    text="Cantidad"
                )
                fig.update_layout(xaxis_title="Rango de probabilidad", yaxis_title="Número de puntos")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Estadísticas adicionales:")
                st.markdown(f"- Número total de puntos: **{len(gdf_resultado)}**")
                st.markdown(f"- Probabilidad promedio: **{gdf_resultado['probabilidad'].mean():.3f}**")
                st.markdown(f"- Máxima: **{gdf_resultado['probabilidad'].max():.3f}** | Mínima: **{gdf_resultado['probabilidad'].min():.3f}**")
                st.markdown(f"- Puntos con probabilidad ≥ 0.8: **{(gdf_resultado['probabilidad'] >= 0.8).sum()}**")

                st.markdown("### Descargar archivo con resultados")
                output_path = f"/tmp/resultados_{nombre_modelo.lower().replace(' ', '_')}.gpkg"
                gdf_resultado.to_file(output_path, driver="GPKG")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label=f"Descargar GPKG - {nombre_modelo}",
                        data=f,
                        file_name=f"resultados_{nombre_modelo.lower().replace(' ', '_')}.gpkg",
                        mime="application/octet-stream"
                    )

        # 🟦 Cuarta pestaña — COMPARACIÓN POR UMBRAL
        with tab4:
            st.subheader("🎯 Coincidencias por umbral en los tres modelos")

            umbral = st.number_input("Selecciona el umbral mínimo", min_value=0.0, max_value=1.0, step=0.01, value=0.8)

            comparado = modelos[0][1][["probabilidad"]].rename(columns={"probabilidad": "prob_modelo_1"}).copy()
            comparado["prob_modelo_2"] = modelos[1][1]["probabilidad"].values
            comparado["prob_modelo_3"] = modelos[2][1]["probabilidad"].values
            comparado["geometry"] = modelos[0][1].geometry.values

            seleccionados = comparado[
                (comparado["prob_modelo_1"] >= umbral) &
                (comparado["prob_modelo_2"] >= umbral) &
                (comparado["prob_modelo_3"] >= umbral)
            ]

            st.markdown(f"🔎 Se encontraron **{len(seleccionados)} puntos** donde los tres modelos tienen probabilidad ≥ {umbral:.2f}")
            st.data_editor(df.drop(columns="geometry"), height=500, use_container_width=True, disabled=True)

            salida_path = "/tmp/seleccionados_tres_modelos.gpkg"
            seleccionados_gdf = gpd.GeoDataFrame(seleccionados, geometry="geometry", crs=modelos[0][1].crs)
            seleccionados_gdf.to_file(salida_path, driver="GPKG")
            with open(salida_path, "rb") as f:
                st.download_button(
                    label="📥 Descargar selección como GPKG",
                    data=f,
                    file_name="seleccionados_tres_modelos.gpkg",
                    mime="application/octet-stream"
                )

    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
