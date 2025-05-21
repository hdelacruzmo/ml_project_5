import Definitions
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from src.back.ModelController import ModelController
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

st.set_page_config(layout="wide", page_title="MaxEnt - Visualizador", page_icon="Sitios Búsquedaﾠ")
st.title(" Delimitación de áreas de interés forense usando Machine Learning y análisis espacial en el Bajo Cauca​")

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

            st.write("Vista previa de los datos de tu archivo:")
            st.dataframe(gdf, height=500, use_container_width=True)
            
            st.write("Vista geográfica del área de estudio:")
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

            #st_folium(mapa, width=900, height=500)
            st_folium(mapa, width='100%', height=600)

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {e}")


# -------------------------------
#  PREDICCIÓN DESDE GPKG - TRES MODELOS
# -------------------------------
st.markdown("---")
st.header("Estimación de probabilidad de sitio de interés para la búsqueda")

ctrl = ModelController()

if uploaded_gpkg is not None:
    try:
        gdf_input = gpd.read_file(uploaded_gpkg)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Modelo 1 : Regresión Logística (MaxEnt)",
            "Modelo 2 : Ensamble de Regresiones",
            "Modelo 3 : Random Forest",
            "Análisis : Comparación por umbral"
        ])

        modelos = [
            ("Modelo 1 : Regresión Logística (MaxEnt)", ctrl.predict_from_gdf(gdf_input)),
            ("Modelo 2 : Ensamble de Regresiones", ctrl.predict_with_second_model(gdf_input)),
            ("Modelo 3 : Random Forest", ctrl.predict_with_third_model(gdf_input))
        ]

        for nombre_modelo, gdf_resultado in modelos:
            tab = tab1 if nombre_modelo == "Modelo 1 : Regresión Logística (MaxEnt)" else tab2 if nombre_modelo == "Modelo 2 : Ensamble de Regresiones" else tab3

            with tab:
                st.subheader(f"Resultados del modelo {nombre_modelo}")

                ### acá

                col1, col2 = st.columns([1, 1])
    
                with col1:
                    st.markdown("### Mapa estático del modelo")
                    x_coords = gdf_resultado.geometry.x
                    y_coords = gdf_resultado.geometry.y
                    probs = gdf_resultado["probabilidad"]
                
                    fig, ax = plt.subplots(figsize=(3.5, 3.5))
                    scatter = ax.scatter(
                        x_coords, y_coords, c=probs,
                        cmap="viridis", s=3, edgecolor="none",
                        vmin=0, vmax=1
                    )
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.01)
                    cbar.set_label("Probabilidad")
                    ax.set_title("Distribución espacial de probabilidad")
                    ax.axis("off")
                    st.pyplot(fig)
                
                with col2:
                    # Espaciador vertical
                    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
                
                    st.markdown("### Estadísticas Generales:")
                    st.markdown(f"- **Número total de puntos**: {len(gdf_resultado)}")
                    st.markdown(f"- **Probabilidad promedio**: {gdf_resultado['probabilidad'].mean():.3f}")
                    st.markdown(f"- **Máxima**: {gdf_resultado['probabilidad'].max():.3f} | **Mínima**: {gdf_resultado['probabilidad'].min():.3f}")
                    st.markdown(f"- **Puntos con probabilidad ≥ 0.8**: {(gdf_resultado['probabilidad'] >= 0.8).sum()}")


                tabla_mostrar = (
                    gdf_resultado
                    .drop(columns=["geometry", "FID_Mina"], errors="ignore")
                    .rename(columns={
                        "Num_PrediosURT": "PrediosURT",
                        "Minas1000m": "Dens_Minas",
                        "Dist_EventoCombatiente": "Dist_Comb",
                        "Dens_EventoCombatiente": "Dens_Comb"
                    })
                )
                #color en la columna probabilidad
                st.dataframe(
                    tabla_mostrar,
                    height=400,
                    use_container_width=True,
                    column_config={
                        "probabilidad": st.column_config.NumberColumn(
                            "Probabilidad",
                            help="Probabilidad de presencia (0 = baja, 1 = alta)",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f"
                        )
                    }
                )
                #st.dataframe(tabla_mostrar, height=400, use_container_width=True)
                
                st.subheader("Estadísticas por rangos de probabilidad")
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]
                gdf_resultado["rango_probabilidad"] = pd.cut(
                    gdf_resultado["probabilidad"], bins=bins, labels=labels, include_lowest=True
                )
                conteo = gdf_resultado["rango_probabilidad"].value_counts().sort_index()

                conteo_df = conteo.reset_index()
                conteo_df.columns = ["Rango", "Cantidad"]

                # Definir colores fijos en orden de los labels
                viridis_5 = {
                    "0–0.2": "#440154",
                    "0.2–0.4": "#3b528b",
                    "0.4–0.6": "#21918c",
                    "0.6–0.8": "#5ec962",
                    "0.8–1": "#fde725"
                }
                
                fig = px.bar(
                    conteo_df,
                    x="Rango",
                    y="Cantidad",
                    color="Rango",
                    title="Cantidad de puntos por rango de probabilidad",
                    text="Cantidad",
                    color_discrete_map=viridis_5
                )
                fig.update_layout(xaxis_title="Rango de probabilidad", yaxis_title="Número de puntos")
                st.plotly_chart(fig, use_container_width=True)

                ### Exploración interactiva por variable
                st.markdown("### Exploración interactiva por variable")

                # Variables predictoras (excepto probabilidad y tipo_punto)
                variables_numericas = [
                    col for col in tabla_mostrar.columns
                    if col not in ["probabilidad", "tipo_punto"] and pd.api.types.is_numeric_dtype(tabla_mostrar[col])
                ]
                
                var_x = st.selectbox("Selecciona una variable para el eje X", variables_numericas, key=f"x_{nombre_modelo}")
                
                fig = px.scatter(
                    tabla_mostrar,
                    x=var_x,
                    y="probabilidad",
                    color="probabilidad",
                    color_continuous_scale="Viridis",
                    title=f"Probabilidad vs {var_x}",
                    hover_data=tabla_mostrar.columns
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)  


                
                ### Descargar archivo con resultados
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
            st.subheader("Coincidencias por umbral en los tres modelos")

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
            st.data_editor(seleccionados.drop(columns="geometry"), height=500, use_container_width=True, disabled=True)

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
