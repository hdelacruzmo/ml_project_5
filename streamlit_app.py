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

