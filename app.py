import streamlit as st
import pandas as pd
from pycaret.time_series import load_model, predict_model
from datetime import timedelta

# Configuración de la página en Streamlits
st.set_page_config(page_title="Predicción de Cantidad de Tarjetas de Crèdito", layout="wide")

# Título de la aplicación
st.title("Predicción de Cantidad de Tarjetas de Crédito")

# Menú de opciones
opcion = st.sidebar.selectbox("Seleccione una opción", ["DASHBOARD", "PREDICCIONES"])

# Enlace al dashboard de Power BI
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiNzg1ZjE1OTktMDNiYS00NDA4LTg4MmMtOTYxZWEzNzhiN2I5IiwidCI6ImI3YWY4Y2FmLTgzZDgtNDY0NC04NWFlLTMxN2M1NDUyMjNjMSIsImMiOjR9"  # Reemplaza XXXX con tu enlace de embebido de Power BI

# Opciones del menú
if opcion == "DASHBOARD":
    # Mostrar el dashboard embebido
    st.subheader("Dashboard Interactivo")
    st.markdown(f'<iframe title="DashboardTesis" width="1024" height="612" src="https://app.powerbi.com/view?r=eyJrIjoiNzg1ZjE1OTktMDNiYS00NDA4LTg4MmMtOTYxZWEzNzhiN2I5IiwidCI6ImI3YWY4Y2FmLTgzZDgtNDY0NC04NWFlLTMxN2M1NDUyMjNjMSIsImMiOjR9" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)

elif opcion == "PREDICCIONES":
    # Título de la sección de predicciones
    st.subheader("Proyectar la cantidad de tarjetas")
    
    # Selección de entidad financiera
    st.sidebar.header("Seleccione la Entidad Financiera")
    entidad = st.sidebar.selectbox("Entidad Financiera", ["Todo", "Diners Club", "Produbanco"])

    # Cargar el modelo y los datos correspondientes según la selección
    if entidad == "Todo":
        modelo_path = 'datos_deploy/modelTesis_Todo'
        datos_path = 'datos_deploy/Datos_Todo.csv'
    elif entidad == "Diners Club":
        modelo_path = 'datos_deploy/modelTesis_Diners Club'
        datos_path = 'datos_deploy/Datos_Diners Club.csv'
    elif entidad == "Produbanco":
        modelo_path = 'datos_deploy/modelTesis_Produbanco'
        datos_path = 'datos_deploy/Datos_Produbanco.csv'

    # Cargar el modelo seleccionado
    try:
        modelo = load_model(modelo_path)
        st.write(f"Modelo cargado: {modelo_path.split('/')[-1]}")
    except FileNotFoundError:
        st.error("Modelo no encontrado. Verifique la ruta y vuelva a intentarlo.")

    # Cargar los datos históricos seleccionados
    try:
        data = pd.read_csv(datos_path, index_col='Fecha', parse_dates=True)
        st.subheader(f"Valores reales de la entidad: {entidad}")
        st.line_chart(data)
    except FileNotFoundError:
        st.error("Datos no encontrados. Verifique la ruta y vuelva a intentarlo.")

    # Configuración del horizonte de predicción en la barra lateral
    st.sidebar.header("Parámetros de Predicción")
    periodos_prediccion = st.sidebar.slider("Horizonte de Predicción (número de periodos)", min_value=1, max_value=24, value=3)

    # Realizar la predicción cuando se presiona el botón
    if st.button("Generar Predicción"):
        try:
            # Realizar la predicción con el modelo cargado
            prediccion = predict_model(modelo, fh=periodos_prediccion)
            
            # Renombrar la columna 'y_pred' a 'Predicción'
            prediccion.rename(columns={'y_pred': 'Predicción'}, inplace=True)
            
            # Generar índice futuro para las predicciones, asegurándose de que las fechas sean el primer día de cada mes
            prediccion.index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periodos_prediccion, freq='MS')  # 'MS' es Month Start
            
            # Mostrar la tabla con los valores predichos
            st.subheader("Valores Predichos")
            st.write(prediccion[['Predicción']])  # Mostrar solo las predicciones con la nueva etiqueta

            # Concatenar los datos históricos y las predicciones para visualización
            data_completa = pd.concat([data, prediccion[['Predicción']]], axis=0)
            
            # Mostrar las predicciones y los datos históricos juntos
            st.subheader("Predicciones y Datos de Entrenamiento")
            st.line_chart(data_completa)

        except Exception as e:
            st.error(f"Ha ocurrido un error durante la predicción: {e}")
