import streamlit as st
import pandas as pd
from pycaret.time_series import load_model, predict_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import timedelta

# Configuración de la página en Streamlit
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
    entidad = st.sidebar.selectbox("Entidad Financiera", ["Todo", "Diners Club", "Banco del Pichincha"])

    # Cargar el modelo y los datos correspondientes según la selección
    if entidad == "Todo":
        modelo_path = 'datos_deploy/modelTesisTodo'
        datos_path = 'datos_deploy/DatosActualesTesisTodo.csv'
    elif entidad == "Diners Club":
        modelo_path = 'datos_deploy/modelTesisDinersClub'
        datos_path = 'datos_deploy/DatosActualesTesisDinersClub.csv'
    elif entidad == "Banco del Pichincha":
        modelo_path = 'datos_deploy/modelTesisBancoPichincha'
        datos_path = 'datos_deploy/DatosActualesTesisBancoPichincha.csv'

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
            
            # Generar índice futuro para las predicciones
            prediccion.index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periodos_prediccion, freq='M')
            
            # Calcular las métricas
            valores_reales = data[-periodos_prediccion:]['Total']  # Ajusta 'Total' al nombre de la columna correcta si es diferente
            prediccion_valores = prediccion['y_pred'].values  # Ajusta 'y_pred' si es necesario

            mse = mean_squared_error(valores_reales, prediccion_valores)
            rmse = mse ** 0.5
            mae = mean_absolute_error(valores_reales, prediccion_valores)
            r2 = r2_score(valores_reales, prediccion_valores)

            # Mostrar las métricas de evaluación
            st.subheader("Métricas de Evaluación")
            st.write(f"MSE: {mse:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"R²: {r2:.2f}")

            # Concatenar los datos históricos y las predicciones para visualización
            data_completa = pd.concat([data, prediccion], axis=0)
            
            # Mostrar las predicciones y los datos históricos juntos
            st.subheader("Predicciones y Datos de Entrenamiento")
            st.line_chart(data_completa)

        except Exception as e:
            st.error(f"Ha ocurrido un error durante la predicción: {e}")
