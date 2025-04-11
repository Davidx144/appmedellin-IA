import streamlit as st
import pandas as pd
import anthropic
import os
import json
import tempfile
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import seaborn as sns
from matplotlib.figure import Figure

# Función para generar gráficos basados en las sugerencias de Claude
def generate_charts(chart_specs, dataframes):
    charts = []
    
    for spec in chart_specs:
        try:
            # Obtener la hoja de datos correcta
            hoja = spec.get("hoja", "")
            if hoja not in dataframes:
                continue
            
            df = dataframes[hoja]
            
            # Verificar que las columnas existen
            eje_x = spec.get("eje_x", "")
            eje_y = spec.get("eje_y", "")
            color_por = spec.get("color_por", None)
            agrupar_por = spec.get("agrupar_por", None)
            operacion = spec.get("operacion", "sum")
            titulo = spec.get("titulo", "Gráfico")
            tipo = spec.get("tipo", "").lower()
            
            # Validar columnas
            if eje_x not in df.columns or (eje_y and eje_y not in df.columns):
                continue
                
            # Aplicar agrupación si es necesario
            if agrupar_por and agrupar_por in df.columns:
                if operacion == "sum":
                    df_agg = df.groupby(agrupar_por)[eje_y].sum().reset_index()
                elif operacion == "mean":
                    df_agg = df.groupby(agrupar_por)[eje_y].mean().reset_index()
                elif operacion == "count":
                    df_agg = df.groupby(agrupar_por)[eje_y].count().reset_index()
                elif operacion == "max":
                    df_agg = df.groupby(agrupar_por)[eje_y].max().reset_index()
                elif operacion == "min":
                    df_agg = df.groupby(agrupar_por)[eje_y].min().reset_index()
                else:
                    df_agg = df
            else:
                df_agg = df
            
            # Crear figura según el tipo
            fig = None
            
            if tipo == "linea":
                if color_por and color_por in df.columns:
                    fig = px.line(df_agg, x=eje_x, y=eje_y, color=color_por, title=titulo)
                else:
                    fig = px.line(df_agg, x=eje_x, y=eje_y, title=titulo)
                    
            elif tipo == "barra":
                if color_por and color_por in df.columns:
                    fig = px.bar(df_agg, x=eje_x, y=eje_y, color=color_por, title=titulo)
                else:
                    fig = px.bar(df_agg, x=eje_x, y=eje_y, title=titulo)
                    
            elif tipo == "dispersion":
                if color_por and color_por in df.columns:
                    fig = px.scatter(df_agg, x=eje_x, y=eje_y, color=color_por, title=titulo)
                else:
                    fig = px.scatter(df_agg, x=eje_x, y=eje_y, title=titulo)
                    
            elif tipo == "pastel":
                fig = px.pie(df_agg, names=eje_x, values=eje_y, title=titulo)
                
            elif tipo == "caja":
                fig = px.box(df_agg, x=eje_x, y=eje_y, title=titulo)
                
            elif tipo == "histograma":
                fig = px.histogram(df_agg, x=eje_x, title=titulo)
                
            elif tipo == "heatmap":
                pivot = df_agg.pivot_table(values=eje_y, index=eje_x, columns=agrupar_por, aggfunc=operacion)
                fig = px.imshow(pivot, title=titulo)
                
            elif tipo == "area":
                fig = px.area(df_agg, x=eje_x, y=eje_y, title=titulo)
            
            if fig:
                charts.append((fig, spec.get("descripcion", "")))
            
        except Exception as e:
            continue
    
    return charts


# Extraer sugerencias de visualización de la respuesta de Claude
def extract_chart_suggestions(response_text):
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```"
    matches = re.search(pattern, response_text)
    
    if not matches:
        return []
    
    try:
        json_str = matches.group(1)
        chart_specs = json.loads(json_str)
        return chart_specs
    except Exception as e:
        return []

# Mostrar respuesta sin las sugerencias de visualización
def clean_response(response_text):
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*[\s\S]*?\s*```"
    clean_text = re.sub(pattern, "", response_text)
    return

# Mostrar respuesta sin las sugerencias de visualización
def clean_response(response_text):
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*[\s\S]*?\s*```"
    clean_text = re.sub(pattern, "", response_text)
    return clean_text

# Configuración de la página
st.set_page_config(page_title="Analizador de Excel con Claude", page_icon="📊", layout="wide")
st.title("📊 Analizador Contable de Excel con IA")
st.markdown("Carga tu archivo Excel y haz preguntas sobre tus datos")

# Sidebar para API key
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Anthropic API Key", type="password")
    st.markdown("Necesitas una API key de Claude para usar esta aplicación.")
    st.markdown("Obtén tu API key en [Anthropic Console](https://console.anthropic.com/)")
    
    st.header("Modelo")
    modelo = st.selectbox(
        "Selecciona el modelo de Claude",
        ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    )

# Función para procesar el archivo Excel
def process_excel(uploaded_file):
    # Leer el archivo Excel
    excel_data = BytesIO(uploaded_file.getvalue())
    return excel_data

# Función para convertir objetos datetime y otras estructuras complejas para JSON
def convert_datetime_keys(obj):
    """
    Convierte recursivamente cualquier clave datetime en string en un diccionario.
    También convierte valores de datetime a string.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convertir clave si es datetime
            if isinstance(k, (pd.Timestamp, np.datetime64)) or hasattr(k, 'strftime'):
                k = str(k)
            # Convertir valor recursivamente
            new_dict[k] = convert_datetime_keys(v)
        return new_dict
    elif isinstance(obj, list):
        # Procesar recursivamente listas
        return [convert_datetime_keys(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, np.datetime64)) or hasattr(obj, 'strftime'):
        # Convertir cualquier objeto datetime a string
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Función para procesar DataFrames para JSON
def process_dataframe_for_json(df):
    """Prepara un DataFrame para serialización JSON segura"""
    # Convertir fechas y timestamps a cadenas
    for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
        df[col] = df[col].astype(str)
    
    # Convertir NaN a None
    df_clean = df.replace({np.nan: None})
    
    # Convertir el DataFrame a dict y asegurar claves seguras
    records = df_clean.to_dict(orient="records")
    return convert_datetime_keys(records)

# Función para convertir a contenido para Claude
def get_dataframes_info(excel_file):
    # Cargamos todas las hojas del Excel
    excel_data = pd.ExcelFile(excel_file)
    sheet_names = excel_data.sheet_names
    
    dataframes_info = []
    
    # Procesamos cada hoja
    for sheet in sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet)
        
        # Procesar el DataFrame para JSON
        datos_completos = process_dataframe_for_json(df)
        
        # Información básica de la hoja (ya procesada para ser JSON segura)
        info = {
            "nombre_hoja": sheet,
            "filas": len(df),
            "columnas": len(df.columns),
            "nombres_columnas": list(df.columns),
            "datos_completos": datos_completos
        }
        
        # Procesar resumen estadístico para ser JSON seguro
        if df.select_dtypes(include=['number']).shape[1] > 0:
            stats_dict = df.describe().to_dict()
            info["resumen_estadistico"] = convert_datetime_keys(stats_dict)
        else:
            info["resumen_estadistico"] = {}
        
        # Procesar tipos de datos
        info["tipos_datos"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Procesar valores únicos
        valores_unicos = {}
        for col in df.columns:
            if df[col].dtype != 'object' or len(df[col].unique()) < 30:
                unicos = list(df[col].unique())[:30]
                # Convertir valores NaN, datetime, etc.
                valores_unicos[col] = convert_datetime_keys(unicos)
        
        info["valores_unicos"] = valores_unicos
        
        dataframes_info.append(info)
    
    return dataframes_info

# Función para consultar a Claude
def query_claude(dataframes_info, question, api_key, modelo, generate_charts=True):
    client = anthropic.Anthropic(api_key=api_key)

    # Crear el sistema prompt para Claude
    system_prompt = f"""
    Eres un asistente experto en análisis de datos de Excel, especialista en contabilidad. Se te ha proporcionado el contenido completo de las hojas de un archivo Excel.
    Tu tarea es responder preguntas sobre los datos utilizando la información proporcionada.

    Utiliza todos los datos disponibles para realizar un análisis completo y preciso. Puedes realizar cálculos estadísticos,
    identificar tendencias, crear resúmenes y responder cualquier pregunta basada en estos datos.

    { "INSTRUCCIONES PARA GRÁFICOS:" if generate_charts else "" }

    { 
    '''  # <-- Añade triple comillas aquí para iniciar una cadena multilinea
    SI LA PREGUNTA REQUIERE VISUALIZACIONES, TU RESPUESTA DEBE INCLUIR UNA SECCIÓN ESPECIAL LLAMADA "SUGERENCIAS_DE_VISUALIZACIÓN"
    CON INSTRUCCIONES PRECISAS EN FORMATO JSON PARA CREAR HASTA 3 GRÁFICOS RELEVANTES.

    Usa exactamente este formato para cada gráfico, incluyendo las llaves y corchetes:

    SUGERENCIAS_DE_VISUALIZACIÓN:
    ```json
    [
      {{  # <-- Escapa la llave de apertura con {{
        "tipo": "linea|barra|dispersion|pastel|caja|histograma|heatmap|area",
        "titulo": "Título descriptivo del gráfico",
        "hoja": "Nombre de la hoja con los datos",
        "eje_x": "Nombre exacto de la columna para el eje X",
        "eje_y": "Nombre exacto de la columna para el eje Y",
        "color_por": "Nombre de columna para colorear (opcional)",
        "agrupar_por": "Nombre de columna para agrupar datos (opcional)",
        "operacion": "sum|mean|count|max|min (solo si hay agrupación)",
        "descripcion": "Breve explicación de lo que muestra este gráfico"
      }}, # <-- Escapa la llave de cierre con }}
      {{  # <-- Escapa la llave de apertura
        // Segundo gráfico si es necesario, con el mismo formato
      }}, # <-- Escapa la llave de cierre
      {{  # <-- Escapa la llave de apertura
        // Tercer gráfico si es necesario, con el mismo formato
      }}  # <-- Escapa la llave de cierre
    ]
    ```

    Es FUNDAMENTAL que incluyas esta sección si la pregunta se beneficia de visualizaciones.
    Asegúrate de que las columnas que menciones existan exactamente en los datos y sean del tipo adecuado.
    ''' 
    if generate_charts else "" 
    } # Fin del bloque condicional

    El análisis debe ser exhaustivo y considerar todo el conjunto de datos, no solo muestras.
    """
    
    # Preparar un contenido más estructurado para Claude
    content = [
        {"type": "text", "text": question},
        {"type": "text", "text": "A continuación se muestra la información de las hojas del Excel para que puedas realizar tu análisis completo:"}
    ]
    
    # Agregar información de cada hoja de forma más estructurada
    for idx, sheet_info in enumerate(dataframes_info):
        # Convertir datos para asegurar que sean serializables
        safe_sheet_info = convert_datetime_keys(sheet_info)
        
        # Crear una versión segura para JSON del diccionario de datos
        try:
            # Limitamos a 1000 registros para evitar problemas de tamaño
            datos_json = json.dumps(safe_sheet_info['datos_completos'][:1000], ensure_ascii=False)
        except TypeError as e:
            st.warning(f"Error de serialización en hoja {safe_sheet_info['nombre_hoja']}: {str(e)}. Intentando método alternativo.")
            # Si hay algún error, fallback a una versión simplificada
            datos_simplificados = []
            for record in safe_sheet_info['datos_completos'][:1000]:
                simplified_record = {}
                for k, v in record.items():
                    if isinstance(k, (str, int, float, bool, type(None))):
                        if isinstance(v, (str, int, float, bool, type(None))):
                            simplified_record[k] = v
                        else:
                            simplified_record[k] = str(v)
                datos_simplificados.append(simplified_record)
            datos_json = json.dumps(datos_simplificados, ensure_ascii=False)
        
        sheet_content = f"""
        === HOJA {idx+1}: {safe_sheet_info['nombre_hoja']} ===

        Información básica:
        - Filas: {safe_sheet_info['filas']}
        - Columnas: {safe_sheet_info['columnas']}
        - Columnas: {', '.join([str(col) for col in safe_sheet_info['nombres_columnas']])}

        Datos completos (en formato JSON):
        {datos_json}
        
        """
        
        content.append({"type": "text", "text": sheet_content})
    
    try:
        # Enviar la consulta a Claude con un límite de tokens más alto
        response = client.messages.create(
            model=modelo,
            max_tokens=6000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error al consultar a Claude: {str(e)}"

# Componente para cargar el archivo
uploaded_file = st.file_uploader("Carga tu archivo Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Mostrar información básica del archivo
    st.success(f"Archivo cargado: {uploaded_file.name}")
    
    # Procesar el archivo
    excel_data = process_excel(uploaded_file)
    
    # Mostrar pestañas con visualización previa
    excel_file = pd.ExcelFile(excel_data)
    sheet_names = excel_file.sheet_names
    
    tabs = st.tabs(sheet_names)
    
    for i, tab in enumerate(tabs):
        with tab:
            df = pd.read_excel(excel_file, sheet_name=sheet_names[i])
            st.dataframe(df.head(10), use_container_width=True)
            st.text(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Sección para hacer preguntas
    st.header("Haz una pregunta sobre tus datos")
    question = st.text_area("Tu pregunta:", height=100, 
                           placeholder="Ej: ¿Cuál es el total de ventas por región? o ¿Qué tendencias observas en los datos?")
    
    # Opciones avanzadas
    with st.expander("Opciones avanzadas"):
        max_rows = st.slider("Número máximo de filas a analizar por hoja", 
                             min_value=100, max_value=10000, value=1000, step=100,
                             help="Limitar el número de filas puede mejorar el rendimiento con archivos grandes")
        include_stats = st.checkbox("Incluir estadísticas descriptivas", value=True)
        include_charts = st.checkbox("Sugerir visualizaciones", value=True)
    
    if st.button("Preguntar a Claude"):
        if not api_key:
            st.error("Por favor, ingresa tu API key de Anthropic en la barra lateral.")
        elif not question:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Claude está analizando tus datos..."):
                # Extraer información para Claude con las opciones seleccionadas
                try:
                    # Reprocesar con los parámetros actualizados
                    excel_data_pd = pd.ExcelFile(excel_data)
                    dataframes_info = []
                    dataframes_dict = {}  # Para usar en la generación de gráficos
                    
                    # Procesar cada hoja con las restricciones definidas
                    for sheet in excel_data_pd.sheet_names:
                        df = pd.read_excel(excel_data_pd, sheet_name=sheet)
                        
                        # Limitar filas si es necesario
                        if len(df) > max_rows:
                            st.info(f"La hoja '{sheet}' tiene {len(df)} filas. Limitando a {max_rows} filas para el análisis.")
                            df = df.head(max_rows)
                        
                        # Guardar el dataframe para uso en visualización
                        dataframes_dict[sheet] = df
                        
                        # Procesar datos para JSON
                        datos_seguros = process_dataframe_for_json(df)
                        
                        # Construir información
                        info = {
                            "nombre_hoja": sheet,
                            "filas": len(df),
                            "columnas": len(df.columns),
                            "nombres_columnas": list(df.columns),
                            "datos_completos": datos_seguros
                        }
                        
                        # Añadir estadísticas si se solicita
                        if include_stats:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                # Usar convert_datetime_keys para asegurar que es serializable
                                info["resumen_estadistico"] = convert_datetime_keys(df[numeric_cols].describe().to_dict())
                            
                            # Añadir conteos para columnas categóricas
                            for col in df.select_dtypes(include=['object']).columns:
                                if df[col].nunique() < 30:  # Solo categorías razonables
                                    # Usar convert_datetime_keys para asegurar que es serializable
                                    info[f"conteo_{col}"] = convert_datetime_keys(df[col].value_counts().to_dict())
                        
                        dataframes_info.append(info)
                    
                    # Obtener la respuesta de Claude con instrucciones para visualizaciones
                    response = query_claude(dataframes_info, question, api_key, modelo, include_charts)
                    
                    # Extraer y procesar sugerencias de visualización
                    chart_specs = []
                    if include_charts:
                        chart_specs = extract_chart_suggestions(response)
                        
                    # Limpiar respuesta para mostrarla sin las sugerencias de visualización
                    clean_text = clean_response(response)
                    
                    st.header("Respuesta")
                    st.markdown(clean_text)
                    
                    # Generar y mostrar visualizaciones
                    if chart_specs:
                        st.header("Visualizaciones")
                        charts = generate_charts(chart_specs, dataframes_dict)
                        
                        if charts:
                            chart_tabs = st.tabs([f"Gráfico {i+1}" for i in range(len(charts))])
                            for i, (fig, desc) in enumerate(charts):
                                with chart_tabs[i]:
                                    st.plotly_chart(fig, use_container_width=True)
                                    if desc:
                                        st.write(f"**Descripción:** {desc}")
                        else:
                            st.info("No se pudieron generar las visualizaciones sugeridas.")
                    
                except Exception as e:
                    st.error(f"Error al procesar los datos: {str(e)}")
                    st.exception(e)
else:
    st.info("👆 Por favor, carga un archivo Excel para comenzar.")




# Información adicional al final
st.markdown("---")
st.markdown("""
### Cómo usar esta aplicación:
1. Carga tu archivo Excel usando el botón de arriba
2. Explora la vista previa de tus datos en las pestañas
3. Escribe una pregunta sobre tus datos
4. Haz clic en "Preguntar a Claude" para obtener un análisis

La aplicación envía solo los metadatos y una muestra de tus datos a la API de Claude, no el archivo completo.
""")