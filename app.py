import streamlit as st
import pandas as pd
# import anthropic # Eliminado
# import openai # Eliminado
import google.generativeai as genai # Importar Gemini
import os
import json
import tempfile
from io import BytesIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Para gráficos más complejos si es necesario
import re

# --- Configuración Inicial y Funciones de Ayuda ---

def clean_response_text(response_text):
    """Elimina la sección de sugerencias de visualización del texto de respuesta."""
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```"
    clean_text = re.sub(pattern, "", response_text)
    return clean_text.strip()

def convert_datetime_keys(obj):
    """Convierte recursivamente claves y valores datetime/numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (pd.Timestamp, np.datetime64)) or hasattr(k, 'strftime'):
                k = str(k)
            new_dict[k] = convert_datetime_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_datetime_keys(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, np.datetime64)) or hasattr(obj, 'strftime'):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj): # Manejar NaT o NaN flotantes que no son None
        return None
    else:
        return obj

def process_dataframe_for_json(df):
    """Prepara un DataFrame para serialización JSON segura."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime', 'datetimetz']).columns:
        df_clean[col] = df_clean[col].astype(str)
    
    df_clean = df_clean.replace({np.nan: None}) 
    
    try:
        records = df_clean.to_dict(orient="records")
        return convert_datetime_keys(records) 
    except Exception as e:
        st.warning(f"Error al convertir DataFrame a dict: {e}. Intentando conversión por filas.")
        processed_records = []
        for _, row in df_clean.iterrows():
            processed_records.append(convert_datetime_keys(row.to_dict()))
        return processed_records


# --- Procesamiento de Archivos Excel ---
# MEJORA 6: Caché de datos
@st.cache_data(ttl=3600) # Cache por 1 hora
def get_single_excel_data_info_cached(_excel_file_bytes_value, file_name, max_rows_limit, include_stats_option):
    """
    Wrapper cacheable para get_single_excel_data_info.
    _excel_file_bytes_value es el resultado de .getvalue() para que sea hasheable.
    """
    excel_file_bytes = BytesIO(_excel_file_bytes_value)
    return get_single_excel_data_info(excel_file_bytes, file_name, max_rows_limit, include_stats_option)

def get_single_excel_data_info(excel_file_bytes, file_name, max_rows_limit, include_stats_option):
    """Procesa un solo archivo Excel y extrae información de sus hojas."""
    excel_data = pd.ExcelFile(excel_file_bytes)
    sheet_names = excel_data.sheet_names
    
    file_info_list = []
    dataframes_for_charts = {}

    # st.write(f"Procesando archivo: **{file_name}** (Hojas encontradas: {', '.join(sheet_names)})") # Movido a la UI principal

    for sheet_name in sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet_name)
        original_rows = len(df)

        # MEJORA 3: Asegurar que se limita el df ANTES de procesarlo para JSON
        if len(df) > max_rows_limit:
            df_limited = df.head(max_rows_limit).copy() 
            # st.info(f"Archivo '{file_name}', Hoja '{sheet_name}': {original_rows} filas. Limitando a {max_rows_limit} filas para el análisis.") # Mensaje en UI
        else:
            df_limited = df.copy()
        
        file_sheet_key = f"{file_name}__{sheet_name}"
        # Guardar el DataFrame (limitado por max_rows_limit) para gráficos y vista previa
        dataframes_for_charts[file_sheet_key] = df_limited.copy() 
        
        datos_completos_json_safe = process_dataframe_for_json(df_limited)
        
        info = {
            "file_name": file_name,
            "sheet_name": sheet_name,
            "file_sheet_key": file_sheet_key,
            "rows_original": original_rows,
            "rows_processed": len(df_limited), 
            "columns_count": len(df_limited.columns),
            "column_names": convert_datetime_keys(list(df_limited.columns)),
            "data_sample_json": datos_completos_json_safe
        }
        
        if include_stats_option:
            numeric_cols = df_limited.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                info["statistical_summary"] = convert_datetime_keys(df_limited[numeric_cols].describe().to_dict())
            
            categorical_cols = df_limited.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_limited[col].nunique() < 30 and df_limited[col].nunique() > 0 : 
                    info[f"value_counts_{col}"] = convert_datetime_keys(df_limited[col].value_counts().to_dict())
        
        info["data_types"] = {str(col): str(dtype) for col, dtype in df_limited.dtypes.items()}
        
        file_info_list.append(info)
            
    return file_info_list, dataframes_for_charts

# --- Interacción con LLMs ---

def get_system_prompt(generate_charts_flag): 
    """Genera el system prompt adaptado para Gemini y la tarea."""
    chart_instructions = ""
    if generate_charts_flag:
        chart_instructions = """
SI LA PREGUNTA REQUIERE VISUALIZACIONES, TU RESPUESTA DEBE INCLUIR UNA SECCIÓN ESPECIAL LLAMADA "SUGERENCIAS_DE_VISUALIZACIÓN"
CON INSTRUCCIONES PRECISAS EN FORMATO JSON PARA CREAR HASTA 3-4 GRÁFICOS RELEVANTES.
Si se comparan dos archivos/hojas, puedes sugerir gráficos que muestren datos de ambos (usa la 'file_and_sheet_key' correcta para cada uno).
Para identificar la fuente de datos de un gráfico, usa 'file_and_sheet_key' que combina el nombre del archivo y la hoja (ej: 'Archivo1__Hoja1').

Usa exactamente este formato para cada gráfico:

SUGERENCIAS_DE_VISUALIZACIÓN:
```json
[
  {
    "tipo": "linea | barra | dispersion | pastel | caja | histograma | heatmap | area | sunburst | treemap | funnel | violin | density_heatmap | scatter_matrix",
    "titulo": "Título descriptivo del gráfico",
    "file_and_sheet_key": "NombreArchivo__NombreHoja", 
    "eje_x": "Nombre exacto de la columna para el eje X (o lista para scatter_matrix)",
    "eje_y": "Nombre exacto de la columna para el eje Y (o lista de columnas para algunos tipos, o dimensiones para scatter_matrix)",
    "color_por": "Nombre de columna para colorear (opcional)",
    "agrupar_por": "Nombre de columna para agrupar datos (opcional, antes de graficar)", 
    "operacion": "sum|mean|count|max|min (solo si hay agrupación)",
    "path": ["ColumnaNivel1", "ColumnaNivel2"] (para sunburst, treemap),
    "values_col": "ColumnaDeValores" (para sunburst, treemap, funnel),
    "names_col": "ColumnaDeNombresOCategorias" (para pastel, funnel),
    "dimensions": ["Col1", "Col2", "Col3"] (para scatter_matrix, opcional, si no, usa todas las numéricas),
    "descripcion": "Breve explicación de lo que muestra este gráfico y su relevancia para la pregunta y los datos. Indica si el gráfico es interactivo."
  }
]
```
Consideraciones para tipos de gráficos:
- `linea`, `barra`, `area`: `eje_y` puede ser una lista de columnas.
- `pastel`: `names_col` para las etiquetas de las porciones, `values_col` para los tamaños. `eje_x` y `eje_y` no se usan directamente.
- `caja`, `violin`: `eje_x` puede ser una categoría, `eje_y` la variable numérica. O `eje_y` una lista de columnas numéricas si `eje_x` no se usa.
- `histograma`: Solo `eje_x`. `color_por` puede usarse para superponer histogramas.
- `heatmap`: Usualmente para matrices de correlación. Si es correlación, indícalo.
- `sunburst`, `treemap`: Requieren `path` (lista de columnas para la jerarquía) y `values_col` (columna numérica para el tamaño).
- `funnel`: Requiere `values_col` (para los valores de cada etapa) y `names_col` (para los nombres de las etapas).
- `density_heatmap`: `eje_x`, `eje_y` numéricos.
- `scatter_matrix`: `dimensions` es una lista de columnas. `color_por` es útil.

Asegúrate de que las columnas mencionadas existan en la 'file_and_sheet_key' especificada y sean del tipo adecuado. Prioriza la claridad y relevancia del gráfico.
"""

    base_prompt = f"""
Eres un asistente experto en análisis de datos de Excel, especialista en contabilidad y comparación de datos.
Se te proporcionará información estructurada (metadatos, nombres de columnas, tipos de datos, resumen estadístico y una muestra de datos en formato JSON) de una o dos hojas de cálculo de Excel previamente seleccionadas por el usuario.
Tu tarea es responder preguntas sobre los datos, compararlos si se proporcionan dos hojas/archivos, identificar tendencias, realizar cálculos y ofrecer insights.

{chart_instructions}

Analiza la información proporcionada de manera exhaustiva.
Si se proporcionan datos de dos hojas (posiblemente de diferentes archivos), enfócate en la comparación cuando sea relevante para la pregunta.
Estructura tu respuesta de forma clara y concisa. Utiliza Markdown para formatear tu respuesta (listas, negritas, etc.).
"""
    return base_prompt

def query_gemini_api(api_key, model_name, system_prompt_text, messages_content_for_llm, max_output_tokens):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt_text,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            # Considerar añadir temperature si se quiere más creatividad o determinismo
            # temperature=0.7 
        )
    )
    try:
        user_prompt_text = ""
        # El contenido del usuario ya está formateado como una cadena única en query_llm
        if messages_content_for_llm and messages_content_for_llm[0]['role'] == 'user':
            # messages_content_for_llm[0]['content'] es una lista de dicts [{"type": "text", "text": prompt_completo}]
            # Necesitamos extraer el texto de esa estructura.
            if isinstance(messages_content_for_llm[0]['content'], list) and \
               len(messages_content_for_llm[0]['content']) > 0 and \
               messages_content_for_llm[0]['content'][0]['type'] == 'text':
                user_prompt_text = messages_content_for_llm[0]['content'][0]['text']
            # Fallback por si la estructura es diferente, aunque no debería serlo con el código actual
            elif isinstance(messages_content_for_llm[0]['content'], str): 
                 user_prompt_text = messages_content_for_llm[0]['content']


        if not user_prompt_text:
            return "Error: No se pudo extraer el contenido del usuario para Gemini."

        response = model.generate_content(user_prompt_text) 
        return response.text
    except Exception as e:
        if hasattr(e, 'message') and "quota" in str(e).lower(): # Convertir e a str para búsqueda
             return f"Error al consultar a Gemini: Se ha excedido la cuota de la API. Por favor, revisa tu plan y uso en Google AI Studio. (Detalle: {str(e)})"
        if hasattr(e, 'message') and "API key not valid" in str(e):
            return f"Error al consultar a Gemini: La API key no es válida. Por favor, verifica tu API Key. (Detalle: {str(e)})"
        # Captura de errores más genéricos de la API de Gemini
        if "response.prompt_feedback" in str(e) and "block_reason" in str(e):
            return f"Error al consultar a Gemini: La solicitud fue bloqueada por políticas de seguridad. (Detalle: {str(e)})"
        return f"Error al consultar a Gemini: {str(e)}"


def query_llm(api_key, model_name, selected_sheets_data_info, question, generate_charts_flag, max_output_tokens_config): 
    """Prepara y envía la consulta al LLM Gemini."""
    
    system_prompt = get_system_prompt(generate_charts_flag) 
    
    prompt_introduction = f"Pregunta del usuario: {question}\n\n"
    
    if len(selected_sheets_data_info) == 1:
        data_info_single = selected_sheets_data_info[0]
        prompt_introduction += f"Se ha proporcionado información de la hoja '{data_info_single['sheet_name']}' del archivo '{data_info_single['file_name']}'. Analiza los siguientes datos:\n"
    elif len(selected_sheets_data_info) == 2:
        data_info1 = selected_sheets_data_info[0]
        data_info2 = selected_sheets_data_info[1]
        prompt_introduction += (
            f"Se ha proporcionado información para comparación de dos hojas: \n"
            f"1. Hoja '{data_info1['sheet_name']}' del archivo '{data_info1['file_name']}'.\n"
            f"2. Hoja '{data_info2['sheet_name']}' del archivo '{data_info2['file_name']}'.\n"
            f"Analiza y compara los datos de estas dos hojas según sea relevante para la pregunta:\n"
        )
    else: 
        prompt_introduction += "Analiza los siguientes datos de las hojas proporcionadas:\n"


    full_prompt_text_parts = [prompt_introduction] # Lista para construir el texto del prompt
    
    for data_info in selected_sheets_data_info: 
        file_name = data_info['file_name']
        sheet_name = data_info['sheet_name']
        file_sheet_key = data_info['file_sheet_key'] 

        sheet_details_text = f"\n=== Archivo: {file_name}, Hoja: {sheet_name} (Clave para gráficos: {file_sheet_key}) ===\n"
        sheet_details_text += f"- Filas Originales en la hoja: {data_info['rows_original']}\n"
        sheet_details_text += f"- Filas Procesadas para Análisis (usadas para la muestra JSON): {data_info['rows_processed']}\n"
        sheet_details_text += f"- Columnas ({data_info['columns_count']}): {', '.join(map(str, data_info['column_names']))}\n"
        sheet_details_text += f"- Tipos de datos: {json.dumps(data_info.get('data_types', {}))}\n"

        if "statistical_summary" in data_info:
            sheet_details_text += f"- Resumen Estadístico (columnas numéricas):\n{json.dumps(data_info['statistical_summary'], indent=2, ensure_ascii=False)}\n"
        
        for key, val_counts in data_info.items():
            if key.startswith("value_counts_"):
                col_name_vc = key.replace("value_counts_", "")
                sheet_details_text += f"- Conteo de Valores ('{col_name_vc}'):\n{json.dumps(val_counts, indent=2, ensure_ascii=False)}\n"

        sample_data_to_send = data_info['data_sample_json']
        try:
            sample_json_str = json.dumps(sample_data_to_send, indent=2, ensure_ascii=False)
            
            # MEJORA 3: Límite de longitud para la muestra JSON.
            # Este límite es para el string JSON, no para el número de filas directamente.
            MAX_SAMPLE_JSON_LEN = 70000 # Ajustar según necesidad y pruebas con Gemini
            
            if len(sample_json_str) > MAX_SAMPLE_JSON_LEN: 
                num_records_original_in_sample = len(sample_data_to_send) 
                
                # Truncar el string JSON para no exceder el límite.
                # Esto es un truncamiento simple, podría ser más inteligente (ej. tomar N primeros registros)
                # Aquí se truncan los registros para no exceder el límite de caracteres.
                # Podría ser más sofisticado, por ejemplo, tomando una muestra aleatoria o primeros y últimos.
                # La estimación de registros a mantener es aproximada.
                estimated_chars_per_record = len(sample_json_str) / num_records_original_in_sample if num_records_original_in_sample > 0 else 100 # Evitar división por cero
                num_records_to_keep_in_json = max(1, int(MAX_SAMPLE_JSON_LEN / estimated_chars_per_record)) if estimated_chars_per_record > 0 else 1
                
                sample_data_to_send_truncated = sample_data_to_send[:num_records_to_keep_in_json]
                sample_json_str = json.dumps(sample_data_to_send_truncated, indent=2, ensure_ascii=False)
                
                truncation_warning = (
                    f"Para la hoja '{sheet_name}', la representación JSON de las {num_records_original_in_sample} filas procesadas era demasiado extensa para el prompt. "
                    f"Se ha enviado una submuestra truncada de aproximadamente {len(sample_data_to_send_truncated)} registros para ajustarse a los límites. "
                    f"El análisis del LLM se basará en esta submuestra JSON."
                )
                # st.warning(truncation_warning) # Mostrar advertencia en la UI
                sheet_details_text += f"\n- Muestra de Datos (JSON, TRUNCADA a ~{len(sample_data_to_send_truncated)} de {num_records_original_in_sample} registros debido al tamaño):\n{sample_json_str}\n"
                sheet_details_text += f"  NOTA: {truncation_warning}\n"

            else:
                sheet_details_text += f"\n- Muestra de Datos (JSON, {len(sample_data_to_send)} registros de los {data_info['rows_processed']} procesados):\n{sample_json_str}\n"
        except Exception as e_json:
            sheet_details_text += f"\n- Muestra de Datos (JSON): [Error al serializar muestra: {e_json}]\n"
            
        full_prompt_text_parts.append(sheet_details_text)

    # Unir todas las partes del prompt en una sola cadena de texto
    final_user_prompt_text = "".join(full_prompt_text_parts)
    
    # La API de Gemini espera una lista de mensajes, pero para generate_content con system_instruction,
    # el prompt del usuario es una sola cadena.
    # La estructura de 'messages_for_llm' se mantiene por consistencia con la firma de query_gemini_api,
    # pero solo se usará el texto concatenado.
    messages_for_llm = [{"role": "user", "content": [{"type": "text", "text": final_user_prompt_text}]}]


    return query_gemini_api(api_key, model_name, system_prompt, messages_for_llm, max_output_tokens_config)


# --- Generación y Extracción de Gráficos ---
def extract_chart_suggestions(response_text):
    """Extrae las especificaciones de gráficos del JSON en la respuesta del LLM."""
    if not response_text: return [] # Manejar respuesta vacía
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```"
    matches = re.search(pattern, response_text)
    
    if not matches:
        return []
    
    try:
        json_str = matches.group(1).strip()
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str) # Corregir comas finales
        chart_specs = json.loads(json_str)
        if isinstance(chart_specs, dict): 
            chart_specs = [chart_specs]
        return chart_specs
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de sugerencias de gráficos: {e}")
        st.text_area("JSON con error (para depuración):", json_str, height=150)
        return []
    except Exception as e: 
        st.error(f"Error inesperado al extraer sugerencias de gráficos: {e}")
        st.text_area("Texto donde se buscó el JSON (para depuración):", response_text, height=150)
        return []

def generate_charts(chart_specs, all_dataframes_dict):
    """Genera figuras de Plotly basadas en las especificaciones del LLM."""
    charts = []
    if not isinstance(chart_specs, list): # Asegurar que chart_specs es una lista
        st.warning("Las especificaciones de gráficos no son una lista válida.")
        return []

    for spec_idx, spec in enumerate(chart_specs):
        if not isinstance(spec, dict): # Asegurar que cada spec es un diccionario
            st.warning(f"Especificación de gráfico {spec_idx+1} no es un diccionario válido. Saltando.")
            continue
        try:
            file_sheet_key = spec.get("file_and_sheet_key") 
            if not file_sheet_key or file_sheet_key not in all_dataframes_dict:
                st.warning(f"Gráfico {spec_idx+1} ('{spec.get('titulo', 'Desconocido')}'): Clave de archivo/hoja '{file_sheet_key}' no encontrada o no válida. Saltando.")
                continue
            
            df_source = all_dataframes_dict[file_sheet_key]
            if df_source is None or df_source.empty:
                st.warning(f"Gráfico {spec_idx+1} ('{spec.get('titulo', 'Desconocido')}'): DataFrame para '{file_sheet_key}' está vacío o no disponible. Saltando.")
                continue
            df = df_source.copy() 
            
            tipo = spec.get("tipo", "").lower()
            titulo = spec.get("titulo", f"Gráfico {tipo.capitalize()} {spec_idx+1}")
            eje_x = spec.get("eje_x")
            eje_y = spec.get("eje_y") 
            color_por = spec.get("color_por")
            agrupar_por_col_spec = spec.get("agrupar_por") # Puede ser string o lista
            operacion = spec.get("operacion", "sum")
            
            path_cols = spec.get("path")
            values_col = spec.get("values_col")
            names_col = spec.get("names_col") 
            dimensions_cols = spec.get("dimensions")

            def check_cols_exist(cols_to_check, df_columns, chart_title, col_purpose):
                if not cols_to_check: return True 
                if isinstance(cols_to_check, str): cols_to_check = [cols_to_check]
                # Filtrar None o strings vacíos de cols_to_check antes de la validación
                valid_cols_to_check = [col for col in cols_to_check if col and isinstance(col, str)]
                if not valid_cols_to_check: return True # Si después de filtrar no hay nada que chequear

                missing = [col for col in valid_cols_to_check if col not in df_columns]
                if missing:
                    #st.warning(f"Gráfico '{chart_title}': Columna(s) para {col_purpose} no encontradas: {', '.join(missing)}. Columnas disponibles: {', '.join(df_columns)}. Saltando.")
                    return False
                return True

            # Validaciones de columnas principales
            if not check_cols_exist(eje_x, df.columns, titulo, "Eje X"): continue
            if tipo not in ["histograma", "scatter_matrix", "heatmap", "sunburst", "treemap", "pastel", "funnel"] and \
               not check_cols_exist(eje_y, df.columns, titulo, "Eje Y"): continue
            
            if color_por and not check_cols_exist(color_por, df.columns, titulo, "Color Por"):
                color_por = None # Ignorar si no existe
            
            agrupar_por_col_list = []
            if agrupar_por_col_spec:
                if isinstance(agrupar_por_col_spec, str):
                    agrupar_por_col_list = [agrupar_por_col_spec]
                elif isinstance(agrupar_por_col_spec, list):
                    agrupar_por_col_list = agrupar_por_col_spec
                
                if not check_cols_exist(agrupar_por_col_list, df.columns, titulo, "Agrupar Por"):
                    agrupar_por_col_list = [] # No agrupar si hay error

            df_current_chart = df 
            eje_x_effective = eje_x
            eje_y_effective = eje_y

            if agrupar_por_col_list and eje_y: # Agrupar solo si hay columnas de agrupación y un eje Y
                # Asegurar que eje_y sea una sola columna string para la agregación estándar
                if isinstance(eje_y, list):
                    if len(eje_y) == 1 and isinstance(eje_y[0], str) and eje_y[0] in df_current_chart.columns:
                        col_para_agg = eje_y[0]
                    else:
                        st.warning(f"Gráfico '{titulo}': Agrupación con múltiples columnas Y no soportada directamente. Se intentará con la primera si es válida, o se omitirá la agrupación.")
                        col_para_agg = None # No se puede determinar una única columna para agregar
                elif isinstance(eje_y, str) and eje_y in df_current_chart.columns:
                    col_para_agg = eje_y
                else:
                    col_para_agg = None # Eje Y no es válido para agregación

                if col_para_agg and pd.api.types.is_numeric_dtype(df_current_chart[col_para_agg]):
                    try:
                        st.write(f"Agrupando datos para '{titulo}' por '{', '.join(agrupar_por_col_list)}', agregando '{col_para_agg}' con '{operacion}'.")
                        df_current_chart = df_current_chart.groupby(agrupar_por_col_list, as_index=False).agg({col_para_agg: operacion})
                        
                        # El eje X efectivo se convierte en las columnas de agrupación
                        if len(agrupar_por_col_list) == 1:
                             eje_x_effective = agrupar_por_col_list[0]
                        else:
                             # Si se agrupa por múltiples, Plotly puede manejarlos si el tipo de gráfico lo permite (ej. barras agrupadas)
                             # o el LLM debe ser más específico. Para ahora, se pasa la lista.
                             eje_x_effective = agrupar_por_col_list 
                        eje_y_effective = col_para_agg # El eje Y es la columna agregada
                    except Exception as e_agg:
                        st.warning(f"Gráfico '{titulo}': No se pudo agregar con '{operacion}' en '{col_para_agg}' agrupado por '{', '.join(agrupar_por_col_list)}': {e_agg}. Usando datos sin agregar.")
                elif col_para_agg: # Si hay col_para_agg pero no es numérica
                    st.warning(f"Gráfico '{titulo}': Columna Y '{col_para_agg}' no es numérica para la operación '{operacion}'. Usando datos sin agregar.")
            
            fig = None
            plot_args = {"title": titulo}
            if color_por: plot_args["color"] = color_por

            # Construcción de gráficos
            if tipo == "linea":
                fig = px.line(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "barra":
                fig = px.bar(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "dispersion":
                fig = px.scatter(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "pastel":
                if not check_cols_exist(names_col, df_current_chart.columns, titulo, "Nombres (names_col)") or \
                   not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.pie(df_current_chart, names=names_col, values=values_col, **plot_args)
            elif tipo == "caja":
                fig = px.box(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "histograma":
                fig = px.histogram(df_current_chart, x=eje_x_effective, **plot_args) # y no es necesario
            elif tipo == "area":
                fig = px.area(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "heatmap":
                numeric_df = df_current_chart.select_dtypes(include=np.number)
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    if not corr_matrix.empty:
                        fig = px.imshow(corr_matrix, title=f"Heatmap de Correlación - {titulo}", text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                    else: st.warning(f"Gráfico '{titulo}': Matriz de correlación vacía.")
                else: st.warning(f"Gráfico '{titulo}': No hay suficientes datos numéricos para un heatmap de correlación.")
            elif tipo == "sunburst":
                if not check_cols_exist(path_cols, df_current_chart.columns, titulo, "Ruta (path)") or \
                   not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.sunburst(df_current_chart, path=path_cols, values=values_col, **plot_args)
            elif tipo == "treemap":
                if not check_cols_exist(path_cols, df_current_chart.columns, titulo, "Ruta (path)") or \
                   not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.treemap(df_current_chart, path=path_cols, values=values_col, **plot_args)
            elif tipo == "funnel":
                if not check_cols_exist(names_col, df_current_chart.columns, titulo, "Etapas (names_col)") or \
                   not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.funnel(df_current_chart, y=names_col, x=values_col, **plot_args) # y=etapas, x=valores
            elif tipo == "violin":
                fig = px.violin(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "density_heatmap":
                # Asegurar que las columnas X e Y son numéricas
                valid_density_cols = True
                if isinstance(eje_x_effective, str) and not pd.api.types.is_numeric_dtype(df_current_chart[eje_x_effective]):
                    valid_density_cols = False
                if isinstance(eje_y_effective, str) and not pd.api.types.is_numeric_dtype(df_current_chart[eje_y_effective]): # y puede ser None
                    valid_density_cols = False
                
                if not valid_density_cols:
                    st.warning(f"Gráfico '{titulo}': Las columnas X e Y deben ser numéricas para density_heatmap. Saltando.")
                    continue
                fig = px.density_heatmap(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "scatter_matrix":
                cols_for_matrix = dimensions_cols
                if not cols_for_matrix: 
                    cols_for_matrix = df_current_chart.select_dtypes(include=np.number).columns.tolist()
                
                if not cols_for_matrix or len(cols_for_matrix) < 2:
                    st.warning(f"Gráfico '{titulo}': Se necesitan al menos 2 columnas numéricas para scatter_matrix. Saltando.")
                    continue
                if not check_cols_exist(cols_for_matrix, df_current_chart.columns, titulo, "Dimensiones (dimensions)"): continue
                
                plot_args_sm = {"title": titulo, "dimensions": cols_for_matrix}
                if color_por: plot_args_sm["color"] = color_por
                fig = px.scatter_matrix(df_current_chart, **plot_args_sm)
            
            if fig:
                fig.update_layout(title_x=0.5) 
                charts.append((fig, spec.get("descripcion", f"Gráfico interactivo tipo '{tipo}'. Pasa el cursor sobre los elementos para más detalles.")))
            # No mostrar advertencia si el tipo es heatmap y no se genera, ya que es común si no hay correlación
            elif tipo not in ["heatmap"]: 
                st.warning(f"Gráfico '{titulo}' (tipo '{tipo}'): No se pudo generar. Verifica la especificación del LLM o los datos.")
                
        except Exception as e:
            st.error(f"Error crítico al generar gráfico '{spec.get('titulo', 'Desconocido')}': {e}")
            import traceback
            st.text_area(f"Traceback del error del gráfico {spec_idx+1}:", traceback.format_exc(), height=150, key=f"error_trace_chart_{spec_idx}_{np.random.randint(1000)}") # Clave única
            continue
    return charts

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Analizador Contable Excel IA (Gemini)", page_icon="📊", layout="wide")

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    /* Estilo general para todas las listas de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px; /* Espacio entre pestañas reducido */
    }
    /* Estilo para cada pestaña individual */
    .stTabs [data-baseweb="tab"] {
        height: auto; /* Altura automática para acomodar texto largo */
        min-height: 40px; /* Altura mínima */
        white-space: normal; /* Permitir que el texto se ajuste */
        word-break: break-word; /* Romper palabras largas si es necesario */
        background-color: #F0F2F6; 
        border-radius: 4px 4px 0px 0px;
        padding: 8px 12px; /* Padding ajustado */
        color: #333333; 
        font-size: 0.9rem; /* Tamaño de fuente ligeramente reducido */
    }
    /* Estilo para la pestaña seleccionada */
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; 
        color: #000000; 
        border-bottom: 2px solid #1E88E5; /* Línea azul para la pestaña activa */
    }
    .stButton>button { /* Estilo para botones en general, si es necesario */
        border-radius: 6px;
    }
</style>""", unsafe_allow_html=True)


st.title("📊 Analizador Contable Multi-Excel con IA")
st.markdown("Carga uno o dos archivos Excel, selecciona las hojas a analizar/comparar, haz preguntas y obtén análisis, insights y visualizaciones.")

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiSx3wlcDkrgENNhc4ftlS_NWTMlygOx48nvQVGzdp2ib34Mu_JBnSZ2YVmkAj06eakuU&usqp=CAU" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### ⚙️ Configuración Principal")


    with st.expander("🔑 Configuración de API Gemini", expanded=True):
        api_key = st.text_input("Google Gemini API Key:", type="password", key="gemini_api_key", help="Obtén tu API key en [Google AI Studio](https://aistudio.google.com/app/apikey)")
        
        gemini_models_list = [
            "gemini-2.5-pro-preview-05-06",
            "gemini-1.5-pro-latest", 
            "gemini-1.5-flash-latest", 
            "gemini-pro", 
        ]
        model_name = st.selectbox(
            "Modelo:",
            gemini_models_list, 
            key="gemini_model",
            index=1, # Preseleccionar Flash por defecto (más rápido)
            help="Gemini 1.5 Pro y gemini-2.5-pro: Son los mas avanzados. Gemini 1.5 Flash: Rápido y eficiente. Gemini Pro: Modelo base sólido."
        )
        max_output_tokens = st.slider(
            "Max Tokens de Salida:", 
            256, 8192, 4096, 128, key="gemini_max_tokens", # Aumentado el mínimo y el paso
            help="Máximo de tokens que Gemini puede generar en su respuesta. Afecta la longitud y detalle de la respuesta. Modelos tienen límites diferentes."
        )

    st.subheader("📁 Carga de Archivos")
    uploaded_files = st.file_uploader(
        "Carga 1 o 2 archivos Excel (.xlsx, .xls)", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    st.subheader("🛠️ Opciones de Análisis")
    max_rows_per_sheet = st.slider(
        "Max Filas por Hoja (Muestra JSON):", 100, 10000, 1000, 100, key="max_rows_slider", 
        help="Limita el número de filas de cada hoja cuya muestra de datos se envía al LLM. Los gráficos también usarán estas filas."
    ) 
    include_statistics = st.checkbox("Incluir Resumen Estadístico en Prompt", value=True, key="include_stats_checkbox", help="Envía estadísticas descriptivas (media, D.E., etc.) de las filas procesadas al LLM.")
    generate_visualizations = st.checkbox("Sugerir Visualizaciones por IA", value=True, key="generate_viz_checkbox", help="Permite al LLM sugerir gráficos basados en tu pregunta y los datos.")
    
    if st.button("🧹 Limpiar Todo y Reiniciar", key="clear_all_button", type="secondary", use_container_width=True):
        for key_to_clear in list(st.session_state.keys()): # Iterar sobre una copia de las claves
            del st.session_state[key_to_clear]
        st.rerun()


# --- Lógica Principal de la Aplicación ---
# Inicialización de estados de sesión
if "processed_excel_data_info_full" not in st.session_state: 
    st.session_state.processed_excel_data_info_full = []
if "all_dfs_for_charts" not in st.session_state: 
    st.session_state.all_dfs_for_charts = {}
if "last_uploaded_files_names" not in st.session_state:
    st.session_state.last_uploaded_files_names = []
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "cleaned_llm_text" not in st.session_state:
    st.session_state.cleaned_llm_text = None
if "generated_charts" not in st.session_state:
    st.session_state.generated_charts = []
if "selected_sheet_info_1" not in st.session_state:
    st.session_state.selected_sheet_info_1 = None
if "selected_sheet_info_2" not in st.session_state:
    st.session_state.selected_sheet_info_2 = None


current_uploaded_files_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []

# Reprocesar archivos solo si han cambiado o si no hay datos procesados y hay archivos
if uploaded_files and (current_uploaded_files_names != st.session_state.get("last_uploaded_files_names", []) or not st.session_state.processed_excel_data_info_full):
    # Limpiar estados relacionados con datos anteriores
    st.session_state.processed_excel_data_info_full = [] 
    st.session_state.all_dfs_for_charts = {}
    st.session_state.llm_response = None 
    st.session_state.cleaned_llm_text = None
    st.session_state.generated_charts = []
    st.session_state.selected_sheet_info_1 = None 
    st.session_state.selected_sheet_info_2 = None 
    
    if len(uploaded_files) > 2:
        st.warning("Por favor, carga un máximo de dos archivos Excel. Se procesarán los dos primeros.")
        actual_files_to_process = uploaded_files[:2] 
    else:
        actual_files_to_process = uploaded_files

    with st.spinner(f"Procesando {len(actual_files_to_process)} archivo(s)... Por favor espera."):
        temp_processed_data_info_full = []
        temp_all_dfs_for_charts = {}
        for uploaded_file_item in actual_files_to_process:
            st.write(f"Cargando y analizando: {uploaded_file_item.name}...")
            try:
                excel_bytes_value = uploaded_file_item.getvalue() 
                file_data_list, dfs_charts_current_file = get_single_excel_data_info_cached(
                    excel_bytes_value, 
                    uploaded_file_item.name,
                    max_rows_per_sheet,
                    include_statistics
                )
                temp_processed_data_info_full.extend(file_data_list)
                temp_all_dfs_for_charts.update(dfs_charts_current_file)
                st.success(f"✅ Archivo '{uploaded_file_item.name}' procesado.")
            except Exception as e:
                st.error(f"Error al procesar el archivo {uploaded_file_item.name}: {e}")
                # status_process.update(label=f"Error procesando {uploaded_file_item.name}", state="error") # No hay status_process aquí
                continue 
        st.session_state.processed_excel_data_info_full = temp_processed_data_info_full
        st.session_state.all_dfs_for_charts = temp_all_dfs_for_charts
        st.session_state.last_uploaded_files_names = sorted([f.name for f in actual_files_to_process])
        # status_process.update(label="Procesamiento de archivos completado.", state="complete")


# --- Lógica de Selección de Hojas ---
selected_data_for_llm = []
sheet_selection_ui_completed = False # Para controlar si la UI de selección se mostró y es válida

if st.session_state.processed_excel_data_info_full:
    st.subheader("📄 Selección de Hojas y Vista Previa")
    
    files_data = {}
    for info in st.session_state.processed_excel_data_info_full:
        if info['file_name'] not in files_data:
            files_data[info['file_name']] = []
        files_data[info['file_name']].append(info)

    file_names_processed = list(files_data.keys())

    if len(file_names_processed) == 1:
        file_name_single = file_names_processed[0]
        sheets_in_file_single = files_data[file_name_single]
        # Crear etiquetas más informativas para el selectbox
        sheet_options_single_map = {f"{s['sheet_name']} ({s['rows_processed']} filas proc.)": s['file_sheet_key'] for s in sheets_in_file_single}
        
        if len(sheets_in_file_single) > 1:
            selected_sheet_display_single = st.selectbox(
                f"Selecciona la hoja a analizar del archivo '{file_name_single}':",
                options=list(sheet_options_single_map.keys()),
                key="sheet_select_single_display",
                index=0 
            )
            selected_sheet_key_single = sheet_options_single_map.get(selected_sheet_display_single)
            st.session_state.selected_sheet_info_1 = next((s for s in sheets_in_file_single if s['file_sheet_key'] == selected_sheet_key_single), None)
        elif sheets_in_file_single: # Solo una hoja en el archivo
            st.session_state.selected_sheet_info_1 = sheets_in_file_single[0]
            st.markdown(f"Analizando hoja: **{st.session_state.selected_sheet_info_1['sheet_name']}** del archivo **{file_name_single}**.")
        else: # No hay hojas en el archivo (raro, pero posible)
            st.warning(f"El archivo '{file_name_single}' no contiene hojas procesables.")

        if st.session_state.selected_sheet_info_1:
            selected_data_for_llm = [st.session_state.selected_sheet_info_1]
            sheet_selection_ui_completed = True

    elif len(file_names_processed) == 2:
        file_name1, file_name2 = file_names_processed[0], file_names_processed[1]
        sheets_in_file1 = files_data[file_name1]
        sheets_in_file2 = files_data[file_name2]

        sheet_options1_map = {f"{s['sheet_name']} ({s['rows_processed']} filas proc.)": s['file_sheet_key'] for s in sheets_in_file1}
        sheet_options2_map = {f"{s['sheet_name']} ({s['rows_processed']} filas proc.)": s['file_sheet_key'] for s in sheets_in_file2}

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            if sheets_in_file1:
                selected_sheet_display1 = st.selectbox(
                    f"Hoja del Archivo 1 ({file_name1}):",
                    options=list(sheet_options1_map.keys()),
                    key="sheet_select_file1_display",
                    index=0
                )
                selected_sheet_key1 = sheet_options1_map.get(selected_sheet_display1)
                st.session_state.selected_sheet_info_1 = next((s for s in sheets_in_file1 if s['file_sheet_key'] == selected_sheet_key1), None)
            else:
                st.warning(f"El archivo '{file_name1}' no contiene hojas procesables.")
                st.session_state.selected_sheet_info_1 = None


        with col_sel2:
            if sheets_in_file2:
                selected_sheet_display2 = st.selectbox(
                    f"Hoja del Archivo 2 ({file_name2}):",
                    options=list(sheet_options2_map.keys()),
                    key="sheet_select_file2_display",
                    index=0
                )
                selected_sheet_key2 = sheet_options2_map.get(selected_sheet_display2)
                st.session_state.selected_sheet_info_2 = next((s for s in sheets_in_file2 if s['file_sheet_key'] == selected_sheet_key2), None)
            else:
                st.warning(f"El archivo '{file_name2}' no contiene hojas procesables.")
                st.session_state.selected_sheet_info_2 = None

        if st.session_state.selected_sheet_info_1 and st.session_state.selected_sheet_info_2:
            selected_data_for_llm = [st.session_state.selected_sheet_info_1, st.session_state.selected_sheet_info_2]
            sheet_selection_ui_completed = True
            st.markdown(f"Comparando: **{st.session_state.selected_sheet_info_1['sheet_name']}** ({file_name1}) **VS** **{st.session_state.selected_sheet_info_2['sheet_name']}** ({file_name2})")
        elif not (st.session_state.selected_sheet_info_1 and st.session_state.selected_sheet_info_2):
            st.warning("Por favor, asegúrate de que ambos archivos tengan hojas seleccionables para la comparación.")


    # --- Vista Previa de Datos de Hojas Seleccionadas ---
    if selected_data_for_llm and sheet_selection_ui_completed: 
        st.markdown(f"Se han seleccionado **{len(selected_data_for_llm)}** hoja(s) para el análisis. A continuación, se muestran las primeras 10 filas de cada hoja seleccionada (limitadas por 'Max Filas por Hoja').")
        
        preview_tab_titles = []
        preview_data_info_list = []

        for data_info_preview in selected_data_for_llm:
            title = f"Vista: {data_info_preview['file_name']} - {data_info_preview['sheet_name']}"
            # Truncar títulos largos para pestañas
            title = title[:50] + '...' if len(title) > 50 else title
            preview_tab_titles.append(title)
            preview_data_info_list.append(data_info_preview)
        
        if preview_tab_titles:
            try:
                preview_tabs = st.tabs(preview_tab_titles)
                for i, tab_preview in enumerate(preview_tabs):
                    with tab_preview:
                        data_info_current = preview_data_info_list[i]
                        key_preview = data_info_current['file_sheet_key']
                        if key_preview in st.session_state.all_dfs_for_charts:
                            df_display = st.session_state.all_dfs_for_charts[key_preview] 
                            st.dataframe(df_display.head(10), use_container_width=True)
                            st.caption(f"Dimensiones originales de la hoja: {data_info_current['rows_original']} filas. Filas procesadas para análisis/JSON: {data_info_current['rows_processed']}.")
                        else:
                            st.warning(f"No se pudo cargar la vista previa para {key_preview}.")
            except Exception as e_tabs:
                st.error(f"Error al crear pestañas de vista previa: {e_tabs}.")
    
elif not uploaded_files:
    st.info("☝️ Por favor, carga uno o dos archivos Excel desde la barra lateral para comenzar.")


st.header("💬 Haz una Pregunta Sobre Tus Datos Seleccionados")
user_question = st.text_area(
    "Escribe tu pregunta aquí:", 
    height=100,
    key="user_question_input",
    placeholder="Ej: Si comparas dos hojas: 'Compara el total de ventas entre ambas'. Si analizas una hoja: '¿Cuál es el promedio de gastos por categoría?'"
)

if st.button("🚀 Analizar y Preguntar", type="primary", use_container_width=True, key="analyze_button"):
    if not api_key:
        st.error("Por favor, ingresa tu API key de Gemini en la barra lateral.")
    elif not user_question:
        st.warning("Por favor, escribe una pregunta.")
    elif not selected_data_for_llm or not sheet_selection_ui_completed: 
        st.warning("Por favor, carga archivos y asegúrate de que las hojas para análisis/comparación estén correctamente seleccionadas y la vista previa se muestre.")
    else:
        with st.spinner(f"Gemini ({model_name}) está analizando tus datos y generando una respuesta... Este proceso puede tardar unos momentos."):
            try:
                llm_response_text = query_llm(
                    api_key,
                    model_name,
                    selected_data_for_llm, 
                    user_question,
                    generate_visualizations,
                    max_output_tokens
                )
                st.session_state.llm_response = llm_response_text 
                
                cleaned_llm_text = clean_response_text(llm_response_text)
                st.session_state.cleaned_llm_text = cleaned_llm_text 
                
                st.session_state.generated_charts = [] # Limpiar gráficos anteriores
                if generate_visualizations and llm_response_text:
                    chart_specs_extracted = extract_chart_suggestions(llm_response_text)
                    if chart_specs_extracted:
                        generated_plotly_charts = generate_charts(chart_specs_extracted, st.session_state.all_dfs_for_charts)
                        st.session_state.generated_charts = generated_plotly_charts
            except Exception as e:
                st.error(f"Ocurrió un error crítico durante el análisis con Gemini: {str(e)}")
                import traceback
                st.exception(traceback.format_exc()) # Muestra el traceback completo en la UI
                st.session_state.llm_response = f"Error en la ejecución: {str(e)}" # Guardar el error para mostrarlo
                st.session_state.cleaned_llm_text = None
                st.session_state.generated_charts = []

# Mostrar resultados si existen
if st.session_state.get("llm_response"):
    st.divider()
    col_respuesta, col_graficos = st.columns([2,3]) 

    with col_respuesta:
        st.subheader("💡 Respuesta del Asistente IA (Gemini)")
        # Mostrar cleaned_llm_text si existe y no está vacío, sino mostrar llm_response (que podría ser un error)
        response_to_display = st.session_state.cleaned_llm_text if st.session_state.cleaned_llm_text else st.session_state.llm_response
        
        if response_to_display:
            st.markdown(response_to_display)
        elif st.session_state.llm_response: # Si cleaned es None pero llm_response tiene algo (ej. solo JSON de gráficos)
             st.info("La respuesta del LLM no contenía texto principal o solo sugerencias de gráficos. Revisa la respuesta completa.")
        else:
            st.info("No se generó texto de respuesta o hubo un error.")
        
        with st.expander("Ver respuesta completa de Gemini (incluye JSON de gráficos si existe)"):
            st.text_area("Respuesta Completa:", st.session_state.llm_response or "No hay respuesta completa disponible.", height=300, disabled=True, key="llm_full_response_area")

    with col_graficos:
        if generate_visualizations and st.session_state.get("generated_charts"):
            st.subheader("📈 Visualizaciones Sugeridas")
            st.markdown("Los gráficos son interactivos: puedes hacer zoom, moverte y obtener detalles al pasar el cursor.")
            
            generated_plotly_charts = st.session_state.generated_charts
            # Re-extraer specs solo para títulos, ya que los gráficos ya están generados
            chart_specs_for_titles = extract_chart_suggestions(st.session_state.llm_response or "") 

            if len(generated_plotly_charts) == 1:
                fig, desc = generated_plotly_charts[0]
                st.plotly_chart(fig, use_container_width=True)
                if desc: st.caption(f"**Descripción:** {desc}")
            elif len(generated_plotly_charts) > 1:
                chart_tab_titles_base = [
                    f"{idx+1}. {spec.get('titulo', 'Gráfico')[:35]}" # Acortar títulos para pestañas
                    if idx < len(chart_specs_for_titles) and isinstance(spec := chart_specs_for_titles[idx], dict) 
                    else f"Gráfico {idx+1}" 
                    for idx in range(len(generated_plotly_charts))
                ]
                
                # Asegurar unicidad de títulos de pestañas
                final_chart_tabs_titles = []
                title_counts = {}
                for title in chart_tab_titles_base:
                    clean_title = re.sub(r'[^\w\s-]', '', title).strip() or f"Gráfico_vacio_{len(final_chart_tabs_titles)}"
                    current_title = clean_title
                    count = title_counts.get(clean_title, 0) + 1
                    title_counts[clean_title] = count
                    if count > 1:
                        current_title = f"{clean_title} ({count})"
                    final_chart_tabs_titles.append(current_title)
                
                if final_chart_tabs_titles:
                    try:
                        chart_display_tabs = st.tabs(final_chart_tabs_titles)
                        for i, (fig, desc) in enumerate(generated_plotly_charts):
                            with chart_display_tabs[i]:
                                st.plotly_chart(fig, use_container_width=True)
                                if desc: st.caption(f"**Descripción:** {desc}")
                    except Exception as e_chart_tabs:
                        st.error(f"Error al crear pestañas de gráficos: {e_chart_tabs}. Mostrando gráficos secuencialmente.")
                        for fig_idx, (fig_item, desc_item) in enumerate(generated_plotly_charts):
                            title_fallback = final_chart_tabs_titles[fig_idx] if fig_idx < len(final_chart_tabs_titles) else f"Gráfico {fig_idx+1}"
                            st.subheader(title_fallback)
                            st.plotly_chart(fig_item, use_container_width=True)
                            if desc_item: st.caption(f"**Descripción:** {desc_item}")
                else: # Fallback si los títulos no se generaron bien
                     st.warning("No se pudieron generar títulos para las pestañas de gráficos. Mostrando secuencialmente.")
                     for fig_idx, (fig_item, desc_item) in enumerate(generated_plotly_charts):
                        st.subheader(f"Gráfico {fig_idx+1}")
                        st.plotly_chart(fig_item, use_container_width=True)
                        if desc_item: st.caption(f"**Descripción:** {desc_item}")

        elif generate_visualizations and "SUGERENCIAS_DE_VISUALIZACIÓN" in (st.session_state.llm_response or ""):
            # Este caso es si hubo sugerencias pero no se pudieron generar los gráficos
            with col_graficos: 
                st.subheader("📈 Visualizaciones Sugeridas")
                st.warning("Gemini intentó sugerir visualizaciones, pero no se pudieron extraer o generar correctamente. Revisa la 'Respuesta Completa de Gemini' para ver el JSON.")
        
        elif generate_visualizations: # Si la opción está activa pero no hay gráficos ni sugerencias
            with col_graficos:
                st.subheader("📈 Visualizaciones Sugeridas")
                st.info("Gemini no sugirió visualizaciones para esta pregunta, o la opción está desactivada, o no se pudieron generar.")


st.markdown("---")
st.markdown("""
### 📖 Guía Rápida de Uso:
1.  **🔑 Configura la API:** En la barra lateral, ingresa tu API key de Google Gemini y elige un modelo.
2.  **📁 Carga Archivos:** Sube uno o dos archivos Excel (`.xlsx` o `.xls`).
3.  **📄 Selecciona Hojas:**
    * **Si cargas 1 archivo:** Selecciona la hoja específica que deseas analizar (si hay más de una).
    * **Si cargas 2 archivos:** Selecciona una hoja de cada archivo para la comparación.
    * Aparecerá una vista previa de las primeras filas de las hojas seleccionadas.
4.  **🛠️ Ajusta Opciones (Opcional):**
    * **Max Filas por Hoja (Muestra JSON):** Controla cuántas filas de la muestra de cada hoja seleccionada se envían al LLM.
    * **Incluir Resumen Estadístico:** Envía estadísticas descriptivas al LLM para un mejor contexto.
    * **Sugerir Visualizaciones:** Permite que la IA sugiera gráficos relevantes.
5.  **💬 Haz tu Pregunta:** Escribe tu consulta sobre los datos de las hojas seleccionadas.
6.  **🚀 Analiza:** Presiona "Analizar y Preguntar". La respuesta y los gráficos (si se solicitaron) aparecerán abajo.

""")