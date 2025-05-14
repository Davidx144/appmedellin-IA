import streamlit as st
import pandas as pd
import anthropic
import openai # Importar OpenAI
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
    
    df_clean = df_clean.replace({np.nan: None}) # Reemplaza NaN con None para JSON
    
    try:
        records = df_clean.to_dict(orient="records")
        return convert_datetime_keys(records) # Asegura que todos los tipos internos sean seguros
    except Exception as e:
        st.warning(f"Error al convertir DataFrame a dict: {e}. Intentando conversión por filas.")
        processed_records = []
        for _, row in df_clean.iterrows():
            processed_records.append(convert_datetime_keys(row.to_dict()))
        return processed_records


# --- Procesamiento de Archivos Excel ---
def get_single_excel_data_info(excel_file_bytes, file_name, max_rows_limit, include_stats_option):
    """Procesa un solo archivo Excel y extrae información de sus hojas."""
    excel_data = pd.ExcelFile(excel_file_bytes)
    sheet_names = excel_data.sheet_names
    
    file_info_list = []
    dataframes_for_charts = {}

    st.write(f"Procesando archivo: **{file_name}** (Hojas encontradas: {', '.join(sheet_names)})")

    for sheet_name in sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet_name)
        original_rows = len(df)

        if len(df) > max_rows_limit:
            df = df.head(max_rows_limit)
            st.info(f"Archivo '{file_name}', Hoja '{sheet_name}': {original_rows} filas. Limitando a {max_rows_limit} filas para el análisis.")
        
        # Clave única para identificar el dataframe para gráficos
        file_sheet_key = f"{file_name}__{sheet_name}"
        dataframes_for_charts[file_sheet_key] = df.copy() 
        
        datos_completos_json_safe = process_dataframe_for_json(df)
        
        info = {
            "file_name": file_name,
            "sheet_name": sheet_name,
            "file_sheet_key": file_sheet_key, # Añadido para referencia directa
            "rows_original": original_rows,
            "rows_processed": len(df),
            "columns_count": len(df.columns),
            "column_names": convert_datetime_keys(list(df.columns)),
            "data_sample_json": datos_completos_json_safe
        }
        
        if include_stats_option:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                info["statistical_summary"] = convert_datetime_keys(df[numeric_cols].describe().to_dict())
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns # Incluir category
            for col in categorical_cols:
                # Umbral para conteo de valores, considerar también cardinalidad relativa
                if df[col].nunique() < 30 and df[col].nunique() > 0 : 
                    info[f"value_counts_{col}"] = convert_datetime_keys(df[col].value_counts().to_dict())
        
        info["data_types"] = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        
        file_info_list.append(info)
            
    return file_info_list, dataframes_for_charts

# --- Interacción con LLMs ---

def get_system_prompt(api_provider, generate_charts_flag):
    """Genera el system prompt adaptado para el LLM y la tarea."""
    chart_instructions = ""
    if generate_charts_flag:
        chart_instructions = """
SI LA PREGUNTA REQUIERE VISUALIZACIONES, TU RESPUESTA DEBE INCLUIR UNA SECCIÓN ESPECIAL LLAMADA "SUGERENCIAS_DE_VISUALIZACIÓN"
CON INSTRUCCIONES PRECISAS EN FORMATO JSON PARA CREAR HASTA 3-4 GRÁFICOS RELEVANTES.
Si se comparan dos archivos, puedes sugerir gráficos que muestren datos de ambos (usa la 'file_and_sheet_key' correcta para cada uno).
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
- `heatmap`: Usualmente para matrices de correlación (no necesita `eje_x`, `eje_y` explícitos) o tablas pivote. Si es correlación, indícalo.
- `sunburst`, `treemap`: Requieren `path` (lista de columnas para la jerarquía) y `values_col` (columna numérica para el tamaño).
- `funnel`: Requiere `values_col` (para los valores de cada etapa) y `names_col` (para los nombres de las etapas). `eje_x` y `eje_y` no se usan directamente.
- `density_heatmap`: Similar a `dispersion` pero para datos densos. `eje_x`, `eje_y` numéricos.
- `scatter_matrix`: `dimensions` es una lista de columnas para incluir en la matriz. Si no se especifica, se pueden usar todas las numéricas. `color_por` es útil aquí.

Asegúrate de que las columnas mencionadas existan en la 'file_and_sheet_key' especificada y sean del tipo adecuado. Prioriza la claridad y relevancia del gráfico para la pregunta del usuario.
"""

    base_prompt = f"""
Eres un asistente experto en análisis de datos de Excel, especialista en contabilidad y comparación de datos.
Se te proporcionará información estructurada (metadatos, nombres de columnas, tipos de datos, resumen estadístico y una muestra de datos en formato JSON) de una o dos hojas de cálculo de Excel.
Tu tarea es responder preguntas sobre los datos, compararlos si se proporcionan dos archivos, identificar tendencias, realizar cálculos y ofrecer insights.

{chart_instructions}

Analiza la información proporcionada de manera exhaustiva.
Si se proporcionan datos de dos archivos, enfócate en la comparación cuando sea relevante para la pregunta.
Estructura tu respuesta de forma clara y concisa. Utiliza Markdown para formatear tu respuesta (listas, negritas, etc.).
"""
    return base_prompt

def query_anthropic_api(api_key, model_name, system_prompt_text, messages_content, max_output_tokens):
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_output_tokens, 
            system=system_prompt_text,
            messages=messages_content
        )
        return response.content[0].text
    except Exception as e:
        return f"Error al consultar a Claude: {str(e)}"

def query_openai_api(api_key, model_name, system_prompt_text, messages_content_for_llm, max_output_tokens):
    client = openai.OpenAI(api_key=api_key)
    try:
        user_prompt_text = ""
        if messages_content_for_llm and messages_content_for_llm[0]['role'] == 'user':
            content_parts = messages_content_for_llm[0]['content']
            if isinstance(content_parts, list): 
                user_prompt_text = "\n".join([item["text"] for item in content_parts if item["type"] == "text"])
            elif isinstance(content_parts, str): 
                user_prompt_text = content_parts

        if not user_prompt_text:
            return "Error: No se pudo extraer el contenido del usuario para OpenAI."

        formatted_messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al consultar a OpenAI: {str(e)}"

def query_gemini_api(api_key, model_name, system_prompt_text, messages_content_for_llm, max_output_tokens):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt_text,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
        )
    )
    try:
        user_prompt_text = ""
        if messages_content_for_llm and messages_content_for_llm[0]['role'] == 'user':
            content_parts = messages_content_for_llm[0]['content']
            if isinstance(content_parts, list): 
                user_prompt_text = "\n".join([item["text"] for item in content_parts if item["type"] == "text"])
            elif isinstance(content_parts, str): 
                user_prompt_text = content_parts
        
        if not user_prompt_text:
            return "Error: No se pudo extraer el contenido del usuario para Gemini."

        response = model.generate_content(user_prompt_text) 
        return response.text
    except Exception as e:
        return f"Error al consultar a Gemini: {str(e)}"


def query_llm(api_provider, api_key, model_name, all_files_data_info, question, generate_charts_flag, max_output_tokens_config):
    """Prepara y envía la consulta al LLM seleccionado."""
    
    system_prompt = get_system_prompt(api_provider, generate_charts_flag)
    
    prompt_introduction = f"Pregunta del usuario: {question}\n\n"
    unique_files_processed = set(data_info['file_name'] for data_info in all_files_data_info)
    
    if len(unique_files_processed) == 1:
        prompt_introduction += f"Se ha proporcionado información del archivo Excel '{next(iter(unique_files_processed))}'. Analiza los siguientes datos de sus hojas:\n"
    elif len(unique_files_processed) > 1:
        prompt_introduction += f"Se ha proporcionado información de los archivos Excel: {', '.join(sorted(list(unique_files_processed)))}. Analiza y compara los datos de sus hojas según sea relevante:\n"
    else:
        prompt_introduction += "Analiza los siguientes datos:\n"


    content_for_llm_parts = [{"type": "text", "text": prompt_introduction}]
    
    for data_info in all_files_data_info: 
        file_name = data_info['file_name']
        sheet_name = data_info['sheet_name']
        file_sheet_key = data_info['file_sheet_key'] # Usar la clave ya generada

        sheet_details_text = f"\n=== Archivo: {file_name}, Hoja: {sheet_name} (Clave para gráficos: {file_sheet_key}) ===\n"
        sheet_details_text += f"- Filas Originales: {data_info['rows_original']}\n"
        sheet_details_text += f"- Filas Procesadas para Análisis: {data_info['rows_processed']}\n"
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
            # Límite de tamaño para la muestra de datos en el prompt
            # Ajustar este límite según sea necesario y la capacidad del modelo
            MAX_SAMPLE_JSON_LEN = 70000 
            if len(sample_json_str) > MAX_SAMPLE_JSON_LEN: 
                num_records_original = len(sample_data_to_send)
                # Reducción proporcional más inteligente o simplemente truncar primeros N registros
                # Aquí se truncan los registros para no exceder el límite de caracteres.
                # Podría ser más sofisticado, por ejemplo, tomando una muestra aleatoria o primeros y últimos.
                estimated_chars_per_record = len(sample_json_str) / num_records_original if num_records_original > 0 else 100
                num_records_to_keep = max(1, int(MAX_SAMPLE_JSON_LEN / estimated_chars_per_record)) if estimated_chars_per_record > 0 else 1
                
                sample_data_to_send_truncated = sample_data_to_send[:num_records_to_keep]
                sample_json_str = json.dumps(sample_data_to_send_truncated, indent=2, ensure_ascii=False)
                sheet_details_text += f"\n- Muestra de Datos (JSON, {len(sample_data_to_send_truncated)} de {num_records_original} registros para brevedad debido al tamaño):\n{sample_json_str}\n"
            else:
                sheet_details_text += f"\n- Muestra de Datos (JSON, {len(sample_data_to_send)} registros):\n{sample_json_str}\n"
        except Exception as e_json:
            sheet_details_text += f"\n- Muestra de Datos (JSON): [Error al serializar muestra: {e_json}]\n"
            
        content_for_llm_parts.append({"type": "text", "text": sheet_details_text})

    messages_for_llm = [{"role": "user", "content": content_for_llm_parts}]

    if api_provider == "Anthropic":
        return query_anthropic_api(api_key, model_name, system_prompt, messages_for_llm, max_output_tokens_config)
    elif api_provider == "OpenAI":
        return query_openai_api(api_key, model_name, system_prompt, messages_for_llm, max_output_tokens_config)
    elif api_provider == "Gemini":
        return query_gemini_api(api_key, model_name, system_prompt, messages_for_llm, max_output_tokens_config)
    else:
        return "Proveedor de API no válido seleccionado."


# --- Generación y Extracción de Gráficos ---
def extract_chart_suggestions(response_text):
    """Extrae las especificaciones de gráficos del JSON en la respuesta del LLM."""
    pattern = r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```"
    matches = re.search(pattern, response_text)
    
    if not matches:
        return []
    
    try:
        json_str = matches.group(1).strip()
        # Intento de corregir comas finales antes de } o ]
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str) 
        chart_specs = json.loads(json_str)
        if isinstance(chart_specs, dict): # Si el LLM devuelve un solo objeto en lugar de una lista
            chart_specs = [chart_specs]
        return chart_specs
    except json.JSONDecodeError as e:
        st.error(f"Error al decodificar JSON de sugerencias de gráficos: {e}")
        st.text_area("JSON con error (para depuración):", json_str, height=150)
        return []
    except Exception as e: # Captura otras excepciones inesperadas
        st.error(f"Error inesperado al extraer sugerencias de gráficos: {e}")
        st.text_area("Texto donde se buscó el JSON (para depuración):", response_text, height=150)
        return []

def generate_charts(chart_specs, all_dataframes_dict):
    """Genera figuras de Plotly basadas en las especificaciones del LLM."""
    charts = []
    for spec_idx, spec in enumerate(chart_specs):
        try:
            file_sheet_key = spec.get("file_and_sheet_key") 
            if not file_sheet_key or file_sheet_key not in all_dataframes_dict:
                st.warning(f"Gráfico {spec_idx+1} ('{spec.get('titulo', 'Desconocido')}'): Clave de archivo/hoja '{file_sheet_key}' no encontrada o no válida. Saltando.")
                continue
            
            df_source = all_dataframes_dict[file_sheet_key]
            if df_source is None or df_source.empty:
                 st.warning(f"Gráfico {spec_idx+1} ('{spec.get('titulo', 'Desconocido')}'): DataFrame para '{file_sheet_key}' está vacío o no disponible. Saltando.")
                 continue
            df = df_source.copy() # Usar una copia para manipulaciones
            
            tipo = spec.get("tipo", "").lower()
            titulo = spec.get("titulo", f"Gráfico {tipo.capitalize()} {spec_idx+1}")
            eje_x = spec.get("eje_x")
            eje_y = spec.get("eje_y") 
            color_por = spec.get("color_por")
            agrupar_por_col = spec.get("agrupar_por") 
            operacion = spec.get("operacion", "sum")
            
            # Parámetros específicos para nuevos tipos de gráficos
            path_cols = spec.get("path")
            values_col = spec.get("values_col")
            names_col = spec.get("names_col") # Para pastel, funnel
            dimensions_cols = spec.get("dimensions") # Para scatter_matrix

            # Validaciones básicas de columnas (se pueden expandir)
            def check_cols_exist(cols_to_check, df_columns, chart_title, col_purpose):
                if not cols_to_check: return True # No hay columnas que chequear
                if isinstance(cols_to_check, str): cols_to_check = [cols_to_check]
                missing = [col for col in cols_to_check if col not in df_columns]
                if missing:
                    st.warning(f"Gráfico '{chart_title}': Columna(s) para {col_purpose} no encontradas: {', '.join(missing)}. Saltando.")
                    return False
                return True

            # Validaciones generales
            if not check_cols_exist(eje_x, df.columns, titulo, "Eje X"): continue
            if tipo not in ["histograma", "scatter_matrix", "heatmap", "sunburst", "treemap", "pastel", "funnel"] and not check_cols_exist(eje_y, df.columns, titulo, "Eje Y"): continue
            if color_por and color_por not in df.columns:
                st.warning(f"Gráfico '{titulo}': Columna de color '{color_por}' no encontrada. Se ignora color_por.")
                color_por = None
            if agrupar_por_col and not all(col in df.columns for col in agrupar_por_col):
                st.warning(f"Gráfico '{titulo}': Columna de agrupación '{agrupar_por_col}' no encontrada. No se agrupará.")
                agrupar_por_col = None


            df_current_chart = df 
            eje_x_effective = eje_x
            eje_y_effective = eje_y

            # Aplicar agrupación si es especificado y válido
            if agrupar_por_col and eje_y and isinstance(eje_y, str) and eje_y in df_current_chart.columns:
                if pd.api.types.is_numeric_dtype(df_current_chart[eje_y]):
                    try:
                        st.write(f"Agrupando datos para '{titulo}' por '{agrupar_por_col}', agregando '{eje_y}' con '{operacion}'.")
                        df_current_chart = df_current_chart.groupby(agrupar_por_col, as_index=False).agg({eje_y: operacion})
                        eje_x_effective = agrupar_por_col 
                        eje_y_effective = eje_y          
                    except Exception as e_agg:
                        st.warning(f"Gráfico '{titulo}': No se pudo agregar con '{operacion}' en '{eje_y}' agrupado por '{agrupar_por_col}': {e_agg}. Usando datos sin agregar.")
                else:
                    st.warning(f"Gráfico '{titulo}': Columna Y '{eje_y}' no es numérica para la operación '{operacion}'. Usando datos sin agregar.")
            
            fig = None
            plot_args = {"title": titulo}
            if color_por: plot_args["color"] = color_por

            if tipo == "linea":
                fig = px.line(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "barra":
                fig = px.bar(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "dispersion":
                fig = px.scatter(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "pastel":
                if not check_cols_exist(names_col, df_current_chart.columns, titulo, "Nombres (names_col)"): continue
                if not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.pie(df_current_chart, names=names_col, values=values_col, **plot_args)
            elif tipo == "caja":
                fig = px.box(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "histograma":
                fig = px.histogram(df_current_chart, x=eje_x_effective, **plot_args)
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
                if not check_cols_exist(path_cols, df_current_chart.columns, titulo, "Ruta (path)"): continue
                if not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.sunburst(df_current_chart, path=path_cols, values=values_col, **plot_args)
            elif tipo == "treemap":
                if not check_cols_exist(path_cols, df_current_chart.columns, titulo, "Ruta (path)"): continue
                if not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                fig = px.treemap(df_current_chart, path=path_cols, values=values_col, **plot_args)
            elif tipo == "funnel":
                if not check_cols_exist(names_col, df_current_chart.columns, titulo, "Etapas (names_col)"): continue
                if not check_cols_exist(values_col, df_current_chart.columns, titulo, "Valores (values_col)"): continue
                # Para funnel, usualmente Y son las etapas (names_col) y X son los valores (values_col)
                fig = px.funnel(df_current_chart, y=names_col, x=values_col, **plot_args)
            elif tipo == "violin":
                fig = px.violin(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "density_heatmap":
                if not pd.api.types.is_numeric_dtype(df_current_chart[eje_x_effective]) or not pd.api.types.is_numeric_dtype(df_current_chart[eje_y_effective]):
                    st.warning(f"Gráfico '{titulo}': Las columnas X e Y deben ser numéricas para density_heatmap. Saltando.")
                    continue
                fig = px.density_heatmap(df_current_chart, x=eje_x_effective, y=eje_y_effective, **plot_args)
            elif tipo == "scatter_matrix":
                cols_for_matrix = dimensions_cols
                if not cols_for_matrix: # Si no se especifican dimensiones, usar todas las numéricas
                    cols_for_matrix = df_current_chart.select_dtypes(include=np.number).columns.tolist()
                
                if not cols_for_matrix or len(cols_for_matrix) < 2:
                    st.warning(f"Gráfico '{titulo}': Se necesitan al menos 2 columnas numéricas para scatter_matrix. Saltando.")
                    continue
                if not check_cols_exist(cols_for_matrix, df_current_chart.columns, titulo, "Dimensiones (dimensions)"): continue
                
                plot_args_sm = {"title": titulo, "dimensions": cols_for_matrix}
                if color_por: plot_args_sm["color"] = color_por
                fig = px.scatter_matrix(df_current_chart, **plot_args_sm)
            
            if fig:
                fig.update_layout(title_x=0.5) # Centrar título
                charts.append((fig, spec.get("descripcion", f"Gráfico interactivo tipo '{tipo}'. Pasa el cursor sobre los elementos para más detalles.")))
            elif tipo not in ["heatmap"]: # Heatmap puede no generarse si los datos no son adecuados, no siempre es un error.
                st.warning(f"Gráfico '{titulo}' (tipo '{tipo}'): No se pudo generar. Verifica la especificación del LLM o los datos.")
                
        except Exception as e:
            st.error(f"Error crítico al generar gráfico '{spec.get('titulo', 'Desconocido')}': {e}")
            import traceback
            st.text_area("Traceback del error del gráfico:", traceback.format_exc(), height=150, key="unique_key")
            continue
    return charts

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Analizador Contable Multi-Excel IA", page_icon="📊", layout="wide")

# --- Estilos CSS Personalizados (Opcional) ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
</style>""", unsafe_allow_html=True)


st.title("📊 Analizador Contable Multi-Excel con IA Avanzado")
st.markdown("Carga uno o dos archivos Excel, haz preguntas y obtén análisis comparativos, insights y visualizaciones dinámicas.")

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/0Z2Z7k9.png", width=100) # Ejemplo de logo, reemplazar con uno real o quitar
    st.header("⚙️ Configuración")

    with st.expander("🔑 Configuración de API y Modelo", expanded=True):
        api_provider = st.selectbox("Proveedor de API:", ["Anthropic", "OpenAI", "Gemini"], key="api_provider_select")
        
        api_key = ""
        model_name = ""
        max_output_tokens = 4000 

        if api_provider == "Anthropic":
            api_key = st.text_input("Anthropic API Key:", type="password", key="anthropic_api_key", help="Obtén tu API key en [Anthropic Console](https://console.anthropic.com/)")
            model_name = st.selectbox("Modelo de Claude:", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"], key="anthropic_model")
            max_output_tokens = st.slider("Max Tokens de Salida (Respuesta):", 200, 8000, 4000, 100, key="anthropic_max_tokens", help="Máximo de tokens que el modelo puede generar en su respuesta.")
        elif api_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key:", type="password", key="openai_api_key", help="Obtén tu API key en [OpenAI Platform](https://platform.openai.com/api-keys)")
            model_name = st.selectbox("Modelo de OpenAI:", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="openai_model")
            max_output_tokens = st.slider("Max Tokens de Salida (Respuesta):", 200, 8000, 4000, 100, key="openai_max_tokens")
        elif api_provider == "Gemini":
            api_key = st.text_input("Google Gemini API Key:", type="password", key="gemini_api_key", help="Obtén tu API key en [Google AI Studio](https://aistudio.google.com/app/apikey)")
            model_name = st.selectbox(
                "Modelo de Gemini:",
                ["gemini-2.5-pro-preview-05-06",
                 "gemini-2.0-flash",
                 "gemini-1.5-pro-latest",  # Más avanzado y multimodal
                 "gemini-1.5-flash-latest", # Más rápido y económico
                 "gemini-1.0-pro"],          # Generación anterior
                key="gemini_model",
                help="Gemini 1.5 Pro: el más avanzado y multimodal. Gemini 1.5 Flash: rápido y eficiente. Gemini 1.0 Pro: generación anterior."
            )
            max_output_tokens = st.slider("Max Tokens de Salida (Respuesta):", 200, 8192, 4000, 100, key="gemini_max_tokens")

    st.subheader("📁 Carga de Archivos")
    uploaded_files = st.file_uploader(
        "Carga 1 o 2 archivos Excel (.xlsx, .xls)", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    st.subheader("🛠️ Opciones de Análisis")
    max_rows_per_sheet = st.slider("Max Filas por Hoja a Analizar:", 100, 10000, 1000, 100, key="max_rows_slider", help="Limita el número de filas enviadas al LLM por cada hoja para controlar costos y tiempos de respuesta. Un valor más alto puede mejorar la precisión del análisis pero aumentar el costo y el tiempo.")
    include_statistics = st.checkbox("Incluir Resumen Estadístico en el Prompt", value=True, key="include_stats_checkbox", help="Envía estadísticas descriptivas (media, D.E., etc.) al LLM.")
    generate_visualizations = st.checkbox("Sugerir Visualizaciones por IA", value=True, key="generate_viz_checkbox", help="Permite al LLM sugerir gráficos basados en tu pregunta y los datos.")
    
    if st.button("Limpiar Todo y Reiniciar", key="clear_all_button"):
        st.session_state.clear() # Limpia todo el session_state
        st.rerun()


# --- Lógica Principal de la Aplicación ---
if "processed_excel_data_info" not in st.session_state:
    st.session_state.processed_excel_data_info = []
if "all_dfs_for_charts" not in st.session_state:
    st.session_state.all_dfs_for_charts = {}
if "last_uploaded_files_names" not in st.session_state:
    st.session_state.last_uploaded_files_names = []
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "generated_charts" not in st.session_state:
    st.session_state.generated_charts = []


current_uploaded_files_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []

# Reprocesar archivos solo si han cambiado o si no hay datos procesados y hay archivos
if uploaded_files and (current_uploaded_files_names != st.session_state.get("last_uploaded_files_names", []) or not st.session_state.processed_excel_data_info):
    st.session_state.processed_excel_data_info = [] 
    st.session_state.all_dfs_for_charts = {}
    st.session_state.llm_response = None # Limpiar respuesta anterior si los archivos cambian
    st.session_state.generated_charts = []
    
    if len(uploaded_files) > 2:
        st.warning("Por favor, carga un máximo de dos archivos Excel para comparación. Se procesarán los dos primeros.")
        actual_files_to_process = uploaded_files[:2] 
    else:
        actual_files_to_process = uploaded_files

    with st.status(f"Procesando {len(actual_files_to_process)} archivo(s)...", expanded=True) as status_process:
        for uploaded_file_item in actual_files_to_process:
            st.write(f"Cargando y analizando: {uploaded_file_item.name}...")
            try:
                excel_bytes = BytesIO(uploaded_file_item.getvalue())
                file_data_list, dfs_charts_current_file = get_single_excel_data_info(
                    excel_bytes, 
                    uploaded_file_item.name,
                    max_rows_per_sheet,
                    include_statistics
                )
                st.session_state.processed_excel_data_info.extend(file_data_list)
                st.session_state.all_dfs_for_charts.update(dfs_charts_current_file)
                st.write(f"✅ Archivo '{uploaded_file_item.name}' procesado.")
            except Exception as e:
                st.error(f"Error al procesar el archivo {uploaded_file_item.name}: {e}")
                status_process.update(label=f"Error procesando {uploaded_file_item.name}", state="error")
                continue 
        st.session_state.last_uploaded_files_names = sorted([f.name for f in actual_files_to_process])
        status_process.update(label="Procesamiento de archivos completado.", state="complete")


if st.session_state.processed_excel_data_info:
    st.subheader("📄 Vista Previa de Datos Procesados")
    st.markdown(f"Se han procesado **{len(st.session_state.processed_excel_data_info)}** hoja(s) de cálculo de **{len(st.session_state.last_uploaded_files_names)}** archivo(s). A continuación, se muestran las primeras 10 filas de cada hoja procesada.")

    tab_titles = []
    seen_titles = {}
    for data_info in st.session_state.processed_excel_data_info:
        base_title = f"{data_info['file_name']} - {data_info['sheet_name']}"
        if base_title in seen_titles:
            seen_titles[base_title] += 1
            title = f"{base_title} ({seen_titles[base_title]})"
        else:
            seen_titles[base_title] = 1
            title = base_title
        tab_titles.append(title)

    try:
        if tab_titles: # Solo crear pestañas si hay títulos
            preview_tabs = st.tabs(tab_titles)
            for i, tab_preview in enumerate(preview_tabs):
                with tab_preview:
                    data_info_current = st.session_state.processed_excel_data_info[i]
                    key = data_info_current['file_sheet_key']
                    if key in st.session_state.all_dfs_for_charts:
                        df_display = st.session_state.all_dfs_for_charts[key]
                        st.dataframe(df_display.head(10), use_container_width=True)
                        st.caption(f"Dimensiones originales: {data_info_current['rows_original']} filas x {data_info_current['columns_count']} columnas. Filas procesadas para análisis: {data_info_current['rows_processed']}.")
                    else:
                        st.warning(f"No se pudo cargar la vista previa para {key}.")
        else:
            st.info("No hay datos procesados para mostrar en la vista previa.")
            
    except Exception as e_tabs:
        st.error(f"Error al crear pestañas de vista previa: {e_tabs}. Puede ocurrir si los nombres de archivo/hoja son muy largos o contienen caracteres especiales.")

elif not uploaded_files:
    st.info("☝️ Por favor, carga uno o dos archivos Excel desde la barra lateral para comenzar.")


st.header("💬 Haz una Pregunta Sobre Tus Datos")
user_question = st.text_area(
    "Escribe tu pregunta aquí:", 
    height=120,
    key="user_question_input",
    placeholder="Ej: ¿Cuál es el total de ventas por región en el Archivo1? Compara los ingresos netos entre Archivo1 y Archivo2. ¿Qué tendencias observas en los gastos del Archivo1? ¿Puedes mostrarme un desglose de categorías de productos en un gráfico de sunburst?"
)

if st.button("🚀 Analizar y Preguntar al LLM", type="primary", use_container_width=True, key="analyze_button"):
    if not api_key:
        st.error(f"Por favor, ingresa tu API key de {api_provider} en la barra lateral.")
    elif not user_question:
        st.warning("Por favor, escribe una pregunta.")
    elif not st.session_state.processed_excel_data_info:
        st.warning("Por favor, carga y procesa al menos un archivo Excel. Asegúrate de que los archivos se hayan cargado y procesado correctamente (revisa mensajes anteriores o recarga los archivos).")
    else:
        with st.spinner(f"{api_provider} ({model_name}) está analizando tus datos y generando una respuesta... Este proceso puede tardar unos momentos."):
            try:
                llm_response_text = query_llm(
                    api_provider,
                    api_key,
                    model_name,
                    st.session_state.processed_excel_data_info, 
                    user_question,
                    generate_visualizations,
                    max_output_tokens
                )
                st.session_state.llm_response = llm_response_text # Guardar respuesta completa
                
                cleaned_llm_text = clean_response_text(llm_response_text)
                st.session_state.cleaned_llm_text = cleaned_llm_text # Guardar texto limpio
                
                if generate_visualizations:
                    chart_specs_extracted = extract_chart_suggestions(llm_response_text)
                    if chart_specs_extracted:
                        generated_plotly_charts = generate_charts(chart_specs_extracted, st.session_state.all_dfs_for_charts)
                        st.session_state.generated_charts = generated_plotly_charts
                    else:
                        st.session_state.generated_charts = [] # Limpiar si no hay nuevas
                else:
                    st.session_state.generated_charts = []


            except Exception as e:
                st.error(f"Ocurrió un error crítico durante el análisis con el LLM: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())
                st.session_state.llm_response = None
                st.session_state.cleaned_llm_text = None
                st.session_state.generated_charts = []

# Mostrar resultados si existen
if st.session_state.get("llm_response"):
    st.divider()
    col_respuesta, col_graficos = st.columns([2,3]) # Ajustar proporción según necesidad

    with col_respuesta:
        st.subheader("💡 Respuesta del Asistente IA")
        if st.session_state.get("cleaned_llm_text"):
            st.markdown(st.session_state.cleaned_llm_text)
        else:
            st.info("No se generó texto de respuesta o hubo un error.")
        
        with st.expander("Ver respuesta completa del LLM (incluye JSON de gráficos si existe)"):
            st.text_area("Respuesta Completa:", st.session_state.llm_response, height=300, disabled=True)

    with col_graficos:
        if generate_visualizations and st.session_state.get("generated_charts"):
            st.subheader("📈 Visualizaciones Sugeridas")
            st.markdown("Los gráficos son interactivos: puedes hacer zoom, moverte y obtener detalles al pasar el cursor.")
            
            generated_plotly_charts = st.session_state.generated_charts
            chart_specs_extracted = extract_chart_suggestions(st.session_state.llm_response) # Re-extraer para títulos

            if len(generated_plotly_charts) == 1:
                fig, desc = generated_plotly_charts[0]
                st.plotly_chart(fig, use_container_width=True)
                if desc: st.caption(f"**Descripción:** {desc}")
            elif len(generated_plotly_charts) > 1:
                # Generar títulos para las pestañas de gráficos
                chart_tab_titles_base = [
                    f"Gráfico {i+1}: {chart_specs_extracted[i].get('titulo', 'Visualización')[:30]}" 
                    if i < len(chart_specs_extracted) else f"Gráfico {i+1}" 
                    for i in range(len(generated_plotly_charts))
                ]
                
                # Asegurar unicidad de títulos de pestañas para gráficos
                unique_chart_tab_titles = []
                title_counts = {}
                for title in chart_tab_titles_base:
                    if title in title_counts:
                        title_counts[title] += 1
                        unique_chart_tab_titles.append(f"{title} ({title_counts[title]})")
                    else:
                        title_counts[title] = 1
                        unique_chart_tab_titles.append(title)

                chart_display_tabs = st.tabs(unique_chart_tab_titles)
                for i, (fig, desc) in enumerate(generated_plotly_charts):
                    with chart_display_tabs[i]:
                        st.plotly_chart(fig, use_container_width=True)
                        if desc: st.caption(f"**Descripción:** {desc}")
            else:
                 st.info("No se pudieron generar las visualizaciones sugeridas o no se sugirieron visualizaciones válidas para esta pregunta.")
        
        elif generate_visualizations and "SUGERENCIAS_DE_VISUALIZACIÓN" in st.session_state.llm_response:
             with col_graficos: # Asegurar que se muestre en la columna de gráficos
                st.subheader("📈 Visualizaciones Sugeridas")
                st.warning("El LLM intentó sugerir visualizaciones, pero no se pudieron extraer o interpretar correctamente. Revisa la 'Respuesta Completa del LLM' en la columna de la izquierda para ver el JSON.")
        
        elif generate_visualizations:
            with col_graficos:
                st.subheader("📈 Visualizaciones Sugeridas")
                st.info("El LLM no sugirió visualizaciones para esta pregunta, o la opción está desactivada.")


st.markdown("---")
st.markdown("""
### 📖 Guía Rápida de Uso:
1.  **🔑 Configura la API:** En la barra lateral, selecciona tu proveedor de IA (Anthropic, OpenAI, Gemini), ingresa tu API key y elige un modelo.
2.  **📁 Carga Archivos:** Sube uno o dos archivos Excel (`.xlsx` o `.xls`). La aplicación procesará todas las hojas de cada archivo.
3.  **🛠️ Ajusta Opciones (Opcional):**
    * **Max Filas por Hoja:** Controla cuántas filas por hoja se envían al LLM. Menos filas = más rápido y barato, pero podría ser menos preciso.
    * **Incluir Resumen Estadístico:** Envía estadísticas descriptivas al LLM para un mejor contexto.
    * **Sugerir Visualizaciones:** Permite que la IA sugiera gráficos relevantes.
4.  **📄 Vista Previa:** Revisa las primeras filas de tus datos en las pestañas de vista previa que aparecerán después de procesar los archivos.
5.  **💬 Haz tu Pregunta:** Escribe tu consulta sobre los datos. Si cargaste dos archivos, puedes pedir comparaciones o análisis combinados. Sé específico para mejores resultados.
6.  **🚀 Analiza:** Presiona "Analizar y Preguntar al LLM". La respuesta y los gráficos (si se solicitaron y generaron) aparecerán abajo.

**Nota sobre Privacidad y Datos:** Esta aplicación envía metadatos (nombres de columnas, tipos de datos), resúmenes estadísticos (si se seleccionan) y una muestra de las filas (limitada por "Max Filas por Hoja") de tus archivos Excel al servicio de LLM que hayas configurado. **Los archivos completos no se almacenan permanentemente en servidores de terceros por esta aplicación.** El procesamiento ocurre en la memoria de la instancia de Streamlit y durante la comunicación con el API del LLM. Sé consciente de la sensibilidad de tus datos al usar servicios de IA externos.
""")

