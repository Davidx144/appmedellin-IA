import streamlit as st
import pandas as pd
# import anthropic # Eliminado
# import openai # Eliminado
import google.generativeai as genai # Importar motor AppIA
import os
import json
import tempfile
from io import BytesIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Para gráficos más complejos si es necesario
import re
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de tokens máximos para AppIA
APPIA_MAX_INPUT_TOKENS = 2097152  # Máximo para modelos AppIA avanzados
APPIA_MAX_OUTPUT_TOKENS = 8192    # Máximo de salida

# Configuración predeterminada
DEFAULT_MODEL = "gemini-1.5-pro-latest"  # Modelo estable y disponible
DEFAULT_MAX_ROWS = 20000  # Análisis más completo por defecto
DEFAULT_INCLUDE_STATS = True  # Siempre incluir análisis estadístico
DEFAULT_GENERATE_CHARTS = True  # Siempre generar gráficos automáticamente

# --- Configuración Inicial y Funciones de Ayuda ---

def clean_response_text(response_text):
    """Elimina la sección de sugerencias de visualización del texto de respuesta de manera robusta."""
    if not response_text:
        return ""
    
    # Múltiples patrones para capturar diferentes formatos de JSON de visualización
    patterns = [
        r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```",
        r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```\s*([\s\S]*?)\s*```",
        r"```json\s*\[([\s\S]*?)\]\s*```",  # JSON directo en código
        r"```\s*\[([\s\S]*?)\]\s*```",     # JSON sin especificar lenguaje
    ]
    
    clean_text = response_text
    for pattern in patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)
    
    # Limpiar líneas vacías múltiples
    clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)
    
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

def fix_dataframe_for_display(df):
    """Corrige tipos de datos problemáticos para Arrow/Streamlit."""
    df_fixed = df.copy()
    
    # Corregir todas las columnas 'Unnamed' que pueden tener tipos mezclados
    for col in df_fixed.columns:
        if str(col).startswith('Unnamed:') or df_fixed[col].dtype == 'object':
            # Convertir a string para evitar tipos mezclados
            df_fixed[col] = df_fixed[col].astype(str)
    
    # Convertir columnas datetime problemáticas
    for col in df_fixed.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime', 'datetimetz']).columns:
        df_fixed[col] = df_fixed[col].astype(str)
    
    # Manejar valores NaN/None
    df_fixed = df_fixed.fillna('')
    
    return df_fixed


def fix_dataframe_for_plotting(df):
    """Prepara un DataFrame específicamente para gráficos de Plotly."""
    df_plot = df.copy()
    
    # Para gráficos, necesitamos mantener tipos numéricos cuando sea posible
    for col in df_plot.columns:
        if str(col).startswith('Unnamed:'):
            # Si la columna Unnamed contiene solo números, mantenerla como numérica
            try:
                # Intentar convertir a numérico, pero solo si la mayoría de valores son números
                numeric_col = pd.to_numeric(df_plot[col], errors='coerce')
                if numeric_col.notna().sum() > len(df_plot) * 0.8:  # 80% de valores válidos
                    df_plot[col] = numeric_col.fillna(0)
                else:
                    df_plot[col] = df_plot[col].astype(str)
            except:
                df_plot[col] = df_plot[col].astype(str)
        elif df_plot[col].dtype == 'object':
            # Para columnas de objeto, intentar mantener como están si no causan problemas
            try:
                # Si contiene valores mixtos, convertir a string
                if df_plot[col].apply(type).nunique() > 1:
                    df_plot[col] = df_plot[col].astype(str)
            except:
                df_plot[col] = df_plot[col].astype(str)
    
    # Convertir columnas datetime a string para gráficos
    for col in df_plot.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime', 'datetimetz']).columns:
        df_plot[col] = df_plot[col].astype(str)
    
    # Manejar valores NaN/None
    df_plot = df_plot.fillna(0)  # Para gráficos, usamos 0 en lugar de string vacío
    
    return df_plot


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
        try:
            df = pd.read_excel(excel_data, sheet_name=sheet_name)
            
            # Aplicar correcciones tempranas para evitar problemas con tipos de datos
            # Esto previene errores de Arrow más adelante
            for col in df.columns:
                if str(col).startswith('Unnamed:'):
                    # Las columnas Unnamed suelen tener problemas de tipos mixtos
                    df[col] = df[col].astype(str)
                elif df[col].dtype == 'object':
                    # Verificar si hay tipos mixtos en columnas de objeto
                    try:
                        unique_types = df[col].dropna().apply(type).nunique()
                        if unique_types > 1:
                            df[col] = df[col].astype(str)
                    except:
                        df[col] = df[col].astype(str)
            
            original_rows = len(df)
        except Exception as e:
            st.error(f"Error al procesar la hoja '{sheet_name}' del archivo '{file_name}': {e}")
            continue

        # MEJORA 3: Asegurar que se limita el df ANTES de procesarlo para JSON
        if len(df) > max_rows_limit:
            df_limited = df.head(max_rows_limit).copy() 
            # st.info(f"Archivo '{file_name}', Hoja '{sheet_name}': {original_rows} filas. Limitando a {max_rows_limit} filas para el análisis.") # Mensaje en UI
        else:
            df_limited = df.copy()
        
        file_sheet_key = f"{file_name}__{sheet_name}"
        # Guardar el DataFrame (limitado por max_rows_limit) para gráficos y vista previa
        # Aplicar correcciones de tipos de datos antes de guardar
        df_for_storage = fix_dataframe_for_display(df_limited)
        dataframes_for_charts[file_sheet_key] = df_for_storage.copy() 
        
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
    """Genera el system prompt adaptado para AppIA y la tarea."""
    chart_instructions = ""
    if generate_charts_flag:
        chart_instructions = """
## 📈 INSTRUCCIONES PARA VISUALIZACIONES (CRÍTICO):

**OBLIGATORIO:** Si la pregunta puede beneficiarse de visualizaciones, DEBES incluir la sección "SUGERENCIAS_DE_VISUALIZACIÓN".

### REGLAS PARA GENERAR GRÁFICOS EXITOSOS:

1. **VALIDACIÓN DE DATOS:**
   - VERIFICA que las columnas mencionadas existan EXACTAMENTE en los datos proporcionados
   - Para columnas numéricas, asegúrate de que contengan valores numéricos válidos
   - Para fechas, verifica que estén en formato de fecha reconocible
   - Si una columna no existe o está vacía, NO la incluyas en el gráfico

2. **SELECCIÓN INTELIGENTE DE GRÁFICOS:**
   - **Barras/Líneas:** Para comparar valores numéricos entre categorías o a lo largo del tiempo
   - **Pastel:** Solo para distribuciones porcentuales con pocas categorías (máximo 7-8)
   - **Dispersión:** Para mostrar correlaciones entre dos variables numéricas
   - **Histograma:** Para distribución de una variable numérica
   - **Cajas:** Para análisis de distribución y outliers

3. **COMPARACIÓN ENTRE DOCUMENTOS:**
   - Si hay dos archivos, crea gráficos que muestren ambos datasets para comparación
   - Usa la misma escala y tipo de gráfico para facilitar la comparación
   - Incluye gráficos que combinen datos de ambos archivos cuando sea relevante

**FORMATO EXACTO REQUERIDO:**

SUGERENCIAS_DE_VISUALIZACIÓN:
```json
[
  {
    "tipo": "barra | linea | dispersion | pastel | caja | histograma | heatmap | area",
    "titulo": "Título específico y descriptivo del gráfico",
    "file_and_sheet_key": "NombreArchivoExacto__NombreHojaExacta",
    "eje_x": "NombreExactoColumnaX",
    "eje_y": "NombreExactoColumnaY",
    "color_por": "NombreColumnaParaColorear (opcional)",
    "agrupar_por": "NombreColumnaParaAgrupar (opcional)",
    "operacion": "sum|mean|count|max|min (solo si hay agrupación)",
    "names_col": "ColumnaDeNombres (solo para pastel)",
    "values_col": "ColumnaDeValores (solo para pastel)",
    "descripcion": "Explicación detallada de lo que muestra este gráfico y por qué es relevante para el análisis."
  }
]
```

**CRÍTICO:** Solo usa nombres de columnas que aparezcan EXACTAMENTE en los datos JSON proporcionados. No inventes nombres de columnas.
"""

    base_prompt = f"""
Eres un asistente experto en análisis de datos de Excel, especialista en contabilidad, finanzas y comparación inteligente de documentos.
Se te proporcionará información estructurada (metadatos, nombres de columnas, tipos de datos, resumen estadístico y una muestra de datos en formato JSON) de una o múltiples hojas de cálculo de Excel previamente seleccionadas por el usuario.

## RESPONSABILIDADES PRINCIPALES:

### 🏦 ANÁLISIS ESPECÍFICO DE DATOS BANCARIOS (CRÍTICO):
**IMPORTANTE:** Para datos bancarios, SIEMPRE procesa cada banco/cuenta/entidad de manera INDIVIDUAL:
- **NUNCA agrupes bancos diferentes** aunque tengan nombres similares (ej: Bancolombia Cuenta-A vs Bancolombia Cuenta-B son DIFERENTES)
- **Identifica cada banco por su nombre COMPLETO** incluyendo números de cuenta, sucursales, o cualquier identificador único
- **Analiza cada entidad bancaria por separado** con sus propios totales, promedios, y tendencias
- **Reporta diferencias entre bancos/cuentas** de manera individual y específica
- **Lista todos los bancos encontrados** con sus características únicas antes del análisis
- **Ejemplos de entidades DIFERENTES que NO se deben agrupar:**
  * Bancolombia-1234 vs Bancolombia-5678 (cuentas diferentes)
  * Banco Popular Principal vs Banco Popular Sucursal Norte
  * BBVA Empresarial vs BBVA Personal
  * Cualquier variación en nombre, número, o denominación

### 📊 ANÁLISIS INDIVIDUAL (1 documento):
- Proporciona análisis **EXTENSO y DETALLADO** (mínimo 300-500 palabras)
- Examina tendencias, patrones y anomalías específicas
- Identifica insights clave y recomendaciones concretas
- Incluye cálculos relevantes y contexto contable/financiero

### 🔄 ANÁLISIS COMPARATIVO AUTOMÁTICO (2+ documentos):
**IMPORTANTE: Cuando recibas datos de DOS hojas/archivos, SIEMPRE realiza lo siguiente:**

1. **IDENTIFICACIÓN AUTOMÁTICA DE CAMPOS SIMILARES:**
   - Busca columnas con nombres similares, equivalentes o relacionados entre ambos documentos
   - Identifica campos que representen conceptos similares (fechas, montos, códigos, categorías, etc.)
   - Detecta relaciones temáticas incluso si los nombres de columnas difieren
   - Ejemplo: "Valor Debito" en un documento vs "Débitos" en otro = mismo concepto

2. **COMPARACIÓN ESCALABLE E INTELIGENTE:**
   - Compara automáticamente valores totales, promedios, rangos y distribuciones
   - Identifica diferencias significativas y sus posibles causas
   - Analiza evoluciones temporales si hay datos de fechas
   - Busca correlaciones y dependencias entre los datasets
   - Detecta inconsistencias, duplicados o datos faltantes

3. **MAPEO DE RELACIONES:**
   - Determina si los documentos son complementarios, secuenciales o independientes
   - Identifica si uno es detalle del otro (ej: transacciones vs balance)
   - Busca puntos de reconciliación entre ambos documentos

4. **SÍNTESIS INTELIGENTE:**
   - Proporciona conclusiones integradas que aprovechen información de ambos documentos
   - Genera insights que solo son posibles con la vista combinada
   - Sugiere acciones basadas en el análisis conjunto

{chart_instructions}

## FORMATO DE RESPUESTA REQUERIDO:
- **EXTENSO:** Mínimo 500-800 palabras para análisis completo
- **ESTRUCTURADO:** Usa títulos, subtítulos y listas claras
- **INSIGHTS PROFUNDOS:** No te limites a describir datos, proporciona interpretaciones y recomendaciones
- **FORMATO MARKDOWN:** Utiliza negritas, listas, tablas y estructura jerárquica
- **CONTEXTO PROFESIONAL:** Enfoque contable/financiero cuando sea aplicable

Analiza la información proporcionada de manera exhaustiva y proporciona el nivel de detalle que esperaría un profesional contable o financiero.
"""
    return base_prompt

def query_appia_api(api_key, model_name, system_prompt_text, messages_content_for_llm, max_output_tokens):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt_text,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.7  # Configuración óptima para análisis de datos
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
            return "Error: No se pudo extraer el contenido del usuario para AppIA."

        response = model.generate_content(user_prompt_text) 
        return response.text
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg:
             return f"Error al consultar a AppIA: Se ha excedido la cuota de la API. Por favor, revisa tu plan y uso en Google AI Studio. (Detalle: {str(e)})"
        elif "api key not valid" in error_msg:
            return f"Error al consultar a AppIA: La API key no es válida. Por favor, verifica tu API Key. (Detalle: {str(e)})"
        elif "not found" in error_msg or "not supported" in error_msg:
            return f"Error al consultar a AppIA: El modelo '{model_name}' no está disponible. Esto puede deberse a que el modelo no existe o no está disponible en tu región. (Detalle: {str(e)})"
        elif "response.prompt_feedback" in error_msg and "block_reason" in error_msg:
            return f"Error al consultar a AppIA: La solicitud fue bloqueada por políticas de seguridad. (Detalle: {str(e)})"
        else:
            return f"Error al consultar a AppIA: {str(e)}"


def query_llm(selected_sheets_data_info, question, generate_charts_flag=True): 
    """Prepara y envía la consulta al LLM AppIA."""
    
    # Obtener API key desde variables de entorno
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if not api_key:
        return "Error: No se encontró la API key de AppIA. Por favor, configura GOOGLE_GEMINI_API_KEY en el archivo .env"
    
    # Lista de modelos en orden de preferencia
    models_to_try = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest", 
        "gemini-pro"
    ]
    
    max_output_tokens_config = APPIA_MAX_OUTPUT_TOKENS
    
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

    # Intentar con diferentes modelos en orden de preferencia
    last_error = None
    for model_name in models_to_try:
        try:
            result = query_appia_api(api_key, model_name, system_prompt, messages_for_llm, max_output_tokens_config)
            # Si no hay error en el resultado, retornarlo
            if not result.startswith("Error al consultar a Gemini:"):
                return result
            else:
                last_error = result
                continue
        except Exception as e:
            last_error = f"Error con modelo {model_name}: {str(e)}"
            continue
    
    # Si llegamos aquí, ningún modelo funcionó
    return last_error or "Error: No se pudo conectar con ningún modelo de Gemini disponible."


# --- Generación y Extracción de Gráficos ---
def extract_chart_suggestions(response_text):
    """Extrae las especificaciones de gráficos del JSON en la respuesta del LLM de manera robusta."""
    if not response_text: 
        return []
    
    # Múltiples patrones para detectar JSON de gráficos en diferentes formatos
    patterns = [
        r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```json\s*([\s\S]*?)\s*```",
        r"SUGERENCIAS_DE_VISUALIZACIÓN:[\s]*```\s*([\s\S]*?)\s*```", 
        r"```json\s*(\[[\s\S]*?\])\s*```",  # JSON directo
        r"```\s*(\[[\s\S]*?\])\s*```",     # JSON sin especificar lenguaje
        r"(\[[\s]*\{[\s\S]*?\}[\s]*\])",   # JSON sin marcadores de código
    ]
    
    json_str = None
    matched_pattern = None
    
    for i, pattern in enumerate(patterns):
        matches = re.search(pattern, response_text, re.IGNORECASE)
        if matches:
            json_str = matches.group(1).strip()
            matched_pattern = i + 1
            st.info(f"🔍 JSON encontrado usando patrón {matched_pattern}")
            break
    
    if not json_str:
        # Buscar cualquier array JSON que contenga objetos con 'tipo' y 'titulo'
        json_array_pattern = r'(\[[\s\S]*?"tipo"[\s\S]*?"titulo"[\s\S]*?\])'
        matches = re.search(json_array_pattern, response_text, re.IGNORECASE)
        if matches:
            json_str = matches.group(1).strip()
            st.info("🔍 JSON encontrado mediante búsqueda de estructura de gráficos")
    
    if not json_str:
        st.warning("⚠️ No se encontraron sugerencias de visualización en la respuesta")
        return []
    
    try:
        # Limpiar el JSON antes de parsearlo
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)  # Eliminar comas finales
        json_str = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1\2', json_str)  # Unir strings divididos
              
        chart_specs = json.loads(json_str)
        
        if isinstance(chart_specs, dict): 
            chart_specs = [chart_specs]
        
        st.success(f"✅ Se extrajeron {len(chart_specs)} especificaciones de gráficos")
        return chart_specs
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al decodificar JSON de gráficos: {e}")
        st.text_area("JSON con error:", json_str, height=200, key="error_json_debug")
        
        # Intentar reparar JSON común
        try:
            # Reparaciones comunes
            fixed_json = json_str
            fixed_json = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1\2', fixed_json)
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
            
            chart_specs = json.loads(fixed_json)
            if isinstance(chart_specs, dict): 
                chart_specs = [chart_specs]
            st.success(f"✅ JSON reparado exitosamente. {len(chart_specs)} gráficos encontrados")
            return chart_specs
        except:
            st.error("❌ No se pudo reparar el JSON automáticamente")
            return []
        
    except Exception as e: 
        st.error(f"❌ Error inesperado al extraer gráficos: {e}")
        return []

def generate_charts(chart_specs, all_dataframes_dict):
    """Genera figuras de Plotly basadas en las especificaciones del LLM con validación robusta."""
    charts = []
    if not isinstance(chart_specs, list):
        st.warning("⚠️ Las especificaciones de gráficos no son una lista válida.")
        return []

    st.info(f"🎯 Procesando {len(chart_specs)} sugerencia(s) de gráfico...")

    for spec_idx, spec in enumerate(chart_specs):
        if not isinstance(spec, dict):
            st.warning(f"❌ Gráfico {spec_idx+1}: Especificación no válida (no es diccionario). Saltando.")
            continue
            
        # Validación mejorada con información detallada
        try:
            file_sheet_key = spec.get("file_and_sheet_key") 
            titulo = spec.get("titulo", f"Gráfico {spec_idx+1}")
            tipo = spec.get("tipo", "").lower()
            
            # Validar archivo/hoja
            if not file_sheet_key:
                st.error(f"❌ Gráfico '{titulo}': Falta 'file_and_sheet_key'. Saltando.")
                continue
                
            if file_sheet_key not in all_dataframes_dict:
                st.error(f"❌ Gráfico '{titulo}': Archivo/hoja '{file_sheet_key}' no encontrado. Disponibles: {list(all_dataframes_dict.keys())}")
                continue
            
            df_source = all_dataframes_dict[file_sheet_key]
            if df_source is None or df_source.empty:
                st.error(f"❌ Gráfico '{titulo}': DataFrame vacío para '{file_sheet_key}'.")
                continue
            
            # Preparar DataFrame específicamente para gráficos
            df = fix_dataframe_for_plotting(df_source.copy())
            
            # Mostrar información del dataset para depuración
            st.info(f"📊 Procesando gráfico '{titulo}' (tipo: {tipo}) con dataset de {len(df)} filas y {len(df.columns)} columnas.")
            
            # Validar tipo de gráfico
            tipos_validos = ["barra", "linea", "dispersion", "pastel", "caja", "histograma", "heatmap", "area"]
            if tipo not in tipos_validos:
                st.warning(f"⚠️ Gráfico '{titulo}': Tipo '{tipo}' no reconocido. Tipos válidos: {tipos_validos}. Intentando continuar...")
            
            eje_x = spec.get("eje_x")
            eje_y = spec.get("eje_y") 
            color_por = spec.get("color_por")
            agrupar_por_col_spec = spec.get("agrupar_por")
            operacion = spec.get("operacion", "sum")
            
            values_col = spec.get("values_col")
            names_col = spec.get("names_col")
            path_cols = spec.get("path")
            dimensions_cols = spec.get("dimensions")

            def check_cols_exist(cols_to_check, df_columns, chart_title, col_purpose):
                """Valida que las columnas especificadas existan en el DataFrame con retroalimentación detallada."""
                if not cols_to_check: 
                    return True 
                if isinstance(cols_to_check, str): 
                    cols_to_check = [cols_to_check]
                
                # Filtrar None o strings vacíos
                valid_cols_to_check = [col for col in cols_to_check if col and isinstance(col, str)]
                if not valid_cols_to_check: 
                    return True

                missing = [col for col in valid_cols_to_check if col not in df_columns]
                if missing:
                    st.error(f"❌ Gráfico '{chart_title}': Columna(s) para {col_purpose} no encontradas: {missing}")
                    st.info(f"📋 Columnas disponibles en el dataset: {list(df_columns)[:10]}{'...' if len(df_columns) > 10 else ''}")
                    return False
                return True

            # Mostrar información sobre las columnas del dataset
            st.info(f"🔍 Dataset '{file_sheet_key}' contiene columnas: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")

            # Validaciones de columnas principales con información detallada
            if not check_cols_exist(eje_x, df.columns, titulo, "Eje X"): 
                continue
            if tipo not in ["histograma", "heatmap", "pastel"] and not check_cols_exist(eje_y, df.columns, titulo, "Eje Y"): 
                continue
            
            if color_por and not check_cols_exist(color_por, df.columns, titulo, "Color Por"):
                st.warning(f"⚠️ Gráfico '{titulo}': Columna de color '{color_por}' no encontrada. Continuando sin colorear.")
                color_por = None
            
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
st.set_page_config(page_title="Analizador Contable Excel con AppIA", page_icon="📊", layout="wide")

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


st.title("📊 Analizador de Excel con AppIA")
st.markdown("**Carga tus archivos Excel y haz preguntas en lenguaje natural. AppIA analizará los datos y generará respuestas con gráficos automáticamente.**")

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
    st.markdown("### ⚙️ Configuración")

    st.subheader("📁 Carga de Archivos")
    uploaded_files = st.file_uploader(
        "Carga de 1 a 5 archivos Excel (.xlsx, .xls)", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if st.button("🧹 Limpiar Todo", key="clear_all_button", type="secondary", use_container_width=True):
        for key_to_clear in list(st.session_state.keys()): # Iterar sobre una copia de las claves
            del st.session_state[key_to_clear]
        st.rerun()
    
    # Verificar configuración de API
    api_key_status = os.getenv('GOOGLE_GEMINI_API_KEY')
    if api_key_status:
        st.success("✅ API configurada correctamente")
    else:
        st.error("❌ API no configurada. Revisa el archivo .env")

    # Mostrar configuración optimizada (solo informativo)
    with st.expander("ℹ️ Configuración del análisis"):
        st.markdown(f"""
        **Configuración optimizada automáticamente:**
        - 📊 Filas máximas por hoja: **{DEFAULT_MAX_ROWS:,}**
        - 📈 Gráficos automáticos: **Activado**
        - 📋 Análisis estadístico: **Activado**
        - 🤖 Modelo de IA: **AppIA Pro (mejor disponible)**
        
        *Estos ajustes garantizan el mejor análisis posible.*
        """)


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
    
    # Permitir hasta 5 archivos para análisis empresarial robusto
    max_files_allowed = 5
    if len(uploaded_files) > max_files_allowed:
        st.warning(f"Por favor, carga un máximo de {max_files_allowed} archivos Excel. Se procesarán los primeros {max_files_allowed}.")
        actual_files_to_process = uploaded_files[:max_files_allowed] 
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
                    DEFAULT_MAX_ROWS,
                    DEFAULT_INCLUDE_STATS
                )
                temp_processed_data_info_full.extend(file_data_list)
                temp_all_dfs_for_charts.update(dfs_charts_current_file)
                st.success(f"✅ Archivo '{uploaded_file_item.name}' procesado correctamente.")
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo {uploaded_file_item.name}: {e}")
                # status_process.update(label=f"Error procesando {uploaded_file_item.name}", state="error") # No hay status_process aquí
                continue 
        
        # Mostrar información sobre correcciones automáticas
        if temp_processed_data_info_full:
            st.info("ℹ️ Se aplicaron correcciones automáticas de tipos de datos para garantizar compatibilidad.")
            
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

    elif len(file_names_processed) >= 2:
        # Interfaz para múltiples documentos (2-5 archivos)
        st.markdown(f"### 🗂️ **Configuración de Análisis Multi-Documento ({len(file_names_processed)} archivos)**")
        st.info("💡 **AppIA analizará automáticamente las relaciones entre todos los documentos seleccionados.**")
        
        # Crear una lista para almacenar todas las hojas seleccionadas
        selected_sheets_info = []
        
        # Crear columnas dinámicamente basadas en el número de archivos
        if len(file_names_processed) <= 3:
            cols = st.columns(len(file_names_processed))
        else:
            # Para más de 3 archivos, usar filas múltiples
            cols_per_row = 3
            rows_needed = (len(file_names_processed) + cols_per_row - 1) // cols_per_row
            
        sheet_selection_widgets = []
        
        for idx, file_name in enumerate(file_names_processed):
            sheets_in_file = files_data[file_name]
            sheet_options_map = {f"{s['sheet_name']} ({s['rows_processed']} filas proc.)": s['file_sheet_key'] for s in sheets_in_file}
            
            # Determinar en qué columna colocar el widget
            if len(file_names_processed) <= 3:
                col = cols[idx]
            else:
                # Para layouts más complejos, usar contenedores secuenciales
                if idx % 3 == 0:
                    col_container = st.container()
                    current_cols = col_container.columns(min(3, len(file_names_processed) - idx))
                col = current_cols[idx % 3]
            
            with col:
                st.markdown(f"#### 📄 **Documento {chr(65 + idx)}**")
                st.caption(f"**{file_name}**")
                
                if sheets_in_file:
                    selected_sheet_display = st.selectbox(
                        f"Hoja de {file_name[:20]}...",
                        options=list(sheet_options_map.keys()),
                        key=f"sheet_select_file{idx}_display",
                        index=0,
                        help=f"Selecciona la hoja del documento {chr(65 + idx)} para el análisis"
                    )
                    selected_sheet_key = sheet_options_map.get(selected_sheet_display)
                    selected_sheet_info = next((s for s in sheets_in_file if s['file_sheet_key'] == selected_sheet_key), None)
                    
                    if selected_sheet_info:
                        selected_sheets_info.append(selected_sheet_info)
                        st.success(f"✅ Seleccionado")
                    else:
                        st.error(f"❌ Error en selección")
                else:
                    st.warning(f"❌ Sin hojas procesables")
        
        # Verificar que todas las selecciones sean válidas
        if len(selected_sheets_info) == len(file_names_processed) and len(selected_sheets_info) >= 2:
            selected_data_for_llm = selected_sheets_info
            sheet_selection_ui_completed = True
            
            # Mensaje de confirmación para análisis multi-documento
            st.success("✅ **Análisis Multi-Documento Configurado Exitosamente**")
            
            documents_list = "\n".join([
                f"- **Documento {chr(65 + i)}:** `{info['sheet_name']}` de **{info['file_name']}**" 
                for i, info in enumerate(selected_sheets_info)
            ])
            
            st.markdown(f"""
            **📊 Análisis Multi-Documento:**
            {documents_list}
            
            🤖 **AppIA realizará automáticamente:**
            - Identificación de campos similares entre todos los documentos
            - Análisis de correlaciones y dependencias cruzadas
            - Detección de patrones comunes y diferencias significativas
            - Mapeo de relaciones financieras/contables entre archivos
            - Síntesis integrada con insights únicos del conjunto completo
            """)
        else:
            missing_count = len(file_names_processed) - len(selected_sheets_info)
            st.warning(f"⚠️ **Configuración incompleta:** Faltan {missing_count} selecciones válidas para completar el análisis multi-documento.")
    
    else:
        st.info("📁 No hay archivos cargados o hay un problema con el procesamiento.")


    # --- Vista Previa de Datos de Hojas Seleccionadas ---
    if selected_data_for_llm and sheet_selection_ui_completed: 
        st.markdown(f"Se han seleccionado **{len(selected_data_for_llm)}** hoja(s) para el análisis. A continuación, se muestran las primeras 10 filas de cada hoja seleccionada (limitadas por 'Max Filas por Hoja').")
        
        preview_tab_titles = []
        preview_data_info_list = []

        for data_info_preview in selected_data_for_llm:
            # Validar y limpiar los nombres de archivo y hoja
            file_name = str(data_info_preview.get('file_name', 'Archivo')).strip()
            sheet_name = str(data_info_preview.get('sheet_name', 'Hoja')).strip()
            
            # Asegurar que no estén vacíos
            if not file_name or file_name == 'None':
                file_name = f"Archivo_{len(preview_tab_titles)+1}"
            if not sheet_name or sheet_name == 'None':
                sheet_name = f"Hoja_{len(preview_tab_titles)+1}"
            
            title = f"Vista: {file_name} - {sheet_name}"
            # Truncar títulos largos para pestañas
            title = title[:50] + '...' if len(title) > 50 else title
            preview_tab_titles.append(title)
            preview_data_info_list.append(data_info_preview)
        
        if preview_tab_titles:
            try:
                # Validación adicional de títulos antes de crear pestañas
                valid_titles = []
                for i, title in enumerate(preview_tab_titles):
                    if isinstance(title, str) and len(title.strip()) > 0:
                        valid_titles.append(title.strip())
                    else:
                        valid_titles.append(f"Vista {i+1}")
                
                # Asegurar que no hay títulos duplicados
                final_titles = []
                seen_titles = {}
                for title in valid_titles:
                    if title in seen_titles:
                        seen_titles[title] += 1
                        final_titles.append(f"{title} ({seen_titles[title]})")
                    else:
                        seen_titles[title] = 1
                        final_titles.append(title)
                
                preview_tabs = st.tabs(final_titles)
                for i, tab_preview in enumerate(preview_tabs):
                    with tab_preview:
                        if i < len(preview_data_info_list):
                            data_info_current = preview_data_info_list[i]
                            key_preview = data_info_current['file_sheet_key']
                            if key_preview in st.session_state.all_dfs_for_charts:
                                df_display = st.session_state.all_dfs_for_charts[key_preview]
                                # Corregir tipos de datos problemáticos para Arrow/Streamlit
                                df_display_fixed = fix_dataframe_for_display(df_display)
                                st.dataframe(df_display_fixed.head(10), width='stretch')
                                st.caption(f"Dimensiones originales de la hoja: {data_info_current['rows_original']} filas. Filas procesadas para análisis/JSON: {data_info_current['rows_processed']}.")
                            else:
                                st.warning(f"No se pudo cargar la vista previa para {key_preview}.")
                        else:
                            st.warning("Datos no disponibles para esta pestaña.")
            except Exception as e_tabs:
                st.error(f"Error al crear pestañas de vista previa: {str(e_tabs)}")
                st.error(f"Debug - Títulos: {preview_tab_titles}")
                st.error(f"Debug - Tipos: {[type(t).__name__ for t in preview_tab_titles]}")
                # Fallback: mostrar datos sin pestañas
                st.subheader("Vista previa de datos (sin pestañas)")
                for i, data_info_current in enumerate(preview_data_info_list):
                    file_name_display = str(data_info_current.get('file_name', f'Archivo_{i+1}'))
                    sheet_name_display = str(data_info_current.get('sheet_name', f'Hoja_{i+1}'))
                    st.markdown(f"**{file_name_display} - {sheet_name_display}**")
                    key_preview = data_info_current['file_sheet_key']
                    if key_preview in st.session_state.all_dfs_for_charts:
                        df_display = st.session_state.all_dfs_for_charts[key_preview]
                        df_display_fixed = fix_dataframe_for_display(df_display)
                        st.dataframe(df_display_fixed.head(10), width='stretch')
                    else:
                        st.warning(f"No se pudo cargar la vista previa para {key_preview}")
                    st.divider()
    
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
    # Verificar configuración de API
    api_key_check = os.getenv('GOOGLE_GEMINI_API_KEY')
    if not api_key_check:
        st.error("❌ No se encontró la API key de AppIA. Por favor, configura GOOGLE_GEMINI_API_KEY en el archivo .env")
    elif not user_question:
        st.warning("Por favor, escribe una pregunta.")
    elif not selected_data_for_llm or not sheet_selection_ui_completed: 
        st.warning("Por favor, carga archivos y asegúrate de que las hojas para análisis/comparación estén correctamente seleccionadas y la vista previa se muestre.")
    else:
        # Crear contenedor para estado de espera que cubra toda la pantalla
        progress_container = st.empty()
        
        with progress_container.container():
            # Pantalla de espera completa
            st.markdown("""
            <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                        background-color: rgba(255, 255, 255, 0.95); z-index: 9999; 
                        display: flex; flex-direction: column; justify-content: center; 
                        align-items: center; backdrop-filter: blur(3px);">
                <div style="text-align: center; background: white; padding: 3rem; 
                           border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
                           border: 1px solid rgba(255,255,255,0.2);">
                    <h1 style="color: #1E88E5; margin-bottom: 2rem;">🤖 AppIA está procesando</h1>
                    <div style="font-size: 1.2rem; color: #666; margin-bottom: 2rem; line-height: 1.6;">
                        🔍 <strong>Analizando datos...</strong><br>
                        📊 <strong>Identificando patrones...</strong><br>
                        📈 <strong>Generando visualizaciones...</strong><br>
                        💡 <strong>Creando insights profesionales...</strong>
                    </div>
                    <div style="color: #888; font-style: italic;">
                        Este proceso puede tardar entre 30 segundos a 2 minutos<br>
                        dependiendo de la complejidad de los datos
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            llm_response_text = query_llm(
                selected_data_for_llm, 
                user_question,
                DEFAULT_GENERATE_CHARTS
            )
            st.session_state.llm_response = llm_response_text 
            
            cleaned_llm_text = clean_response_text(llm_response_text)
            st.session_state.cleaned_llm_text = cleaned_llm_text 
            
            st.session_state.generated_charts = [] # Limpiar gráficos anteriores
            if DEFAULT_GENERATE_CHARTS and llm_response_text:
                chart_specs_extracted = extract_chart_suggestions(llm_response_text)
                if chart_specs_extracted:
                    generated_plotly_charts = generate_charts(chart_specs_extracted, st.session_state.all_dfs_for_charts)
                    st.session_state.generated_charts = generated_plotly_charts
                    
            # Limpiar la pantalla de espera
            progress_container.empty()
            
            # Forzar rerun para mostrar resultados
            st.rerun()
            
        except Exception as e:
            # Limpiar la pantalla de espera en caso de error
            progress_container.empty()
            
            st.error(f"Ocurrió un error crítico durante el análisis con AppIA: {str(e)}")
            import traceback
            st.exception(traceback.format_exc()) # Muestra el traceback completo en la UI
            st.session_state.llm_response = f"Error en la ejecución: {str(e)}" # Guardar el error para mostrarlo
            st.session_state.cleaned_llm_text = None
            st.session_state.generated_charts = []

# Mostrar resultados si existen
if st.session_state.get("llm_response"):
    st.divider()
    
    # Reorganizar la interfaz para respuestas extensas
    st.markdown("## 📊 **Resultados del Análisis de IA**")
    
    # Crear pestañas para mejor organización del contenido expandido
    tab_analisis, tab_graficos, tab_detalles = st.tabs(["📝 **Análisis Detallado**", "📈 **Visualizaciones**", "🔍 **Información Técnica**"])
    
    with tab_analisis:
        st.markdown("### 💡 **Respuesta del Asistente AppIA**")
        
        # Mostrar cleaned_llm_text si existe y no está vacío, sino mostrar llm_response (que podría ser un error)
        response_to_display = st.session_state.cleaned_llm_text if st.session_state.cleaned_llm_text else st.session_state.llm_response
        
        if response_to_display:
            # Contenedor expandible para respuestas largas
            with st.container():
                st.markdown(response_to_display)
        elif st.session_state.llm_response: # Si cleaned es None pero llm_response tiene algo (ej. solo JSON de gráficos)
             st.info("ℹ️ La respuesta del modelo contenía principalmente sugerencias de gráficos. Revisa la pestaña 'Información Técnica' para ver la respuesta completa.")
        else:
            st.warning("⚠️ No se generó texto de respuesta o hubo un error en el proceso.")
    
    with tab_graficos:
        if DEFAULT_GENERATE_CHARTS and st.session_state.get("generated_charts"):
            st.subheader("📈 Visualizaciones Sugeridas")
            st.markdown("Los gráficos son interactivos: puedes hacer zoom, moverte y obtener detalles al pasar el cursor.")
            
            generated_plotly_charts = st.session_state.generated_charts
            # Re-extraer specs solo para títulos, ya que los gráficos ya están generados
            chart_specs_for_titles = extract_chart_suggestions(st.session_state.llm_response or "") 

            if len(generated_plotly_charts) == 1:
                fig, desc = generated_plotly_charts[0]
                st.plotly_chart(fig, width='stretch')
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
                                st.plotly_chart(fig, width='stretch')
                                if desc: st.caption(f"**Descripción:** {desc}")
                    except Exception as e_chart_tabs:
                        st.error(f"Error al crear pestañas de gráficos: {e_chart_tabs}. Mostrando gráficos secuencialmente.")
                        for fig_idx, (fig_item, desc_item) in enumerate(generated_plotly_charts):
                            title_fallback = final_chart_tabs_titles[fig_idx] if fig_idx < len(final_chart_tabs_titles) else f"Gráfico {fig_idx+1}"
                            st.subheader(title_fallback)
                            st.plotly_chart(fig_item, width='stretch')
                            if desc_item: st.caption(f"**Descripción:** {desc_item}")
                else: # Fallback si los títulos no se generaron bien
                     st.warning("No se pudieron generar títulos para las pestañas de gráficos. Mostrando secuencialmente.")
                     for fig_idx, (fig_item, desc_item) in enumerate(generated_plotly_charts):
                        st.subheader(f"Gráfico {fig_idx+1}")
                        st.plotly_chart(fig_item, width='stretch')
                        if desc_item: st.caption(f"**Descripción:** {desc_item}")

        elif DEFAULT_GENERATE_CHARTS and "SUGERENCIAS_DE_VISUALIZACIÓN" in (st.session_state.llm_response or ""):
            # Este caso es si hubo sugerencias pero no se pudieron generar los gráficos
            st.subheader("📈 Visualizaciones Sugeridas")
            st.warning("⚠️ AppIA intentó sugerir visualizaciones, pero no se pudieron extraer o generar correctamente. Revisa la pestaña 'Información Técnica' para ver el JSON completo.")
        
        elif DEFAULT_GENERATE_CHARTS: # Si la opción está activa pero no hay gráficos ni sugerencias
            st.subheader("📈 Visualizaciones")
            st.info("ℹ️ AppIA no sugirió visualizaciones específicas para esta consulta, o no se pudieron generar gráficos con los datos disponibles.")
    
    with tab_detalles:
        st.markdown("### 🔍 **Información Técnica**")
        
        # Información sobre el procesamiento
        if st.session_state.get("processed_excel_data_info_full"):
            st.markdown("#### � **Datos Procesados:**")
            for info in st.session_state.processed_excel_data_info_full:
                st.markdown(f"- **{info['file_name']}** → Hoja: `{info['sheet_name']}` ({info['rows_processed']} filas procesadas)")
        
        # Respuesta completa del modelo
        with st.expander("🤖 Ver respuesta completa de AppIA (incluye JSON de gráficos si existe)"):
            st.text_area(
                "Respuesta Completa:", 
                st.session_state.llm_response or "No hay respuesta completa disponible.", 
                height=400, 
                disabled=True, 
                key="llm_full_response_area"
            )
        
        # Información de configuración
        with st.expander("⚙️ Configuración del análisis"):
            st.markdown(f"""
            **Configuración utilizada:**
            - 📊 Filas máximas procesadas: **{DEFAULT_MAX_ROWS:,}**
            - 📈 Gráficos automáticos: **{'Activado' if DEFAULT_GENERATE_CHARTS else 'Desactivado'}**
            - 📋 Análisis estadístico: **{'Incluido' if DEFAULT_INCLUDE_STATS else 'No incluido'}**
            - 🤖 Modelo de IA: **AppIA (mejor disponible)**
            """)


st.markdown("---")
st.markdown("""
### 📖 Guía Rápida de Uso:
1.  **� Configuración inicial:** Asegúrate de que la API esté configurada (verifica el indicador en la barra lateral).
2.  **📁 Carga Archivos:** Sube uno o dos archivos Excel (`.xlsx` o `.xls`).
3.  **📄 Selecciona Hojas:**
    * **Si cargas 1 archivo:** Selecciona la hoja específica que deseas analizar (si hay más de una).
    * **Si cargas 2 archivos:** Selecciona una hoja de cada archivo para la comparación.
    * Aparecerá una vista previa de las primeras filas de las hojas seleccionadas.
4.  **🛠️ Ajusta Opciones (Opcional):**
    * **Filas máximas por hoja:** Controla cuántas filas se analizarán de cada hoja.
    * **Incluir análisis estadístico:** Agrega estadísticas descriptivas al análisis.
    * **Generar gráficos automáticamente:** La IA creará visualizaciones relevantes.
5.  **💬 Haz tu Pregunta:** Escribe tu consulta sobre los datos de las hojas seleccionadas.
6.  **🚀 Analiza:** Presiona "Analizar y Preguntar". La respuesta y los gráficos aparecerán abajo.

**💡 Nota:** La aplicación está preconfigurada con los mejores ajustes para análisis de datos. No necesitas configurar parámetros técnicos.
""")

# Mostrar información de configuración al final para desarrolladores
with st.expander("ℹ️ Información técnica (para desarrolladores)"):
    models_list = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro"]
    st.markdown(f"""
    - **Modelos disponibles:** {", ".join(models_list)} (se prueba automáticamente en orden de preferencia)
    - **Modelo preferido:** {DEFAULT_MODEL}
    - **Tokens máximos de entrada:** {APPIA_MAX_INPUT_TOKENS:,}
    - **Tokens máximos de salida:** {APPIA_MAX_OUTPUT_TOKENS:,}
    - **Filas por defecto:** {DEFAULT_MAX_ROWS}
    - **Temperatura:** 0.7 (equilibrio entre creatividad y precisión)
    - **Fallback automático:** Si un modelo no está disponible, se intenta el siguiente
    """)