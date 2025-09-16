# 📊 Analizador Contable Multi-Excel con IA

Una aplicación web intuitiva que utiliza Inteligencia Artificial (Google Gemini) para analizar archivos Excel y generar insights automáticamente, incluyendo visualizaciones interactivas.

## 🚀 Características Principales

- **Análisis inteligente** de datos de Excel usando IA de Google Gemini
- **Visualizaciones automáticas** - La IA genera gráficos relevantes
- **Comparación entre archivos** - Analiza hasta 2 archivos Excel simultáneamente
- **Interfaz intuitiva** - No requiere conocimientos técnicos
- **Análisis estadístico** automático incluido

## 📋 Requisitos Previos

- Python 3.8 o superior
- Una API key de Google Gemini (gratuita)

## 🛠️ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd appmedellin-IA
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar la API key de Google Gemini

#### Paso 4.1: Obtener la API key
1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Inicia sesión con tu cuenta de Google
3. Crea una nueva API key
4. Copia la API key generada

#### Paso 4.2: Configurar el archivo .env
1. Abre el archivo `.env` en el directorio del proyecto
2. Reemplaza `tu_api_key_aqui` con tu API key real:
```
GOOGLE_GEMINI_API_KEY=AIzaSyC-tu_api_key_real_aqui
```
3. Guarda el archivo

⚠️ **IMPORTANTE**: El archivo `.env` contiene información sensible y no debe compartirse públicamente.

### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Cómo Usar la Aplicación

### Paso a Paso:

1. **Verificar configuración** - Asegúrate de ver "✅ API configurada correctamente" en la barra lateral

2. **Cargar archivos** - Sube uno o dos archivos Excel (.xlsx o .xls)

3. **Seleccionar hojas** - Elige las hojas específicas a analizar

4. **Hacer preguntas** - Escribe consultas en lenguaje natural como:
   - "¿Cuáles son las tendencias de ventas?"
   - "Compara los gastos entre ambos archivos"
   - "Muestra un resumen por categorías"

5. **Obtener resultados** - La IA generará análisis detallados y gráficos automáticamente

### Ejemplos de Preguntas:

**Para un archivo:**
- "¿Cuál es el total de ingresos por mes?"
- "Muestra la distribución de gastos por categoría"
- "¿Hay tendencias estacionales en los datos?"

**Para comparar dos archivos:**
- "¿Cuáles son las diferencias principales entre ambos?"
- "Compara los totales de ventas"
- "¿En qué categorías hay mayores variaciones?"

## ⚙️ Configuraciones Avanzadas

La aplicación está preconfigurada con los mejores ajustes, pero puedes modificar:

- **Filas máximas por hoja**: Controla la cantidad de datos a analizar
- **Análisis estadístico**: Incluye/excluye estadísticas descriptivas
- **Gráficos automáticos**: Activa/desactiva la generación de visualizaciones

## 🔧 Configuración Técnica

- **Modelo de IA**: Gemini-1.5-Pro (el más avanzado disponible)
- **Tokens máximos**: 2,097,152 (entrada) / 8,192 (salida)
- **Temperatura**: 0.7 (equilibrio entre creatividad y precisión)

## 🆘 Solución de Problemas

### "❌ API no configurada"
- Verifica que el archivo `.env` existe
- Confirma que la API key es correcta
- Asegúrate de que no hay espacios extra en la API key

### "Error al consultar a Gemini"
- Revisa tu conexión a internet
- Verifica que la API key es válida y no ha expirado
- Confirma que tienes cuota disponible en Google AI Studio

### Errores de instalación
```bash
# Si hay problemas con dependencias:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📞 Soporte

Si encuentras problemas:
1. Revisa la sección de "Información técnica" en la aplicación
2. Verifica que todos los pasos de instalación se completaron
3. Consulta los logs de error en la terminal

## 🔒 Seguridad

- El archivo `.env` está incluido en `.gitignore` para proteger tu API key
- Nunca compartas tu API key públicamente
- La aplicación no almacena datos de tus archivos Excel permanentemente

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.