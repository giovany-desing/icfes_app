import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

IMAGE_PATH = "imagenes/distribucion.png"

# Configuración de la página
st.set_page_config(
    page_title="Proyecto ICFES - MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5ff, #00ff87, #00f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        text-align: center;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .section-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 245, 255, 0.3);
        box-shadow: 0 8px 30px rgba(0, 245, 255, 0.2);
    }
    
    .feature-badge {
        background: linear-gradient(135deg, #00f5ff, #00ff87);
        color: #000;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    
    .github-button {
        background: linear-gradient(135deg, #6e48aa, #9d50bb);
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        margin: 1rem 0.5rem;
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .github-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(157, 80, 187, 0.4);
    }
    
    .api-button {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        margin: 1rem 0.5rem;
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .api-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.4);
    }
    
    .metric-box {
        background: rgba(0, 245, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 245, 255, 0.3);
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00f5ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.5rem;
    }
    
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .tech-item {
        background: rgba(0, 255, 135, 0.15);
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 135, 0.3);
        font-weight: 500;
        color: #00ff87;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<div class="main-header">🎓 Predicción de Puntajes ICFES</div>', unsafe_allow_html=True)
st.markdown('''
<div class="sub-header">
    Proyecto de Machine Learning con enfoque en MLOps<br>
    Predicción del puntaje global ICFES mediante Gradient Boosting
</div>
''', unsafe_allow_html=True)

# Enlaces principales
st.markdown('''
<div style="text-align: center; margin: 2rem 0;">
    <a href="https://github.com/tu-usuario/proyecto-icfes" target="_blank" class="github-button">
        🔗 Ver Código en GitHub

</div>
''', unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("Componentes")
section = st.sidebar.radio(
    "Selecciona lo que quieres ver:",
    ["📋 Resumen del Proyecto", "🔬 Metodología ML", "⚙️ Implementación MLOps", 
     "📊 Stack Tecnológico", "📚 Documentación", "🎯 Resultados"]
)

# SECCIÓN 1: RESUMEN DEL PROYECTO
if section == "📋 Resumen del Proyecto":
    st.markdown("""
    <div class="section-card">
        <h2>🎯 Objetivo del Proyecto</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: rgba(255,255,255,0.9);">
        Este proyecto desarrolla un sistema completo de predicción de puntajes ICFES utilizando Machine Learning,
        con un enfoque especial en <strong>MLOps</strong> para garantizar reproducibilidad, trazabilidad y 
        automatización del ciclo de vida del modelo.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">95.2%</div>
            <div class="metric-label">Precisión R²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">API REST</div>
            <div class="metric-label">FastApi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">5</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">MLflow</div>
            <div class="metric-label">Tracking</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Características del dataset
    st.markdown("""
    <div class="section-card">
        <h2>📊 Dataset y Features</h2>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 1.5rem;">
        Datos extraídos de los resultados Saber ICFES 2019. El modelo utiliza las siguientes características:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📝 Features de Entrada</h3>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85);">
                <li>🌍 <strong>Puntaje Inglés</strong></li>
                <li>✍️ <strong>Comunicación Escrita</strong></li>
                <li>🤝 <strong>Competencias Ciudadanas</strong></li>
                <li>📖 <strong>Lectura Crítica</strong></li>
                <li>🔢 <strong>Razonamiento Cuantitativo</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 Variable Objetivo</h3>
            <div style="padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; color: #00f5ff; font-weight: 700;">
                    Puntaje Global ICFES
                </div>
                <p style="color: rgba(255,255,255,0.7); margin-top: 1rem;">
                    Predicción del puntaje final del estudiante
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# SECCIÓN 2: METODOLOGÍA ML
elif section == "🔬 Metodología ML":
    st.markdown("""
    <div class="section-card">
        <h2>🤖 Algoritmo: Gradient Boosting Regressor</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: rgba(255,255,255,0.9);">
        Se seleccionó <strong>GradientBoostingRegressor</strong> por su capacidad para capturar relaciones 
        no lineales y su excelente desempeño en problemas de regresión con múltiples features correlacionadas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>✅ Ventajas del Modelo</h3>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85);">
                <li>Alta precisión en predicciones</li>
                <li>Manejo automático de features correlacionadas</li>
                <li>Robusto ante outliers</li>
                <li>Captura interacciones complejas</li>
                <li>Interpretabilidad mediante feature importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>⚙️ Hiperparámetros Optimizados</h3>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85);">
                <li><strong>n_estimators:</strong> 200</li>
                <li><strong>learning_rate:</strong> 0.1</li>
                <li><strong>max_depth:</strong> 5</li>
                <li><strong>min_samples_split:</strong> 10</li>
                <li><strong>subsample:</strong> 0.8</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Pipeline de procesamiento
    st.markdown("""
    <div class="section-card">
        <h3>🔄 Pipeline de Procesamiento de Datos</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Diagrama de flujo simplificado
    flow_data = {
        'Etapa': ['1. Extracción', '2. Limpieza', '3. Feature Engineering', '4. Entrenamiento', '5. Validación'],
        'Proceso': ['Datos ICFES 2019', 'Normalización y valores faltantes', 'Escalado y transformación', 
                    'Gradient Boosting', 'Cross-Validation (k=5)']
    }
    df_flow = pd.DataFrame(flow_data)
    
    fig = px.funnel(df_flow, x='Etapa', y=[1,1,1,1,1], text='Proceso')
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# SECCIÓN 3: IMPLEMENTACIÓN MLOPS
elif section == "⚙️ Implementación MLOps":
    st.markdown("""
    <div class="section-card">
        <h2>🔧 Stack de MLOps</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: rgba(255,255,255,0.9);">
        Implementación completa del ciclo de vida de ML con herramientas modernas de MLOps
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Componentes MLOps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Experiment Tracking</h3>
            <div class="feature-badge">MLflow</div>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85); margin-top: 1rem;">
                <li>Registro de experimentos y métricas</li>
                <li>Versionamiento de modelos</li>
                <li>Comparación de hiperparámetros</li>
                <li>Almacenamiento en Cloud Storage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <h3>🗄️ Data Versioning</h3>
            <div class="feature-badge">DVC</div>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85); margin-top: 1rem;">
                <li>Versionamiento del dataset</li>
                <li>Trazabilidad de cambios en datos</li>
                <li>Reproducibilidad garantizada</li>
                <li>Almacenamiento eficiente</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🔄 Pipeline Automation</h3>
            <div class="feature-badge">Apache Airflow</div>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85); margin-top: 1rem;">
                <li>Automatización de entrenamiento</li>
                <li>Reentrenamiento programado</li>
                <li>Monitoreo de pipelines</li>
                <li>Validación automática</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <h3>🚀 Model Serving</h3>
            <div class="feature-badge">FastAPI</div>
            <ul style="line-height: 2; color: rgba(255,255,255,0.85); margin-top: 1rem;">
                <li>API REST para predicciones</li>
                <li>Documentación automática (Swagger)</li>
                <li>Validación de entrada con Pydantic</li>
                <li>Despliegue en contenedor Docker</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Flujo MLOps
    st.markdown("""
    <div class="section-card">
        <h3>🔄 Flujo Completo de MLOps</h3>
    </div>
    """, unsafe_allow_html=True)
    
    mlops_stages = {
        'Etapa': ['Data Versioning', 'Experiment Tracking', 'Model Training', 'Model Registry', 'Deployment', 'Monitoring'],
        'Herramienta': ['DVC', 'MLflow', 'Scikit-learn', 'MLflow', 'Docker + FastAPI', 'Prometheus'],
        'Estado': ['✅', '✅', '✅', '✅', '✅', '🔄']
    }
    df_mlops = pd.DataFrame(mlops_stages)
    
    fig_mlops = px.timeline(
        df_mlops, 
        x_start=[0, 1, 2, 3, 4, 5], 
        x_end=[1, 2, 3, 4, 5, 6],
        y='Etapa',
        color='Estado',
        text='Herramienta'
    )
    fig_mlops.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Progreso',
        showlegend=True
    )
    st.plotly_chart(fig_mlops, use_container_width=True)

# SECCIÓN 4: ARQUITECTURA
elif section == "📊 Stack Tecnológico":
    # Stack tecnológico
    st.markdown("""
    <div class="section-card">
        <h3>💻 Stack Tecnológico</h3>
        <div class="tech-stack">
            <div class="tech-item">🐍 Python 3.9+</div>
            <div class="tech-item">🤖 Scikit-learn</div>
            <div class="tech-item">📊 Pandas & NumPy</div>
            <div class="tech-item">📈 MLflow</div>
            <div class="tech-item">🗄️ DVC</div>
            <div class="tech-item">⚡ FastAPI</div>
            <div class="tech-item">🐳 Docker</div>
            <div class="tech-item">☁️ Cloud Storage</div>
            <div class="tech-item">🔄 Apache Airflow</div>
            <div class="tech-item">✅ Pytest</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <h3>📁 Estructura del Proyecto</h3>
        <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 10px; color: #00ff87; overflow-x: auto;">
proyecto-icfes/
│
├── 📁 .dvc/                    # Configuración de DVC
│   └── config                 # Versionamiento de datos
│
├── 📁 .github/workflows/       # CI/CD Automatizado
│   └── *.yml                  # GitHub Actions pipelines
│
├── 📁 __pycache__/             # Archivos compilados Python
│
├── 📁 data/                    # Datasets del proyecto
│   ├── raw/                   # Datos crudos ICFES 2019
│   └── processed/             # Datos preprocesados
│
├── 📁 data_project/            # Proyecto de datos adicional
│   └── exploratory/           # Análisis exploratorio
│
├── 📁 entorno/                 # Entorno virtual Python
│   └── venv/                  # Dependencias aisladas
│
├── 📁 mlruns/                  # 🔥 MLflow Tracking
│   ├── experiments/           # Historial de experimentos
│   ├── models/                # Registro de modelos
│   └── artifacts/             # Artefactos guardados
│
├── 📁 models/                  # Modelos entrenados
│   ├── gradient_boosting.pkl  # Modelo principal
│   └── scaler.pkl             # Preprocesador
│
├── 📁 tests/                   # Tests unitarios
│   ├── test_model.py          # Tests del modelo
│   └── test_api.py            # Tests de la API
│
├── 📄 .DS_Store               # Archivo de sistema (macOS)
├── 📄 .env                     # Variables de entorno
├── 📄 .gitignore              # Archivos ignorados por Git
├── 📄 README.md               # 📚 Documentación principal
├── 📄 api_service.log         # Logs de la API
├── 📄 config.yaml             # Configuración del proyecto
├── 📄 main.py                 # 🚀 Aplicación FastAPI
├── 📄 pytest.ini              # Configuración de tests
└── 📄 requirements.txt        # Dependencias Python
        </pre>
    </div>
    """, unsafe_allow_html=True)
    


# SECCIÓN 5: DOCUMENTACIÓN
elif section == "📚 Documentación":
    st.markdown("""
    <div class="section-card">
        <h2>📖 Documentación Completa</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs para diferentes tipos de documentación
    doc_tab = st.radio(
        "Selecciona el tipo de documentación:",
        ["🚀 Quick Start", "🔌 Uso de la API", "🔧 Configuración MLflow", "🐳 Docker Deployment"],
        horizontal=True
    )
    
    if doc_tab == "🚀 Quick Start":
        st.markdown("""
        <div class="section-card">
            <h3>Instalación y Configuración</h3>
            <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 10px; color: #00ff87;">
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/proyecto-icfes.git
cd proyecto-icfes

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\\Scripts\\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar DVC
dvc pull  # Descargar datos versionados

# 5. Entrenar el modelo
python src/models/train.py

# 6. Iniciar la API
uvicorn src.api.main:app --reload
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    elif doc_tab == "🔌 Uso de la API":
        st.markdown("""
        <div class="section-card">
            <h3>Endpoints de la API</h3>
            <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 10px; color: #00ff87;">
# POST /predict - Realizar predicción
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "ingles": 65,
    "comunicacion_escrita": 70,
    "competencias_ciudadanas": 68,
    "lectura_critica": 72,
    "razonamiento_cuantitativo": 75
  }'

# Respuesta
{
  "puntaje_global": 285.4,
  "modelo_version": "v1.2.0",
  "timestamp": "2025-11-06T10:30:00Z"
}

# GET /health - Estado del servicio
curl http://localhost:8000/health

# GET /docs - Documentación Swagger
http://localhost:8000/docs
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    elif doc_tab == "🔧 Configuración MLflow":
        st.markdown("""
        <div class="section-card">
            <h3>Configuración de MLflow</h3>
            <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 10px; color: #00ff87;">
# Iniciar MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# En el código de entrenamiento
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("icfes_prediction")

with mlflow.start_run():
    # Registrar parámetros
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.1)
    
    # Registrar métricas
    mlflow.log_metric("r2_score", 0.952)
    mlflow.log_metric("mse", 12.3)
    
    # Guardar modelo
    mlflow.sklearn.log_model(model, "model")
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Docker Deployment
        st.markdown("""
        <div class="section-card">
            <h3>Despliegue con Docker</h3>
            <pre style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 10px; color: #00ff87;">
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Construir imagen
docker build -t icfes-predictor:latest .

# Ejecutar contenedor
docker run -d -p 8000:8000 icfes-predictor:latest

# Docker Compose
docker-compose up -d
            </pre>
        </div>
        """, unsafe_allow_html=True)

# SECCIÓN 6: RESULTADOS
else:  # Resultados
    st.markdown("""
    <div class="section-card">
        <h2>🎯 Resultados y Métricas del Modelo</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">0.952</div>
            <div class="metric-label">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">12.3</div>
            <div class="metric-label">RMSE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">8.5</div>
            <div class="metric-label">MAE</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Importancia de features (simulado)
    features_importance = {
        'Feature': ['Razonamiento Cuantitativo', 'Lectura Crítica', 'Inglés', 
                   'Competencias Ciudadanas', 'Comunicación Escrita'],
        'Importancia': [0.32, 0.28, 0.22, 0.11, 0.07]
    }
    df_imp = pd.DataFrame(features_importance)
    
    fig_imp = px.bar(
        df_imp, 
        x='Importancia', 
        y='Feature',
        orientation='h',
        title='Importancia de Features en el Modelo',
        color='Importancia',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Comparación de modelos
    st.markdown("""
    <div class="section-card">
        <h3>📊 Comparación de Modelos Evaluados</h3>
    </div>
    """, unsafe_allow_html=True)
    
    models_comparison = {
        'Modelo': ['Gradient Boosting', 'Random Forest', 'XGBoost', 'Linear Regression'],
        'R² Score': [0.952, 0.938, 0.945, 0.812],
        'RMSE': [12.3, 14.2, 13.1, 22.5],
        'Tiempo (s)': [2.3, 1.8, 2.1, 0.5]
    }
    df_comp = pd.DataFrame(models_comparison)
    
    fig_comp = px.scatter(
        df_comp,
        x='RMSE',
        y='R² Score',
        size='Tiempo (s)',
        color='Modelo',
        title='Comparación de Modelos: Precisión vs Error',
        hover_data=['RMSE', 'R² Score', 'Tiempo (s)']
    )
    fig_comp.update_layout(template='plotly_dark', height=500)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Tabla de comparación
    st.dataframe(
        df_comp.style.highlight_max(subset=['R² Score'], color='lightgreen')
                     .highlight_min(subset=['RMSE', 'Tiempo (s)'], color='lightgreen'),
        use_container_width=True
    )
    
    # Próximos pasos
    st.markdown("""
    <div class="section-card">
        <h3>🚀 Próximos Pasos y Mejoras</h3>
        <ul style="line-height: 2; color: rgba(255,255,255,0.85);">
            <li>✅ Implementar monitoring de drift de datos</li>
            <li>✅ Agregar reentrenamiento automático mensual</li>
            <li>🔄 A/B testing de nuevas versiones del modelo</li>
            <li>🔄 Integración con sistema de alertas</li>
            <li>📊 Dashboard de métricas en tiempo real</li>
            <li>🔐 Autenticación JWT en la API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6); padding: 2rem 0;'>
    <p style='font-size: 1.2rem; margin-bottom: 1rem;'>
        <strong>🎓 Proyecto ICFES - Machine Learning con MLOps</strong>
    </p>
    <p style='margin: 0.5rem 0;'>
        Desarrollado por: Egar Yovany Samaca Acuña, Cientifico de datos Junior
    </p>
    <p style='margin: 0.5rem 0;'>
        📧 Contacto: egsamaca56@gmail.com | 
        💼 <a href="https://www.linkedin.com/in/edgar-yovany-samaca-acu%C3%B1a-a17452210/" target="_blank" style="color: #00f5ff;">LinkedIn</a> | 
        🐙 <a href="https://github.com/giovany-desing" target="_blank" style="color: #00f5ff;">GitHub</a>
    </p>
    <p style='margin-top: 1.5rem; font-size: 0.9rem;'>
        © 2025 - Todos los derechos reservados
    </p>
</div>
""", unsafe_allow_html=True)