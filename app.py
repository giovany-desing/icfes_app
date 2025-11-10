import streamlit as st
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from PIL import Image
warnings.filterwarnings('ignore')


def load_image(image_path):
    """Carga una imagen de forma segura con múltiples intentos de ruta"""
    possible_paths = [
        image_path,
        Path(image_path),
        Path(__file__).parent / image_path,
        Path.cwd() / image_path,
    ]
    
    for path in possible_paths:
        try:
            if Path(path).exists():
                return Image.open(path)
        except Exception as e:
            continue
    
    st.error(f"""
    ⚠️ **Image not found:** `{image_path}`
    
    **Paths tried:**
    - `{Path.cwd() / image_path}`
    - `{Path(__file__).parent / image_path}`
    
    **Please ensure:**
    - File exists in the same folder as `app.py`
    - Filename matches exactly (case-sensitive)
    - File is a valid image (.png, .jpg, etc.)
    """)
    return None


# Configuración de la página
st.set_page_config(
    page_title="Proyecto ICFES - MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado estilo Apple
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: #000000;
        color: #f5f5f7;
    }
    
    /* Ocultar sidebar completamente */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Hero Section - Estilo Apple */
    .hero-section {
        text-align: center;
        padding: 8rem 2rem 6rem;
        background: linear-gradient(180deg, #000 0%, #0a0a0a 100%);
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fff 0%, #999 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        letter-spacing: -0.03em;
        line-height: 1.05;
    }
    
    .hero-subtitle {
        font-size: 1.75rem;
        font-weight: 400;
        color: #86868b;
        margin-bottom: 2.5rem;
        line-height: 1.4;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .hero-cta {
        display: inline-flex;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    /* Botones estilo Apple */
    .apple-button {
        background: #0071e3;
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 980px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 400;
        font-size: 1.0625rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        cursor: pointer;
    }
    
    .apple-button:hover {
        background: #0077ed;
        transform: scale(1.02);
    }
    
    .apple-button-outline {
        background: transparent;
        color: #0071e3 !important;
        border: 1px solid #0071e3;
    }
    
    .apple-button-outline:hover {
        background: rgba(0, 113, 227, 0.1);
    }
    
    /* Secciones con padding consistente */
    .content-section {
        padding: 5rem 10%;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .section-dark {
        background: #000000;
    }
    
    .section-light {
        background: #0a0a0a;
    }
    
    /* Cards estilo Apple */
    .apple-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 18px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 0.5px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
    }
    
    .apple-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }
    
    .card-title {
        font-size: 2rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .card-subtitle {
        font-size: 1.125rem;
        color: #86868b;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .section-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #f5f5f7;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.03em;
    }
    
    .section-subtitle {
        font-size: 1.5rem;
        color: #86868b;
        text-align: center;
        margin-bottom: 4rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Métricas estilo Apple Watch */
    .metric-ring {
        background: linear-gradient(135deg, rgba(0, 113, 227, 0.1) 0%, rgba(0, 113, 227, 0.05) 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(0, 113, 227, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-ring:hover {
        border-color: rgba(0, 113, 227, 0.4);
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 600;
        color: #0071e3;
        letter-spacing: -0.03em;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #86868b;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Feature badges estilo iOS */
    .ios-badge {
        display: inline-block;
        background: rgba(0, 113, 227, 0.15);
        color: #0071e3;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 0.5px solid rgba(0, 113, 227, 0.3);
    }
    
    /* Lista minimalista */
    .apple-list {
        list-style: none;
        padding: 0;
    }
    
    .apple-list li {
        padding: 1rem 0;
        border-bottom: 0.5px solid rgba(255, 255, 255, 0.1);
        font-size: 1.0625rem;
        color: #f5f5f7;
        line-height: 1.6;
    }
    
    .apple-list li:last-child {
        border-bottom: none;
    }
    
    /* Código estilo Xcode */
    pre {
        background: #1e1e1e !important;
        border-radius: 12px;
        padding: 1.5rem !important;
        border: 0.5px solid rgba(255, 255, 255, 0.1);
        font-family: 'SF Mono', Monaco, monospace !important;
        font-size: 0.875rem;
        line-height: 1.6;
        overflow-x: auto;
    }
    
    /* Grid de features estilo Apple.com */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 0.5px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-8px);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9375rem;
        color: #86868b;
        line-height: 1.5;
    }
    
    /* Progress bar estilo iOS */
    .ios-progress {
        background: rgba(255, 255, 255, 0.1);
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .ios-progress-fill {
        background: linear-gradient(90deg, #0071e3, #00a2ff);
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Tabla estilo Apple */
    .dataframe {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 0.5px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .dataframe th {
        color: #86868b !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 1rem !important;
    }
    
    .dataframe td {
        color: #f5f5f7 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    /* Footer estilo Apple */
    .apple-footer {
        text-align: center;
        padding: 3rem 0;
        border-top: 0.5px solid rgba(255, 255, 255, 0.1);
        margin-top: 5rem;
        color: #86868b;
    }
    
    .apple-footer a {
        color: #0071e3;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .apple-footer a:hover {
        color: #0077ed;
    }
    
    /* Animaciones suaves */
    * {
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }
        .hero-subtitle {
            font-size: 1.25rem;
        }
        .section-title {
            font-size: 2.5rem;
        }
        .content-section {
            padding: 3rem 5%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Proyecto de Machine Learning con enfoque en MLOps</div>
    <div class="hero-subtitle">
        Predicción del puntaje global ICFES mediante Gradient Boosting
    </div>
    <div class="hero-cta">
        <a href="https://github.com/giovany-desing/proyecto_icfes" target="_blank" class="apple-button">
            Ver Codigo en GitHub

</div>
""", unsafe_allow_html=True)


# Features grid
st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <span class="feature-icon">🧠</span>
        <div class="feature-name">Gradient Boosting</div>
        <div class="feature-desc">Algoritmo de aprendizaje ensemble de última generación</div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">📊</span>
        <div class="feature-name">5 Features Clave</div>
        <div class="feature-desc">Puntajes académicos optimizados para predicción</div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">⚡</span>
        <div class="feature-name">API de predicción costruida con fast api</div>
        <div class="feature-desc">Predicciones instantáneas vía FastAPI</div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🔄</span>
        <div class="feature-name">Pipeline de predicción automatizado</div>
        <div class="feature-desc">Entrenamiento y despliegue continuo</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Features de Entrada</div>
        <ul class="apple-list">
            <li>🌍 Puntaje de Inglés</li>
            <li>✍️ Comunicación Escrita</li>
            <li>🤝 Competencias Ciudadanas</li>
            <li>📖 Lectura Crítica</li>
            <li>🔢 Razonamiento Cuantitativo</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Variable Objetivo</div>
        <div style="padding: 3rem 0; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 600; color: #0071e3; margin-bottom: 0.5rem;">
                Puntaje Global ICFES
            </div>
            <div style="color: #86868b; font-size: 1rem;">
                Predicción del rendimiento académico total
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# SECCIÓN: METODOLOGÍA
st.markdown('<div class="content-section section-light">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Metodología de Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">GradientBoostingRegressor optimizado para capturar relaciones complejas</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Ventajas</div>
        <ul class="apple-list">
            <li>Precisión superior al 95%</li>
            <li>Manejo automático de correlaciones</li>
            <li>Robusto ante valores atípicos</li>
            <li>Captura patrones no lineales</li>
            <li>Interpretabilidad clara</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Hiperparámetros</div>
        <ul class="apple-list">
            <li>n_estimators: 200</li>
            <li>learning_rate: 0.1</li>
            <li>max_depth: 5</li>
            <li>min_samples_split: 10</li>
            <li>subsample: 0.8</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)





# SECCIÓN: MLOPS
st.markdown('<div class="content-section section-dark">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Arquitectura MLOps</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Stack completo de herramientas para gestión del ciclo de vida del modelo</div>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <span class="feature-icon">📊</span>
        <div class="feature-name">MLflow</div>
        <div class="feature-desc">Tracking de experimentos y registro de modelos</div>
        <div style="margin-top: 1rem;">
            <span class="ios-badge">Tracking</span>
            <span class="ios-badge">Registry</span>
        </div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🗄️</span>
        <div class="feature-name">DVC</div>
        <div class="feature-desc">Versionamiento de datos y reproducibilidad</div>
        <div style="margin-top: 1rem;">
            <span class="ios-badge">Data Version</span>
            <span class="ios-badge">Pipeline</span>
        </div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🔄</span>
        <div class="feature-name">Apache Airflow</div>
        <div class="feature-desc">Orquestación y automatización de pipelines</div>
        <div style="margin-top: 1rem;">
            <span class="ios-badge">Automation</span>
            <span class="ios-badge">Scheduling</span>
        </div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🚀</span>
        <div class="feature-name">FastAPI</div>
        <div class="feature-desc">API REST de alto rendimiento</div>
        <div style="margin-top: 1rem;">
            <span class="ios-badge">API</span>
            <span class="ios-badge">Docker</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# SECCIÓN: STACK TÉCNICO
st.markdown('<div class="content-section section-light">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Stack Tecnológico</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Construido con las herramientas más modernas del ecosistema ML</div>', unsafe_allow_html=True)

tech_categories = {
    "Machine Learning": ["Python 3.9+", "Scikit-learn", "Pandas", "NumPy"],
    "MLOps": ["MLflow", "DVC",],
    "API & Deployment": ["FastAPI"],
    "Testing": ["Pytest", "GitHub Actions"]
}

for category, techs in tech_categories.items():
    st.markdown(f"""
    <div class="apple-card">
        <div style="font-size: 1.25rem; font-weight: 600; color: #f5f5f7; margin-bottom: 1rem;">
            {category}
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
            {"".join([f'<span class="ios-badge">{tech}</span>' for tech in techs])}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
# DISTRIBUCION DE 
st.markdown('<div class="content-section section-dark" id="documentacion">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Distribución de archivos</div>', unsafe_allow_html=True)

# Cargar imagen con efecto premium
pipeline_img = load_image("archivos.png")
if pipeline_img:
    st.image(pipeline_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# SECCIÓN: DOCUMENTACIÓN
st.markdown('<div class="content-section section-dark" id="documentacion">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Documentación</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Guías completas para instalación, uso y despliegue</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Instalación Rápida</div>
        <pre style="color: #0071e3;">
# Clonar repositorio
git clone https://github.com/tu-usuario/proyecto-icfes.git
cd proyecto-icfes

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos
dvc pull

# Entrenar modelo
python src/models/train.py

# Iniciar API
uvicorn src.api.main:app --reload
        </pre>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Uso de la API</div>
        <pre style="color: #0071e3;">
# Realizar predicción
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
  "modelo_version": "v1.2.0"
}
        </pre>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# SECCIÓN: RESULTADOS
st.markdown('<div class="content-section section-light">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Resultados del Modelo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Métricas de rendimiento y comparación con otros algoritmos</div>', unsafe_allow_html=True)

# Importancia de features
st.markdown("""
<div class="apple-card">
    <div class="card-title" style="font-size: 1.5rem;">Importancia de Features</div>
</div>
""", unsafe_allow_html=True)

features = [
    ("Razonamiento Cuantitativo", 32),
    ("Lectura Crítica", 28),
    ("Inglés", 22),
    ("Competencias Ciudadanas", 11),
    ("Comunicación Escrita", 7)
]

for feature, importance in features:
    st.markdown(f"""
    <div class="apple-card" style="padding: 1.25rem; margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
            <span style="font-weight: 500; color: #f5f5f7;">{feature}</span>
            <span style="color: #0071e3; font-weight: 600;">{importance}%</span>
        </div>
        <div class="ios-progress">
            <div class="ios-progress-fill" style="width: {importance}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Comparación de modelos
st.markdown("""
<div class="apple-card" style="margin-top: 3rem;">
    <div class="card-title" style="font-size: 1.5rem;">Comparación de Modelos</div>
</div>
""", unsafe_allow_html=True)

models_data = {
    'Modelo': ['Gradient Boosting', 'Random Forest', 'XGBoost', 'Linear Regression'],
    'R² Score': [0.952, 0.938, 0.945, 0.812],
    'RMSE': [12.3, 14.2, 13.1, 22.5],
    'Tiempo (s)': [2.3, 1.8, 2.1, 0.5]
}
df_models = pd.DataFrame(models_data)

st.dataframe(
    df_models.style.highlight_max(subset=['R² Score'], color='rgba(0, 113, 227, 0.2)')
                  .highlight_min(subset=['RMSE', 'Tiempo (s)'], color='rgba(0, 113, 227, 0.2)')
                  .format({'R² Score': '{:.3f}', 'RMSE': '{:.1f}', 'Tiempo (s)': '{:.1f}'}),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="apple-card" style="margin-top: 2rem;">
        <div class="card-title" style="font-size: 1.25rem;">Métricas Principales</div>
        <ul class="apple-list">
            <li>R² Score: 0.952</li>
            <li>RMSE: 12.3</li>
            <li>MAE: 8.5</li>
            <li>Tiempo de entrenamiento: 2.3s</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="apple-card" style="margin-top: 2rem;">
        <div class="card-title" style="font-size: 1.25rem;">Próximos Pasos</div>
        <ul class="apple-list">
            <li>Monitoring de drift de datos</li>
            <li>A/B testing de modelos</li>
            <li>Dashboard en tiempo real</li>
            <li>Autenticación JWT en API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

