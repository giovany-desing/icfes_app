import streamlit as st
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from PIL import Image
import requests
import json
warnings.filterwarnings('ignore')


def load_image(image_path):
    """Carga y renderiza una imagen con la máxima resolución y calidad posible"""
    possible_paths = [
        image_path,
        Path(image_path),
        Path(__file__).parent / image_path,
        Path.cwd() / image_path,
        Path(__file__).parent / "images" / image_path,
        Path.cwd() / "images" / image_path,
    ]

    for path in possible_paths:
        try:
            if Path(path).exists():
                # Cargar imagen original sin alteraciones
                original_image = Image.open(path)
                
                # Preservar TODOS los datos de la imagen
                original_image.load()
                
                # Convertir a RGB solo si es necesario para compatibilidad
                if original_image.mode in ('P', 'RGBA', 'LA'):
                    # Preservar transparencia si existe
                    if original_image.mode in ('RGBA', 'LA'):
                        image_rgb = Image.new('RGBA', original_image.size)
                        image_rgb.paste(original_image, (0, 0))
                    else:
                        image_rgb = original_image.convert('RGB')
                else:
                    image_rgb = original_image
                
                return image_rgb
                
        except Exception as e:
            continue

    st.error(f"❌ No se pudo cargar la imagen: {image_path}")
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
    
    /* Inputs estilo Apple */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 0.5px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #f5f5f7 !important;
        font-size: 1.125rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #0071e3 !important;
        box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.2) !important;
    }
    
    .stNumberInput label {
        color: #f5f5f7 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Text input estilo Apple */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 0.5px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #f5f5f7 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput label {
        color: #f5f5f7 !important;
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
    }
    
    /* Botón de Streamlit estilo Apple */
    .stButton > button {
        background: linear-gradient(135deg, #0071e3, #0077ed) !important;
        color: white !important;
        padding: 1rem 3rem !important;
        border-radius: 980px !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 1.125rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(0, 113, 227, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(0, 113, 227, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
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

# ==================== MÉTRICAS IMPACTANTES ====================
st.markdown('<div class="section-title">🎓 Predicción de Puntajes ICFES</div>', unsafe_allow_html=True)
st.markdown('<div class="content-section section-dark">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <div style="font-size: 2rem; font-weight: 600; color: #f5f5f7; margin-bottom: 0.5rem;">
    </div>
    <div style="font-size: 1.125rem; color: #86868b;">
        Este proyecto representa una implementación completa de MLOps siguiendo los estándares de la industria moderna, diseñado
        para predecir con alta precisión los puntajes del examen ICFES. Va más allá de un simple modelo de machine
        learning: es un sistema productivo end-to-end que integra automatización, versionado, experimentación sistemática y
        deployment continuo.
    </div>
</div>
""", unsafe_allow_html=True)



# ==================== ARQUITECTURA DEL SISTEMA ====================
st.markdown('<div class="content-section section-light">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Stack tecnológico</div>', unsafe_allow_html=True)
# ==================== MLOPS FEATURES ====================
st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <span class="feature-icon">📊</span>
        <div class="feature-name">EXPERIMENTACIÓN</div>
        <div class="feature-desc">MLflow + Optuna</div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🗄️</span>
        <div class="feature-name">VERSIONADO</div>
        <div class="feature-desc">Git + DVC + AWS S3</div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
    <div class="feature-item">
        <span class="feature-icon">🚀</span>
        <div class="feature-name">API PRODUCCIÓN</div>
        <div class="feature-desc">FastAPI + Uvicorn</div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
        <div class="feature-item">
        <span class="feature-icon">🧠</span>
        <div class="feature-name">ENTRENAMIENTO</div>
        <div class="feature-desc">scikit-learn, XGBoost </div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
        <div class="feature-item">
        <span class="feature-icon">🔄</span>
        <div class="feature-name">CI/CD </div>
        <div class="feature-desc">GitHub Actions</div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
        <div class="feature-item">
        <span class="feature-icon">🐳</span>
        <div class="feature-name">CONTAINERIZACIÓN</div>
        <div class="feature-desc">Docker + Docker Hub </div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
        <div class="feature-item">
        <span class="feature-icon">🌐</span>
        <div class="feature-name">DEPLOYMENT</div>
        <div class="feature-desc">Render (PaaS) </div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
        <div class="feature-item">
        <span class="feature-icon">👀</span>
        <div class="feature-name">MONITOREO</div>
        <div class="feature-desc">Structured Logging + Health Checks </div>
        <div style="margin-top: 1rem;">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

arch_components = st.columns(3)
# ==================== SECCIÓN DE PREDICCIÓN INTERACTIVA ====================
st.markdown('<div class="content-section section-dark">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prueba el Modelo en Vivo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Ingresa los puntajes de cada área y obtén la predicción del puntaje global ICFES</div>', unsafe_allow_html=True)

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None
if 'api_response_raw' not in st.session_state:
    st.session_state.api_response_raw = None

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Features de Entrada</div>
        <div style="color: #86868b; font-size: 0.9375rem; margin-bottom: 1.5rem;">
            Ingresa los puntajes de cada área evaluada
        </div>
    </div>
    """, unsafe_allow_html=True)

    ingles = st.number_input(
        "🌍 Puntaje de Inglés",
        min_value=0,
        max_value=100,
        value=65,
        step=1,
        help="Ingresa el puntaje obtenido en Inglés (0-100)",
        key="ingles_input"
    )

    comunicacion = st.number_input(
        "✍️ Comunicación Escrita",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
        help="Ingresa el puntaje obtenido en Comunicación Escrita (0-100)",
        key="comunicacion_input"
    )

    competencias = st.number_input(
        "🤝 Competencias Ciudadanas",
        min_value=0,
        max_value=100,
        value=68,
        step=1,
        help="Ingresa el puntaje obtenido en Competencias Ciudadanas (0-100)",
        key="competencias_input"
    )

    lectura = st.number_input(
        "📖 Lectura Crítica",
        min_value=0,
        max_value=100,
        value=72,
        step=1,
        help="Ingresa el puntaje obtenido en Lectura Crítica (0-100)",
        key="lectura_input"
    )

    razonamiento = st.number_input(
        "🔢 Razonamiento Cuantitativo",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        help="Ingresa el puntaje obtenido en Razonamiento Cuantitativo (0-100)",
        key="razonamiento_input"
    )

with col2:
    st.markdown("""
    <div class="apple-card">
        <div class="card-title" style="font-size: 1.5rem;">Variable Objetivo</div>
        <div style="color: #86868b; font-size: 0.9375rem; margin-bottom: 1.5rem;">
            Predicción del rendimiento académico total
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.prediction_result:
        prediction_value = None
        prediction_keys = ['predicted_score', 'result', 'puntaje_global', 'PUNT_GLOBAL', 'prediction', 'score']

        for key in prediction_keys:
            if key in st.session_state.prediction_result:
                prediction_value = st.session_state.prediction_result[key]
                break

        if prediction_value is not None:
            st.markdown(f"""
            <div class="apple-card" style="background: linear-gradient(135deg, rgba(0, 113, 227, 0.15) 0%, rgba(0, 113, 227, 0.05) 100%); border-color: rgba(0, 113, 227, 0.3);">
                <div style="text-align: center; padding: 2rem 0;">
                    <div style="font-size: 4rem; font-weight: 700; color: #0071e3; margin-bottom: 1rem;">
                        {float(prediction_value):.0f}
                    </div>
                    <div style="font-size: 1.25rem; color: #f5f5f7; font-weight: 500; margin-bottom: 0.5rem;">
                        Puntaje Global ICFES
                    </div>
                    <div style="font-size: 0.875rem; color: #86868b;">
                        Predicción generada por el modelo
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="apple-card" style="background: rgba(255, 204, 0, 0.1); border-color: rgba(255, 204, 0, 0.3);">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                    <div style="color: #ffcc00; font-size: 1.125rem; font-weight: 500;">
                        No se pudo obtener la predicción
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif st.session_state.prediction_error:
        st.markdown(f"""
        <div class="apple-card" style="background: rgba(255, 59, 48, 0.1); border-color: rgba(255, 59, 48, 0.3);">
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <div style="color: #ff3b30; font-size: 1.125rem; font-weight: 500;">
                    Error en la predicción
                </div>
                <div style="color: #86868b; font-size: 0.875rem; margin-top: 0.5rem;">
                    {st.session_state.prediction_error}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="apple-card" style="background: rgba(255, 255, 255, 0.02);">
            <div style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;">🎯</div>
                <div style="color: #86868b; font-size: 1rem;">
                    Ingresa los puntajes y haz clic en<br>"Realizar Predicción"
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)

# URL CONFIGURABLE - CAMBIO PRINCIPAL AQUÍ
api_url = st.text_input(
    "🔗 URL del Endpoint de Predicción",
    value="https://prediccion-icfes-latest.onrender.com/predict",
    help="Endpoint configurado para realizar las predicciones (editable)",
    key="api_endpoint",
    disabled=False  # ← Ahora es editable
)

if st.button("🚀 Realizar Predicción", use_container_width=False, key="predict_button"):
    with st.spinner("Procesando predicción..."):
        try:
            payload = {
                "MOD_INGLES_PNAL": float(ingles),
                "MOD_COMUNI_ESCRITA_PNAL": float(comunicacion),
                "MOD_COMPETEN_CIUDADA_PNAL": float(competencias),
                "MOD_LECTURA_CRITICA_PNAL": float(lectura),
                "MOD_RAZONA_CUANTITATIVO_PNAL": float(razonamiento)
            }

            # USAR LA URL DEL INPUT - CAMBIO PRINCIPAL AQUÍ
            response = requests.post(
                api_url,  # ← Usa la URL configurable
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            st.session_state.api_response_raw = f"Status: {response.status_code}\nBody: {response.text}"

            if response.status_code == 200:
                try:
                    result = response.json()
                    st.session_state.prediction_result = result
                    st.session_state.prediction_error = None
                    st.success("✅ Predicción realizada exitosamente")
                except json.JSONDecodeError:
                    st.session_state.prediction_error = f"La API devolvió una respuesta no JSON: {response.text}"
                    st.session_state.prediction_result = None
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                st.session_state.prediction_error = error_message
                st.session_state.prediction_result = None

            st.rerun()

        except requests.exceptions.ConnectionError:
            st.session_state.prediction_error = "No se pudo conectar con la API."
            st.session_state.prediction_result = None
            st.error("❌ No se pudo conectar con la API")
        except requests.exceptions.Timeout:
            st.session_state.prediction_error = "La solicitud tardó demasiado tiempo."
            st.session_state.prediction_result = None
            st.error("⏱️ Timeout")
        except Exception as e:
            st.session_state.prediction_error = f"Error: {str(e)}"
            st.session_state.prediction_result = None
            st.error(f"❌ {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# ==================== FIN TEST ENDPOINT ====================


# github
st.markdown("""
<div class="hero-section">
    <div class="hero-cta">
        <a href="https://github.com/giovany-desing/predecir-puntaje-icfes-final" target="_blank" class="apple-button">
            Ver Código en GitHub
        </a>
        <a href="https://github.com/giovany-desing/predecir-puntaje-icfes-final/blob/main/README.md" target="_blank" class="apple-button">
            ir a Documentación
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


# DISTRIBUCION DE ARCHIVOS
st.markdown('<div class="content-section section-dark" id="documentacion">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Archivos del proyecto</div>', unsafe_allow_html=True)

# Cargar imagen con efecto premium
pipeline_img = load_image("archivos.png")
if pipeline_img:
    st.image(pipeline_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# PIPELINE DE DATOS
st.markdown('<div class="content-section section-dark" id="documentacion">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flujos del proyecto</div>', unsafe_allow_html=True)

# Cargar imagen con efecto premium
pipeline_img = load_image("flujos.png")
if pipeline_img:
    st.image(pipeline_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)




st.markdown('</div>', unsafe_allow_html=True)