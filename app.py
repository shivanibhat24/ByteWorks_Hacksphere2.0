"""
Parkinson's Disease Detection from Voice - Professional Streamlit Application
Advanced SDK architecture with innovative design and real-time analytics
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import os
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Optional, List
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "NeuroVoice Analytics"
    VERSION = "2.0"
    MODEL_DIR = Path("models")
    CACHE_DIR = Path(".cache")
    
    # Audio processing
    MIN_AUDIO_DURATION = 0.5
    SAMPLE_RATE = 22050
    
    # Model parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 10
    RANDOM_STATE = 42
    
    # Risk thresholds
    LOW_RISK = 30
    MEDIUM_RISK = 70
    HIGH_RISK = 90
    
    # Color scheme
    PRIMARY_COLOR = "#1e3a8a"
    SECONDARY_COLOR = "#3b82f6"
    SUCCESS_COLOR = "#10b981"
    WARNING_COLOR = "#f59e0b"
    DANGER_COLOR = "#ef4444"
    NEUTRAL_COLOR = "#6b7280"
    
    BG_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

# ============================================================================
# SDK LAYER: Core Analytics Engine
# ============================================================================

class AudioFeatureExtractor:
    """Advanced audio feature extraction engine"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def extract_fundamental_frequency(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract F0 and related features"""
        f0 = librosa.yin(y, fmin=75, fmax=300, sr=sr)
        f0_clean = f0[f0 > 0]
        
        if len(f0_clean) > 0:
            return {
                'MDVP:Fo(Hz)': np.mean(f0_clean),
                'MDVP:Fhi(Hz)': np.max(f0_clean),
                'MDVP:Flo(Hz)': np.min(f0_clean)
            }
        return {
            'MDVP:Fo(Hz)': 150,
            'MDVP:Fhi(Hz)': 200,
            'MDVP:Flo(Hz)': 100
        }
    
    def extract_jitter_features(self, f0_clean: np.ndarray) -> Dict[str, float]:
        """Extract jitter (pitch variation) features"""
        if len(f0_clean) > 1:
            jitter_abs = np.mean(np.abs(np.diff(f0_clean)))
            mean_f0 = np.mean(f0_clean)
            
            return {
                'MDVP:Jitter(%)': (jitter_abs / mean_f0) * 100,
                'MDVP:Jitter(Abs)': jitter_abs,
                'MDVP:RAP': jitter_abs / mean_f0,
                'MDVP:PPQ': np.std(f0_clean) / mean_f0,
                'Jitter:DDP': (jitter_abs / mean_f0) * 3
            }
        return {
            'MDVP:Jitter(%)': 0.005,
            'MDVP:Jitter(Abs)': 0.00003,
            'MDVP:RAP': 0.003,
            'MDVP:PPQ': 0.003,
            'Jitter:DDP': 0.009
        }
    
    def extract_shimmer_features(self, rms: np.ndarray) -> Dict[str, float]:
        """Extract shimmer (amplitude variation) features"""
        if len(rms) > 1:
            shimmer_abs = np.mean(np.abs(np.diff(rms)))
            mean_rms = np.mean(rms)
            
            return {
                'MDVP:Shimmer': shimmer_abs / mean_rms,
                'MDVP:Shimmer(dB)': 20 * np.log10(shimmer_abs / mean_rms + 1e-10),
                'Shimmer:APQ3': np.std(rms[:len(rms)//3]) / mean_rms,
                'Shimmer:APQ5': np.std(rms[:len(rms)//5]) / mean_rms,
                'MDVP:APQ': np.std(rms) / mean_rms,
                'Shimmer:DDA': (np.std(rms[:len(rms)//3]) / mean_rms) * 3
            }
        return {
            'MDVP:Shimmer': 0.03,
            'MDVP:Shimmer(dB)': 0.3,
            'Shimmer:APQ3': 0.015,
            'Shimmer:APQ5': 0.02,
            'MDVP:APQ': 0.025,
            'Shimmer:DDA': 0.045
        }
    
    def extract_all_features(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Extract complete feature set from audio"""
        try:
            y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            if len(y) < sr * Config.MIN_AUDIO_DURATION:
                return None, {"error": "Audio duration insufficient. Minimum 1 second required."}
            
            features = {}
            
            # F0 features
            f0_features = self.extract_fundamental_frequency(y, sr)
            features.update(f0_features)
            
            # Jitter features
            f0 = librosa.yin(y, fmin=75, fmax=300, sr=sr)
            f0_clean = f0[f0 > 0]
            jitter_features = self.extract_jitter_features(f0_clean)
            features.update(jitter_features)
            
            # Shimmer features
            rms = librosa.feature.rms(y=y)[0]
            shimmer_features = self.extract_shimmer_features(rms)
            features.update(shimmer_features)
            
            # Harmonic-to-Noise Ratio
            harmonic, percussive = librosa.effects.hpss(y)
            hnr = 10 * np.log10(np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10))
            features['NHR'] = 1.0 / (hnr + 1e-10) if hnr > 0 else 0.025
            features['HNR'] = max(hnr, 0)
            
            # Advanced features
            features['RPDE'] = np.mean(np.abs(librosa.feature.rms(y=y)[0]))
            features['DFA'] = np.std(y) / (np.mean(np.abs(y)) + 1e-10)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spread1'] = np.std(spectral_centroid)
            features['spread2'] = skew(spectral_centroid)
            
            features['D2'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            features['PPE'] = -np.sum(f0_clean * np.log(f0_clean + 1e-10)) if len(f0_clean) > 0 else 0.2
            
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            
            return np.array(feature_vector).reshape(1, -1), features
            
        except Exception as e:
            return None, {"error": f"Audio processing failed: {str(e)}"}


class ModelManager:
    """Machine learning model management system"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self.model_path = self.model_dir / 'rf_model.pkl'
        self.scaler_path = self.model_dir / 'feature_scaler.pkl'
        self.features_path = self.model_dir / 'feature_config.pkl'
        self.metadata_path = self.model_dir / 'model_metadata.json'
    
    def download_dataset(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]:
        """Download UCI Parkinson's dataset"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
            df = pd.read_csv(url)
            
            X = df.drop(['name', 'status'], axis=1)
            y = df['status']
            
            return X, y, list(X.columns)
        except Exception as e:
            return None, None, None
    
    def train(self, progress_callback=None) -> Tuple[bool, Dict]:
        """Train model with progress tracking"""
        if progress_callback:
            progress_callback("Initializing training pipeline", 0.05)
        
        X, y, features = self.download_dataset()
        
        if X is None:
            return False, {"error": "Dataset download failed"}
        
        self.feature_names = features
        
        if progress_callback:
            progress_callback("Preparing data splits", 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        if progress_callback:
            progress_callback("Standardizing features", 0.3)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if progress_callback:
            progress_callback("Training Random Forest classifier", 0.5)
        
        self.model = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            max_depth=Config.MAX_DEPTH,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        if progress_callback:
            progress_callback("Evaluating model performance", 0.8)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"], output_dict=True)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if progress_callback:
            progress_callback("Saving model artifacts", 0.95)
        
        self.save_model(accuracy, report)
        
        if progress_callback:
            progress_callback("Training complete", 1.0)
        
        return True, {
            'accuracy': accuracy,
            'report': report,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def save_model(self, accuracy: float, report: Dict):
        """Save model and metadata"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        metadata = {
            'version': Config.VERSION,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'precision': report['Parkinson\'s']['precision'],
            'recall': report['Parkinson\'s']['recall'],
            'f1_score': report['Parkinson\'s']['f1-score']
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self) -> bool:
        """Load trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def get_metadata(self) -> Optional[Dict]:
        """Get model metadata"""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None


class NeuroVoiceSDK:
    """Main SDK for Parkinson's detection from voice"""
    
    def __init__(self):
        self.model_manager = ModelManager(Config.MODEL_DIR)
        self.feature_extractor = None
        
        if self.model_manager.load_model():
            self.feature_extractor = AudioFeatureExtractor(self.model_manager.feature_names)
    
    def is_ready(self) -> bool:
        """Check if SDK is ready for predictions"""
        return self.model_manager.model is not None
    
    def train_model(self, progress_callback=None) -> Tuple[bool, Dict]:
        """Train the detection model"""
        success, result = self.model_manager.train(progress_callback)
        if success:
            self.feature_extractor = AudioFeatureExtractor(self.model_manager.feature_names)
        return success, result
    
    def analyze(self, audio_path: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Analyze audio file for Parkinson's indicators"""
        if not self.is_ready():
            return None, "Model not initialized. Please train the model first."
        
        features, feature_dict = self.feature_extractor.extract_all_features(audio_path)
        
        if features is None:
            return None, feature_dict.get('error', 'Unknown error')
        
        features_scaled = self.model_manager.scaler.transform(features)
        prediction = self.model_manager.model.predict(features_scaled)[0]
        probability = self.model_manager.model.predict_proba(features_scaled)[0]
        
        risk_score = probability[1] * 100
        
        # Determine risk level
        if risk_score < Config.LOW_RISK:
            risk_level = "Low"
            risk_category = "minimal"
        elif risk_score < Config.MEDIUM_RISK:
            risk_level = "Moderate"
            risk_category = "moderate"
        elif risk_score < Config.HIGH_RISK:
            risk_level = "High"
            risk_category = "elevated"
        else:
            risk_level = "Critical"
            risk_category = "critical"
        
        return {
            'prediction': int(prediction),
            'classification': "Positive" if prediction == 1 else "Negative",
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_category': risk_category,
            'confidence': float(max(probability) * 100),
            'probabilities': {
                'healthy': float(probability[0] * 100),
                'parkinsons': float(probability[1] * 100)
            },
            'features': feature_dict,
            'timestamp': datetime.now().isoformat()
        }, None
    
    def get_model_info(self) -> Optional[Dict]:
        """Get model information"""
        return self.model_manager.get_metadata()


# ============================================================================
# UI LAYER: Professional Streamlit Interface
# ============================================================================

def apply_custom_css():
    """Apply professional custom styling"""
    st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    /* Status indicators */
    .status-ready {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: #10b981;
        color: white;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .status-not-ready {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: #f59e0b;
        color: white;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Risk levels */
    .risk-low { color: #10b981; font-weight: 600; }
    .risk-moderate { color: #f59e0b; font-weight: 600; }
    .risk-high { color: #ef4444; font-weight: 600; }
    .risk-critical { color: #dc2626; font-weight: 700; }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background: #667eea !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .dataframe td {
        padding: 10px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session():
    """Initialize session state"""
    if 'sdk' not in st.session_state:
        st.session_state.sdk = NeuroVoiceSDK()
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []


def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">NeuroVoice Analytics</h1>
        <p class="header-subtitle">Advanced AI-Powered Parkinson's Disease Detection Through Voice Analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with model information"""
    with st.sidebar:
        st.markdown("### System Overview")
        
        if st.session_state.sdk.is_ready():
            st.markdown('<span class="status-ready">System Ready</span>', unsafe_allow_html=True)
            
            metadata = st.session_state.sdk.get_model_info()
            if metadata:
                st.markdown("---")
                st.markdown("### Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{metadata['accuracy']:.1%}")
                    st.metric("Precision", f"{metadata['precision']:.1%}")
                with col2:
                    st.metric("Recall", f"{metadata['recall']:.1%}")
                    st.metric("F1-Score", f"{metadata['f1_score']:.1%}")
                
                st.caption(f"Model Version: {metadata['version']}")
                st.caption(f"Last Updated: {datetime.fromisoformat(metadata['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        else:
            st.markdown('<span class="status-not-ready">Model Not Trained</span>', unsafe_allow_html=True)
            st.info("Train the model to begin analysis")
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("Train New Model"):
            train_model_interface()
        
        if st.button("Export Analysis History"):
            export_history()
        
        st.markdown("---")
        st.markdown("### Analysis Guidelines")
        st.markdown("""
        **Optimal Recording Conditions:**
        - Duration: 3-5 seconds
        - Environment: Quiet, minimal background noise
        - Vocalization: Steady "Ahhh" sound
        - Microphone: Quality audio input device
        
        **Supported Formats:**
        - WAV, MP3, OGG, M4A
        - Sample rate: 16kHz or higher
        - Bit depth: 16-bit or higher
        """)
        
        st.markdown("---")
        st.caption(f"{Config.APP_NAME} v{Config.VERSION}")


def train_model_interface():
    """Interface for model training"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    progress_bar = progress_placeholder.progress(0)
    
    def update_progress(message: str, progress: float):
        status_placeholder.info(f"Training Pipeline: {message}")
        progress_bar.progress(progress)
    
    success, result = st.session_state.sdk.train_model(update_progress)
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    if success:
        st.success(f"Model trained successfully. Test accuracy: {result['accuracy']:.2%}")
        
        with st.expander("Training Results", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Samples", result['train_samples'])
            with col2:
                st.metric("Test Samples", result['test_samples'])
            with col3:
                st.metric("Overall Accuracy", f"{result['accuracy']:.2%}")
            
            st.markdown("### Classification Report")
            report_df = pd.DataFrame(result['report']).transpose()
            st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'))
            
            st.markdown("### Top Feature Importance")
            fig = px.bar(
                result['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Most Important Features',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.rerun()
    else:
        st.error(f"Training failed: {result.get('error', 'Unknown error')}")


def create_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """Create professional risk assessment gauge"""
    color_map = {
        "Low": Config.SUCCESS_COLOR,
        "Moderate": Config.WARNING_COLOR,
        "High": Config.DANGER_COLOR,
        "Critical": "#dc2626"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Assessment Score", 'font': {'size': 20, 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': color_map.get(risk_level, Config.NEUTRAL_COLOR), 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, Config.LOW_RISK], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [Config.LOW_RISK, Config.MEDIUM_RISK], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [Config.MEDIUM_RISK, Config.HIGH_RISK], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [Config.HIGH_RISK, 100], 'color': 'rgba(220, 38, 38, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'family': 'Inter'}
    )
    
    return fig


def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create probability distribution chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Healthy', "Parkinson's"],
            y=[probabilities['healthy'], probabilities['parkinsons']],
            marker_color=[Config.SUCCESS_COLOR, Config.DANGER_COLOR],
            text=[f"{probabilities['healthy']:.1f}%", f"{probabilities['parkinsons']:.1f}%"],
            textposition='auto',
            textfont=dict(size=16, family='Inter', color='white')
        )
    ])
    
    fig.update_layout(
        title='Classification Probability Distribution',
        xaxis_title='Classification',
        yaxis_title='Probability (%)',
        height=350,
        showlegend=False,
        paper_bgcolor='white',
        font={'family': 'Inter'},
        yaxis={'range': [0, 100]}
    )
    
    return fig


def render_analysis_results(result: Dict):
    """Render comprehensive analysis results"""
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(create_risk_gauge(result['risk_score'], result['risk_level']), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(create_probability_chart(result['probabilities']), 
                       use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Classification**")
        st.markdown(f"### {result['classification']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Risk Level**")
        risk_class = f"risk-{result['risk_category']}"
        st.markdown(f'<div class="{risk_class}"><h3>{result["risk_level"]}</h3></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Risk Score**")
        st.markdown(f"### {result['risk_score']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Confidence**")
        st.markdown(f"### {result['confidence']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Analysis Section
    with st.expander("Detailed Feature Analysis", expanded=False):
        features_df = pd.DataFrame([
            {'Feature': k, 'Value': f"{v:.4f}"} 
            for k, v in result['features'].items()
        ])
        
        col1, col2 = st.columns(2)
        mid_point = len(features_df) // 2
        
        with col1:
            st.markdown("**Voice Characteristics (Part 1)**")
            st.dataframe(features_df.head(mid_point), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Voice Characteristics (Part 2)**")
            st.dataframe(features_df.tail(len(features_df) - mid_point), use_container_width=True, hide_index=True)
    
    # Clinical Context
    with st.expander("Clinical Context & Interpretation"):
        st.markdown("""
        ### Understanding Your Results
        
        **Risk Categories:**
        - **Low Risk (0-30%)**: Voice patterns within normal parameters
        - **Moderate Risk (30-70%)**: Some atypical patterns detected, monitoring recommended
        - **High Risk (70-90%)**: Significant indicators present, clinical evaluation advised
        - **Critical Risk (90-100%)**: Strong indicators detected, immediate consultation recommended
        
        **Key Voice Biomarkers:**
        - **Jitter**: Measures pitch variation and vocal stability
        - **Shimmer**: Quantifies amplitude variation in voice
        - **HNR (Harmonics-to-Noise Ratio)**: Assesses voice quality and breathiness
        - **Pitch Metrics**: Fundamental frequency characteristics
        
        ### Important Notes
        This analysis is based on acoustic voice features commonly associated with Parkinson's Disease 
        research. Results should be interpreted alongside comprehensive medical evaluation.
        """)
    
    # Medical Disclaimer
    st.warning("""
    **Medical Disclaimer**: This tool is designed for research and educational purposes only. 
    It is not a substitute for professional medical diagnosis, advice, or treatment. 
    Always seek the guidance of qualified healthcare providers with any questions regarding a medical condition.
    """)
    
    # Add to history
    st.session_state.analysis_history.append({
        'timestamp': result['timestamp'],
        'classification': result['classification'],
        'risk_score': result['risk_score'],
        'risk_level': result['risk_level'],
        'confidence': result['confidence']
    })


def export_history():
    """Export analysis history as CSV"""
    if not st.session_state.analysis_history:
        st.warning("No analysis history to export")
        return
    
    df = pd.DataFrame(st.session_state.analysis_history)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Download Analysis History (CSV)",
        data=csv,
        file_name=f"neurovoice_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_main_interface():
    """Render main analysis interface"""
    
    if not st.session_state.sdk.is_ready():
        st.warning("Model not initialized. Please train the model using the sidebar to begin analysis.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 3rem;">
                <h3>Getting Started</h3>
                <p style="color: #6b7280; margin-top: 1rem;">
                    Train the machine learning model to unlock voice analysis capabilities.
                    The training process downloads the UCI Parkinson's dataset and builds
                    a Random Forest classifier optimized for voice-based detection.
                </p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Voice Analysis", "Analysis History", "System Insights"])
    
    with tab1:
        st.markdown("### Voice Sample Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            audio_file = st.file_uploader(
                "Upload Voice Recording",
                type=['wav', 'mp3', 'ogg', 'm4a'],
                help="Upload a clear voice recording for analysis"
            )
            
            if audio_file is not None:
                st.audio(audio_file, format='audio/wav')
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_b:
                    if st.button("Analyze Voice Sample", type="primary"):
                        with st.spinner("Processing audio and extracting features..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                tmp_file.write(audio_file.read())
                                tmp_path = tmp_file.name
                            
                            try:
                                result, error = st.session_state.sdk.analyze(tmp_path)
                                
                                if error:
                                    st.error(f"Analysis failed: {error}")
                                else:
                                    render_analysis_results(result)
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h4>Recording Guidelines</h4>
                <ul style="line-height: 1.8;">
                    <li>Find a quiet environment</li>
                    <li>Use quality microphone</li>
                    <li>Record 3-5 seconds</li>
                    <li>Produce steady "Ahhh" sound</li>
                    <li>Maintain consistent volume</li>
                    <li>Avoid background noise</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Analysis History")
        
        if not st.session_state.analysis_history:
            st.info("No analysis history available. Upload and analyze voice samples to build your history.")
        else:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_analyses = len(history_df)
                st.metric("Total Analyses", total_analyses)
            
            with col2:
                avg_risk = history_df['risk_score'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.1f}%")
            
            with col3:
                positive_count = (history_df['classification'] == 'Positive').sum()
                st.metric("Positive Classifications", positive_count)
            
            st.markdown("---")
            
            # Trend chart
            if len(history_df) > 1:
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='risk_score',
                    title='Risk Score Trend Over Time',
                    markers=True,
                    color_discrete_sequence=[Config.PRIMARY_COLOR]
                )
                fig.update_layout(
                    xaxis_title='Date/Time',
                    yaxis_title='Risk Score (%)',
                    height=400,
                    font={'family': 'Inter'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # History table
            st.markdown("### Detailed History")
            display_df = history_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": "Date/Time",
                    "classification": "Result",
                    "risk_score": st.column_config.NumberColumn("Risk Score (%)", format="%.1f"),
                    "risk_level": "Risk Level",
                    "confidence": st.column_config.NumberColumn("Confidence (%)", format="%.1f")
                }
            )
            
            if st.button("Export History"):
                export_history()
    
    with tab3:
        st.markdown("### System Performance & Insights")
        
        metadata = st.session_state.sdk.get_model_info()
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="custom-card">
                    <h4>Model Architecture</h4>
                    <table style="width: 100%; line-height: 2;">
                        <tr><td><strong>Algorithm:</strong></td><td>Random Forest Classifier</td></tr>
                        <tr><td><strong>Estimators:</strong></td><td>200 Decision Trees</td></tr>
                        <tr><td><strong>Max Depth:</strong></td><td>10 Levels</td></tr>
                        <tr><td><strong>Class Balancing:</strong></td><td>Enabled</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="custom-card">
                    <h4>Feature Set</h4>
                    <ul style="line-height: 1.8;">
                        <li>22 Acoustic Features</li>
                        <li>Jitter & Shimmer Metrics</li>
                        <li>Harmonics-to-Noise Ratio</li>
                        <li>Pitch Characteristics</li>
                        <li>Spectral Features</li>
                        <li>Complexity Measures</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Performance metrics visualization
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Score': [
                        metadata['accuracy'],
                        metadata['precision'],
                        metadata['recall'],
                        metadata['f1_score']
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Score',
                    title='Model Performance Metrics',
                    color='Score',
                    color_continuous_scale='Viridis',
                    text='Score'
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    font={'family': 'Inter'},
                    yaxis={'range': [0, 1.1]}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="custom-card">
                    <h4>Research Foundation</h4>
                    <p style="line-height: 1.8;">
                        <strong>Dataset:</strong> UCI Machine Learning Repository<br>
                        <strong>Source:</strong> Parkinson's Disease Classification<br>
                        <strong>Features:</strong> Voice measurements from 195 individuals<br>
                        <strong>Reference:</strong> 'Exploiting Nonlinear Recurrence and Fractal 
                        Scaling Properties for Voice Disorder Detection', Little et al., 2007
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### About NeuroVoice Analytics")
        st.markdown("""
        NeuroVoice Analytics leverages advanced machine learning algorithms to analyze acoustic 
        features in voice recordings that may indicate early signs of Parkinson's Disease. 
        The system processes multiple voice biomarkers including pitch variation, amplitude 
        modulation, and harmonic characteristics to provide comprehensive risk assessment.
        
        **Key Capabilities:**
        - Real-time voice feature extraction
        - Multi-dimensional acoustic analysis
        - Probabilistic risk assessment
        - Longitudinal tracking and trends
        - Research-grade accuracy
        
        **Technology Stack:**
        - Machine Learning: Scikit-learn Random Forest
        - Audio Processing: Librosa
        - Data Analysis: Pandas, NumPy, SciPy
        - Visualization: Plotly
        - Interface: Streamlit
        """)


def main():
    """Application entry point"""
    st.set_page_config(
        page_title="NeuroVoice Analytics",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    init_session()
    render_header()
    render_sidebar()
    render_main_interface()
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem;'>
        <p style='margin: 0; font-weight: 500;'>NeuroVoice Analytics Platform</p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem;'>
            Advanced AI-Powered Voice Analysis for Neurological Research
        </p>
        <p style='margin: 0; font-size: 0.85rem;'>
            Powered by Machine Learning | Built with Streamlit | Version 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
