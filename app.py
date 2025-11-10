"""
Parkinson's Disease Detection from Voice - Complete MVP
Flask web application with model training and real-time prediction
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import librosa
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

def download_and_prepare_data():
    """Download and prepare the UCI Parkinson's dataset"""
    try:
        # UCI Parkinson's Dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        df = pd.read_csv(url)
        
        # Separate features and target
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        return X, y, list(X.columns)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None, None

def train_model():
    """Train the Random Forest model on Parkinson's dataset"""
    global model, scaler, feature_names
    
    print("Downloading and preparing data...")
    X, y, features = download_and_prepare_data()
    
    if X is None:
        print("Failed to load data. Using dummy model.")
        return False
    
    feature_names = features
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))
    
    # Save model and scaler
    with open('parkinsons_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model trained and saved successfully!")
    return True

def load_model():
    """Load pre-trained model and scaler"""
    global model, scaler, feature_names
    
    try:
        with open('parkinsons_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        return train_model()

def extract_features(audio_path):
    """Extract voice features from audio file compatible with training dataset"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Ensure minimum duration
        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return None, "Audio too short. Please record at least 1 second."
        
        # Extract features
        features = {}
        
        # 1. Fundamental Frequency (F0) statistics
        f0 = librosa.yin(y, fmin=75, fmax=300, sr=sr)
        f0_clean = f0[f0 > 0]
        if len(f0_clean) > 0:
            features['MDVP:Fo(Hz)'] = np.mean(f0_clean)
            features['MDVP:Fhi(Hz)'] = np.max(f0_clean)
            features['MDVP:Flo(Hz)'] = np.min(f0_clean)
        else:
            features['MDVP:Fo(Hz)'] = 150
            features['MDVP:Fhi(Hz)'] = 200
            features['MDVP:Flo(Hz)'] = 100
        
        # 2. Jitter measures (pitch variation)
        if len(f0_clean) > 1:
            jitter_abs = np.mean(np.abs(np.diff(f0_clean)))
            features['MDVP:Jitter(%)'] = (jitter_abs / np.mean(f0_clean)) * 100
            features['MDVP:Jitter(Abs)'] = jitter_abs
            features['MDVP:RAP'] = np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean)
            features['MDVP:PPQ'] = np.std(f0_clean) / np.mean(f0_clean)
            features['Jitter:DDP'] = features['MDVP:RAP'] * 3
        else:
            features['MDVP:Jitter(%)'] = 0.005
            features['MDVP:Jitter(Abs)'] = 0.00003
            features['MDVP:RAP'] = 0.003
            features['MDVP:PPQ'] = 0.003
            features['Jitter:DDP'] = 0.009
        
        # 3. Shimmer measures (amplitude variation)
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            shimmer_abs = np.mean(np.abs(np.diff(rms)))
            features['MDVP:Shimmer'] = shimmer_abs / np.mean(rms)
            features['MDVP:Shimmer(dB)'] = 20 * np.log10(shimmer_abs / np.mean(rms) + 1e-10)
            features['Shimmer:APQ3'] = np.std(rms[:len(rms)//3]) / np.mean(rms)
            features['Shimmer:APQ5'] = np.std(rms[:len(rms)//5]) / np.mean(rms)
            features['MDVP:APQ'] = np.std(rms) / np.mean(rms)
            features['Shimmer:DDA'] = features['Shimmer:APQ3'] * 3
        else:
            features['MDVP:Shimmer'] = 0.03
            features['MDVP:Shimmer(dB)'] = 0.3
            features['Shimmer:APQ3'] = 0.015
            features['Shimmer:APQ5'] = 0.02
            features['MDVP:APQ'] = 0.025
            features['Shimmer:DDA'] = 0.045
        
        # 4. Harmonics-to-Noise Ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = 10 * np.log10(np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10))
        features['NHR'] = 1.0 / (hnr + 1e-10) if hnr > 0 else 0.025
        features['HNR'] = max(hnr, 0)
        
        # 5. RPDE (Recurrence Period Density Entropy) - approximation
        features['RPDE'] = np.mean(np.abs(librosa.feature.rms(y=y)[0]))
        
        # 6. DFA (Detrended Fluctuation Analysis) - approximation
        features['DFA'] = np.std(y) / (np.mean(np.abs(y)) + 1e-10)
        
        # 7. Spread metrics
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spread1'] = np.std(spectral_centroid)
        features['spread2'] = skew(spectral_centroid)
        
        # 8. D2 (Correlation Dimension) - approximation
        features['D2'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # 9. PPE (Pitch Period Entropy) - approximation
        features['PPE'] = -np.sum(f0_clean * np.log(f0_clean + 1e-10)) if len(f0_clean) > 0 else 0.2
        
        # Create feature vector in correct order
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        
        return np.array(feature_vector).reshape(1, -1), None
        
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        # Extract features
        features, error = extract_features(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get risk score (probability of Parkinson's)
        risk_score = probability[1] * 100
        
        result = {
            'prediction': int(prediction),
            'label': 'Parkinson\'s Risk Detected' if prediction == 1 else 'Healthy',
            'risk_score': float(risk_score),
            'confidence': float(max(probability) * 100)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load or train model
    load_model()
    
    # Run Flask app
    print("\n" + "="*50)
    print("Parkinson's Detection App Running!")
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
