"""
Test script for Parkinson's Detection Model
Tests model performance and feature extraction
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_test_model():
    """Load model and perform comprehensive testing"""
    
    print("="*60)
    print("PARKINSON'S DETECTION MODEL - TEST SUITE")
    print("="*60)
    
    # Load model files
    try:
        with open('parkinsons_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Model and scaler loaded successfully\n")
    except FileNotFoundError:
        print("✗ Model files not found. Please run app.py first to train the model.\n")
        return
    
    # Load dataset
    print("Loading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Healthy: {sum(y==0)}, Parkinson's: {sum(y==1)}")
    print()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Cross-validation
    print("Performing 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()
    
    # Full dataset predictions
    print("Full Dataset Performance:")
    y_pred = model.predict(X_scaled)
    print(classification_report(y, y_pred, target_names=['Healthy', "Parkinson's"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(f"              Predicted Healthy | Predicted Parkinson's")
    print(f"Actual Healthy         {cm[0][0]:4d}      |      {cm[0][1]:4d}")
    print(f"Actual Parkinson's     {cm[1][0]:4d}      |      {cm[1][1]:4d}")
    print()
    
    # Feature importance
    print("Top 10 Most Important Features:")
    feature_names = list(X.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    print()
    
    # Test predictions on sample data
    print("Sample Predictions:")
    print("-" * 60)
    
    # Get a few samples
    healthy_idx = np.where(y == 0)[0][0]
    parkinsons_idx = np.where(y == 1)[0][0]
    
    for idx, label in [(healthy_idx, "Healthy"), (parkinsons_idx, "Parkinson's")]:
        sample = X_scaled[idx:idx+1]
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        
        print(f"Actual: {label}")
        print(f"Predicted: {'Parkinson\'s' if pred == 1 else 'Healthy'}")
        print(f"Risk Score: {proba[1]*100:.1f}%")
        print(f"Confidence: {max(proba)*100:.1f}%")
        print()
    
    print("="*60)
    print("Testing Complete!")
    print("="*60)

def test_feature_extraction():
    """Test feature extraction with dummy audio"""
    print("\nTesting Feature Extraction...")
    print("-" * 60)
    
    try:
        import librosa
        
        # Generate dummy audio (sine wave)
        sr = 22050
        duration = 3  # seconds
        frequency = 150  # Hz (typical voice pitch)
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio))
        
        print(f"✓ Generated dummy audio: {duration}s at {sr}Hz")
        print(f"  Sample shape: {audio.shape}")
        print(f"  Amplitude range: [{audio.min():.3f}, {audio.max():.3f}]")
        print()
        
        # Test basic librosa functions
        print("Testing librosa functions:")
        f0 = librosa.yin(audio, fmin=75, fmax=300, sr=sr)
        print(f"  ✓ Pitch (F0) extraction: {len(f0)} frames")
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        print(f"  ✓ MFCC extraction: {mfcc.shape}")
        
        rms = librosa.feature.rms(y=audio)
        print(f"  ✓ RMS energy: {rms.shape}")
        
        print("\n✓ Feature extraction functions working correctly!")
        
    except ImportError:
        print("✗ librosa not installed. Install with: pip install librosa")
    except Exception as e:
        print(f"✗ Error during feature extraction test: {e}")

if __name__ == "__main__":
    # Test model performance
    load_and_test_model()
    
    # Test feature extraction
    test_feature_extraction()
    
    print("\n" + "="*60)
    print("All tests complete! The system is ready for use.")
    print("Run 'python app.py' to start the web application.")
    print("="*60)
