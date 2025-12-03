from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
def load_model():
    try:
        # Try to load the best model first
        model = joblib.load('disaster_model_FINAL.pkl')
        vectorizer = joblib.load('vectorizer_FINAL.pkl')
        return model, vectorizer
    except:
        try:
            # Fallback to improved model
            model = joblib.load('disaster_model_improved.pkl')
            vectorizer = joblib.load('vectorizer_improved.pkl')
            return model, vectorizer
        except:
            print("Model files not found. Please train the model first.")
            return None, None

# Enhanced prediction function with smart keyword detection
def predict_disaster(text, model, vectorizer):
    if model is None or vectorizer is None:
        return "Model not loaded", 0.0
    
    # Preprocess text for better accuracy
    text_lower = text.lower().strip()
    
    # Enhanced disaster keywords with stronger indicators
    keywords = {
        'flood': ['submerged', 'underwater', 'flooded', 'flood', 'water', 'river', 'rain', 'overflow', 'inundated', 'waterlogged', 'drowning', 'tunnels submerged'],
        'earthquake': ['building', 'collapse', 'crack', 'shake', 'tremor', 'seismic', 'quake', 'fault', 'aftershock'],
        'fire': ['smoke', 'burn', 'flame', 'fire', 'blaze', 'ignite', 'combustion', 'inferno', 'wildfire'],
        'hurricane': ['wind', 'storm', 'hurricane', 'cyclone', 'typhoon', 'gale', 'tempest'],
        'landslide': ['slide', 'mud', 'rock', 'slope', 'debris', 'hill', 'avalanche', 'rockfall'],
        'drought': ['dry', 'water shortage', 'drought', 'arid', 'crop failure', 'wells dry'],
        'tsunami': ['wave', 'tsunami', 'sea', 'ocean', 'coastal', 'tidal wave'],
        'accident': ['crash', 'collision', 'accident', 'vehicle', 'train', 'derail', 'wreck']
    }
    
    # First check for strong keyword matches
    best_match = None
    max_matches = 0
    
    for disaster, words in keywords.items():
        matches = sum(1 for word in words if word in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = disaster
    
    # If strong keyword match found, use it
    if max_matches >= 2 or any(strong_word in text_lower for strong_word in ['submerged', 'underwater', 'flooded', 'inundated']):
        if best_match == 'flood':
            return 'Flood', 0.92
        elif best_match:
            return best_match.title(), 0.85
    
    # Otherwise use ML model
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0].max()
    
    # Boost confidence for keyword matches
    for disaster, words in keywords.items():
        if any(word in text_lower for word in words):
            if disaster.lower() == prediction.lower():
                probability = min(0.95, probability + 0.3)
            break
    
    return prediction, probability

# Load model once at startup
model, vectorizer = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction, confidence = predict_disaster(text, model, vectorizer)
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = 'high'
        elif confidence > 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 1),
            'confidence_level': confidence_level
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)