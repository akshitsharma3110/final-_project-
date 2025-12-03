from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
from audio_preprocessor import AudioPreprocessor

app = Flask(__name__)

class DisasterClassificationAPI:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.scaler = None
        self.model = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Load the best performing model (adjust based on your results)
            self.model = tf.keras.models.load_model('saved_models/cnn_model.h5')
            self.preprocessor.load_preprocessor('saved_models/label_encoder.pkl')
            self.scaler = joblib.load('saved_models/scaler.pkl')
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_disaster(self, audio_file_path):
        """Predict disaster type from audio file"""
        try:
            # Extract features
            features = self.preprocessor.extract_features(audio_file_path)
            if features is None:
                return None, "Error processing audio file"
            
            # Prepare traditional features
            feature_vector = []
            feature_vector.extend(features['mfcc_mean'])
            feature_vector.extend(features['mfcc_std'])
            feature_vector.append(features['spectral_centroid_mean'])
            feature_vector.append(features['spectral_centroid_std'])
            feature_vector.append(features['spectral_rolloff_mean'])
            feature_vector.append(features['spectral_rolloff_std'])
            feature_vector.append(features['zcr_mean'])
            feature_vector.append(features['zcr_std'])
            feature_vector.extend(features['chroma_mean'])
            feature_vector.extend(features['chroma_std'])
            
            # Scale traditional features
            traditional_features = self.scaler.transform([feature_vector])
            
            # Prepare mel spectrogram
            mel_spec = features['mel_spec']
            mel_features = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
            
            # Make prediction
            prediction = self.model.predict(mel_features)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            
            # Get class name
            class_name = self.preprocessor.label_encoder.inverse_transform([predicted_class])[0]
            
            return {
                'disaster_type': class_name,
                'confidence': confidence,
                'all_probabilities': {
                    class_name: float(prob) for class_name, prob in 
                    zip(self.preprocessor.label_encoder.classes_, prediction[0])
                }
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

# Initialize API
api = DisasterClassificationAPI()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        # Make prediction
        result, error = api.predict_disaster(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': api.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5003)