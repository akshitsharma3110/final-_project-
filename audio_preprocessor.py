import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_length = int(sample_rate * duration)
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(y) < self.max_length:
                y = np.pad(y, (0, self.max_length - len(y)))
            else:
                y = y[:self.max_length]
            
            # Extract features
            features = {}
            
            # MFCC features (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            features['mel_spec'] = mel_spec
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def prepare_dataset(self, data_dir):
        """Prepare dataset from directory structure"""
        features_list = []
        labels = []
        
        disaster_types = ['cyclone', 'earthquake', 'explosion', 'fire', 'flood', 'landslide', 'thunderstorm']
        
        for disaster_type in disaster_types:
            disaster_path = os.path.join(data_dir, disaster_type)
            if os.path.exists(disaster_path):
                for file in os.listdir(disaster_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(disaster_path, file)
                        features = self.extract_features(file_path)
                        if features is not None:
                            # Flatten features for traditional ML
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
                            
                            features_list.append({
                                'features': feature_vector,
                                'mel_spec': features['mel_spec'],
                                'label': disaster_type
                            })
                            labels.append(disaster_type)
        
        return features_list, labels
    
    def split_data(self, features_list, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # Extract features and labels
        X_traditional = np.array([item['features'] for item in features_list])
        X_mel = np.array([item['mel_spec'] for item in features_list])
        y = np.array([item['label'] for item in features_list])
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train_trad, X_temp_trad, X_train_mel, X_temp_mel, y_train, y_temp = train_test_split(
            X_traditional, X_mel, y_encoded, test_size=test_size + val_size, random_state=42, stratify=y_encoded
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val_trad, X_test_trad, X_val_mel, X_test_mel, y_val, y_test = train_test_split(
            X_temp_trad, X_temp_mel, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
        )
        
        return {
            'X_train_traditional': X_train_trad,
            'X_val_traditional': X_val_trad,
            'X_test_traditional': X_test_trad,
            'X_train_mel': X_train_mel,
            'X_val_mel': X_val_mel,
            'X_test_mel': X_test_mel,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def save_preprocessor(self, filepath):
        """Save label encoder"""
        joblib.dump(self.label_encoder, filepath)
    
    def load_preprocessor(self, filepath):
        """Load label encoder"""
        self.label_encoder = joblib.load(filepath)