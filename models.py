import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

class DisasterClassificationModels:
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
    
    def create_cnn_model(self, input_shape=(128, 130, 1)):
        """Create CNN model for mel-spectrogram input"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_rnn_model(self, input_shape=(130, 128)):
        """Create RNN model for sequential mel-spectrogram input"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_hybrid_model(self, mel_input_shape=(128, 130, 1), traditional_input_shape=(56,)):
        """Create hybrid CNN-RNN model with traditional features"""
        # Mel-spectrogram branch (CNN)
        mel_input = layers.Input(shape=mel_input_shape, name='mel_input')
        x1 = layers.Conv2D(32, (3, 3), activation='relu')(mel_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(128, activation='relu')(x1)
        
        # Traditional features branch
        trad_input = layers.Input(shape=traditional_input_shape, name='traditional_input')
        x2 = layers.Dense(64, activation='relu')(trad_input)
        x2 = layers.Dropout(0.3)(x2)
        x2 = layers.Dense(32, activation='relu')(x2)
        
        # Combine branches
        combined = layers.concatenate([x1, x2])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        output = layers.Dense(self.num_classes, activation='softmax')(combined)
        
        model = models.Model(inputs=[mel_input, trad_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_traditional_ml_models(self):
        """Create traditional ML models"""
        models_dict = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        return models_dict
    
    def create_wavenet_inspired_model(self, input_shape=(66150, 1)):
        """Create WaveNet-inspired model for raw audio"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Dilated convolutions
        x = layers.Conv1D(32, 3, dilation_rate=1, padding='causal', activation='relu')(model.input)
        x = layers.Conv1D(32, 3, dilation_rate=2, padding='causal', activation='relu')(x)
        x = layers.Conv1D(64, 3, dilation_rate=4, padding='causal', activation='relu')(x)
        x = layers.Conv1D(64, 3, dilation_rate=8, padding='causal', activation='relu')(x)
        x = layers.Conv1D(128, 3, dilation_rate=16, padding='causal', activation='relu')(x)
        
        # Global pooling and dense layers
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=model.input, outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model