import numpy as np
import librosa
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class AccuracyImprovementTechniques:
    def __init__(self, preprocessor, model_builder):
        self.preprocessor = preprocessor
        self.model_builder = model_builder
    
    def data_augmentation(self, audio_data, sr=22050):
        """Apply data augmentation techniques"""
        augmented_data = []
        
        # Original
        augmented_data.append(audio_data)
        
        # Time stretching
        stretched = librosa.effects.time_stretch(audio_data, rate=0.9)
        augmented_data.append(stretched)
        
        stretched = librosa.effects.time_stretch(audio_data, rate=1.1)
        augmented_data.append(stretched)
        
        # Pitch shifting
        pitched = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=2)
        augmented_data.append(pitched)
        
        pitched = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=-2)
        augmented_data.append(pitched)
        
        # Add noise
        noise = np.random.normal(0, 0.005, audio_data.shape)
        noisy = audio_data + noise
        augmented_data.append(noisy)
        
        # Volume adjustment
        louder = audio_data * 1.2
        augmented_data.append(louder)
        
        quieter = audio_data * 0.8
        augmented_data.append(quieter)
        
        return augmented_data
    
    def create_ensemble_model(self, input_shapes):
        """Create ensemble of different architectures"""
        # CNN branch
        cnn_input = layers.Input(shape=input_shapes['mel'], name='cnn_input')
        cnn_x = layers.Conv2D(64, (3, 3), activation='relu')(cnn_input)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.MaxPooling2D((2, 2))(cnn_x)
        cnn_x = layers.Conv2D(128, (3, 3), activation='relu')(cnn_x)
        cnn_x = layers.BatchNormalization()(cnn_x)
        cnn_x = layers.GlobalAveragePooling2D()(cnn_x)
        cnn_x = layers.Dense(128, activation='relu')(cnn_x)
        
        # RNN branch (reshaped mel input)
        rnn_input = layers.Input(shape=input_shapes['rnn'], name='rnn_input')
        rnn_x = layers.LSTM(64, return_sequences=True)(rnn_input)
        rnn_x = layers.LSTM(32)(rnn_x)
        rnn_x = layers.Dense(64, activation='relu')(rnn_x)
        
        # Traditional features branch
        trad_input = layers.Input(shape=input_shapes['traditional'], name='trad_input')
        trad_x = layers.Dense(64, activation='relu')(trad_input)
        trad_x = layers.Dropout(0.3)(trad_x)
        trad_x = layers.Dense(32, activation='relu')(trad_x)
        
        # Attention mechanism
        attention_input = layers.concatenate([cnn_x, rnn_x, trad_x])
        attention = layers.Dense(224, activation='tanh')(attention_input)
        attention = layers.Dense(224, activation='softmax')(attention)
        attended = layers.multiply([attention_input, attention])
        
        # Final layers
        combined = layers.Dense(128, activation='relu')(attended)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        output = layers.Dense(7, activation='softmax')(combined)
        
        model = models.Model(
            inputs=[cnn_input, rnn_input, trad_input], 
            outputs=output
        )
        
        # Custom optimizer with learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for traditional ML models"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        # SVM tuning
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svm = SVC(random_state=42, probability=True)
        svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        
        return {
            'best_rf': rf_grid.best_estimator_,
            'best_svm': svm_grid.best_estimator_,
            'rf_score': rf_grid.best_score_,
            'svm_score': svm_grid.best_score_
        }
    
    def create_voting_ensemble(self, models):
        """Create voting ensemble from multiple models"""
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['best_rf']),
                ('svm', models['best_svm'])
            ],
            voting='soft'
        )
        return voting_clf
    
    def advanced_feature_engineering(self, audio_path):
        """Extract advanced audio features"""
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        features = {}
        
        # Spectral features
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Rhythm features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['beat_count'] = len(beats)
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonic_energy'] = np.sum(y_harmonic**2)
        features['percussive_energy'] = np.sum(y_percussive**2)
        
        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        features['tonnetz_std'] = np.std(tonnetz, axis=1)
        
        # Poly features
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        features['poly_mean'] = np.mean(poly_features, axis=1)
        features['poly_std'] = np.std(poly_features, axis=1)
        
        return features
    
    def cross_validation_evaluation(self, model, X, y, cv=5):
        """Perform cross-validation evaluation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'scores': scores
        }
    
    def learning_curve_analysis(self, model, X, y):
        """Analyze learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), 
                        alpha=0.1)
        plt.fill_between(train_sizes, 
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), 
                        alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()
    
    def feature_importance_analysis(self, model, feature_names):
        """Analyze feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
            
            return importances[indices], [feature_names[i] for i in indices]
        else:
            print("Model doesn't support feature importance analysis")
            return None, None

def improve_model_accuracy(trainer, data_splits):
    """Main function to apply accuracy improvement techniques"""
    improvement = AccuracyImprovementTechniques(trainer.preprocessor, trainer.model_builder)
    
    print("Applying accuracy improvement techniques...")
    
    # 1. Hyperparameter tuning for traditional models
    print("1. Hyperparameter tuning...")
    tuned_models = improvement.hyperparameter_tuning(
        data_splits['X_train_traditional'], 
        data_splits['y_train']
    )
    
    # 2. Create voting ensemble
    print("2. Creating voting ensemble...")
    voting_ensemble = improvement.create_voting_ensemble(tuned_models)
    voting_ensemble.fit(data_splits['X_train_traditional'], data_splits['y_train'])
    
    # 3. Cross-validation evaluation
    print("3. Cross-validation evaluation...")
    cv_results = improvement.cross_validation_evaluation(
        voting_ensemble, 
        data_splits['X_train_traditional'], 
        data_splits['y_train']
    )
    print(f"CV Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    # 4. Learning curve analysis
    print("4. Learning curve analysis...")
    improvement.learning_curve_analysis(
        voting_ensemble, 
        data_splits['X_train_traditional'], 
        data_splits['y_train']
    )
    
    # 5. Feature importance analysis
    print("5. Feature importance analysis...")
    feature_names = [f'feature_{i}' for i in range(data_splits['X_train_traditional'].shape[1])]
    improvement.feature_importance_analysis(tuned_models['best_rf'], feature_names)
    
    return voting_ensemble, tuned_models, cv_results