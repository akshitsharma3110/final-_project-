import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os

from audio_preprocessor import AudioPreprocessor
from models import DisasterClassificationModels

class DisasterClassificationTrainer:
    def __init__(self, data_dir='audio_dataset'):
        self.data_dir = data_dir
        self.preprocessor = AudioPreprocessor()
        self.model_builder = DisasterClassificationModels()
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Prepare and split the dataset"""
        print("Extracting features from audio files...")
        features_list, labels = self.preprocessor.prepare_dataset(self.data_dir)
        
        print(f"Total samples: {len(features_list)}")
        print(f"Classes: {set(labels)}")
        
        # Split data
        data_splits = self.preprocessor.split_data(features_list)
        
        # Scale traditional features
        data_splits['X_train_traditional'] = self.scaler.fit_transform(data_splits['X_train_traditional'])
        data_splits['X_val_traditional'] = self.scaler.transform(data_splits['X_val_traditional'])
        data_splits['X_test_traditional'] = self.scaler.transform(data_splits['X_test_traditional'])
        
        # Reshape mel spectrograms for CNN
        mel_shape = data_splits['X_train_mel'][0].shape
        data_splits['X_train_mel'] = data_splits['X_train_mel'].reshape(-1, mel_shape[0], mel_shape[1], 1)
        data_splits['X_val_mel'] = data_splits['X_val_mel'].reshape(-1, mel_shape[0], mel_shape[1], 1)
        data_splits['X_test_mel'] = data_splits['X_test_mel'].reshape(-1, mel_shape[0], mel_shape[1], 1)
        
        return data_splits
    
    def train_cnn_model(self, data_splits):
        """Train CNN model"""
        print("Training CNN model...")
        model = self.model_builder.create_cnn_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            data_splits['X_train_mel'], data_splits['y_train'],
            validation_data=(data_splits['X_val_mel'], data_splits['y_val']),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_rnn_model(self, data_splits):
        """Train RNN model"""
        print("Training RNN model...")
        # Reshape for RNN (samples, timesteps, features)
        mel_shape = data_splits['X_train_mel'].shape
        X_train_rnn = data_splits['X_train_mel'].reshape(-1, mel_shape[2], mel_shape[1])
        X_val_rnn = data_splits['X_val_mel'].reshape(-1, mel_shape[2], mel_shape[1])
        
        model = self.model_builder.create_rnn_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_rnn_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X_train_rnn, data_splits['y_train'],
            validation_data=(X_val_rnn, data_splits['y_val']),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_hybrid_model(self, data_splits):
        """Train hybrid model"""
        print("Training Hybrid model...")
        model = self.model_builder.create_hybrid_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_hybrid_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            [data_splits['X_train_mel'], data_splits['X_train_traditional']], 
            data_splits['y_train'],
            validation_data=([data_splits['X_val_mel'], data_splits['X_val_traditional']], 
                           data_splits['y_val']),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_traditional_models(self, data_splits):
        """Train traditional ML models"""
        print("Training traditional ML models...")
        models = self.model_builder.create_traditional_ml_models()
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(data_splits['X_train_traditional'], data_splits['y_train'])
            trained_models[name] = model
            
        return trained_models
    
    def evaluate_model(self, model, data_splits, model_type='deep'):
        """Evaluate model performance"""
        if model_type == 'cnn':
            y_pred = model.predict(data_splits['X_test_mel'])
            y_pred_classes = np.argmax(y_pred, axis=1)
        elif model_type == 'rnn':
            mel_shape = data_splits['X_test_mel'].shape
            X_test_rnn = data_splits['X_test_mel'].reshape(-1, mel_shape[2], mel_shape[1])
            y_pred = model.predict(X_test_rnn)
            y_pred_classes = np.argmax(y_pred, axis=1)
        elif model_type == 'hybrid':
            y_pred = model.predict([data_splits['X_test_mel'], data_splits['X_test_traditional']])
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:  # traditional ML
            y_pred_classes = model.predict(data_splits['X_test_traditional'])
        
        accuracy = accuracy_score(data_splits['y_test'], y_pred_classes)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(data_splits['y_test'], y_pred_classes, 
                                  target_names=self.preprocessor.label_encoder.classes_))
        
        return accuracy, y_pred_classes
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.preprocessor.label_encoder.classes_,
                   yticklabels=self.preprocessor.label_encoder.classes_)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.show()
    
    def plot_training_history(self, history, title="Training History"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.show()
    
    def save_models(self, models_dict):
        """Save trained models"""
        os.makedirs('saved_models', exist_ok=True)
        
        for name, model in models_dict.items():
            if hasattr(model, 'save'):  # Deep learning models
                model.save(f'saved_models/{name}_model.h5')
            else:  # Traditional ML models
                joblib.dump(model, f'saved_models/{name}_model.pkl')
        
        # Save preprocessor
        self.preprocessor.save_preprocessor('saved_models/label_encoder.pkl')
        joblib.dump(self.scaler, 'saved_models/scaler.pkl')
        
        print("Models saved successfully!")

def main():
    # Initialize trainer
    trainer = DisasterClassificationTrainer()
    
    # Prepare data
    data_splits = trainer.prepare_data()
    
    # Train models
    results = {}
    
    # CNN Model
    cnn_model, cnn_history = trainer.train_cnn_model(data_splits)
    cnn_accuracy, cnn_pred = trainer.evaluate_model(cnn_model, data_splits, 'cnn')
    results['CNN'] = {'model': cnn_model, 'accuracy': cnn_accuracy, 'predictions': cnn_pred}
    trainer.plot_training_history(cnn_history, "CNN Training History")
    trainer.plot_confusion_matrix(data_splits['y_test'], cnn_pred, "CNN Confusion Matrix")
    
    # RNN Model
    rnn_model, rnn_history = trainer.train_rnn_model(data_splits)
    rnn_accuracy, rnn_pred = trainer.evaluate_model(rnn_model, data_splits, 'rnn')
    results['RNN'] = {'model': rnn_model, 'accuracy': rnn_accuracy, 'predictions': rnn_pred}
    trainer.plot_training_history(rnn_history, "RNN Training History")
    trainer.plot_confusion_matrix(data_splits['y_test'], rnn_pred, "RNN Confusion Matrix")
    
    # Hybrid Model
    hybrid_model, hybrid_history = trainer.train_hybrid_model(data_splits)
    hybrid_accuracy, hybrid_pred = trainer.evaluate_model(hybrid_model, data_splits, 'hybrid')
    results['Hybrid'] = {'model': hybrid_model, 'accuracy': hybrid_accuracy, 'predictions': hybrid_pred}
    trainer.plot_training_history(hybrid_history, "Hybrid Training History")
    trainer.plot_confusion_matrix(data_splits['y_test'], hybrid_pred, "Hybrid Confusion Matrix")
    
    # Traditional ML Models
    traditional_models = trainer.train_traditional_models(data_splits)
    for name, model in traditional_models.items():
        accuracy, pred = trainer.evaluate_model(model, data_splits, 'traditional')
        results[name] = {'model': model, 'accuracy': accuracy, 'predictions': pred}
        trainer.plot_confusion_matrix(data_splits['y_test'], pred, f"{name} Confusion Matrix")
    
    # Compare results
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    # Save best models
    models_to_save = {name: result['model'] for name, result in results.items()}
    trainer.save_models(models_to_save)

if __name__ == "__main__":
    main()