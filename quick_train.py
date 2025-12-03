import numpy as np
from audio_preprocessor import AudioPreprocessor
from models import DisasterClassificationModels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def quick_train():
    print("Starting quick training...")
    
    # Initialize
    preprocessor = AudioPreprocessor()
    model_builder = DisasterClassificationModels()
    scaler = StandardScaler()
    
    # Prepare data
    print("Extracting features...")
    features_list, labels = preprocessor.prepare_dataset('audio_dataset')
    data_splits = preprocessor.split_data(features_list)
    
    # Scale traditional features
    data_splits['X_train_traditional'] = scaler.fit_transform(data_splits['X_train_traditional'])
    data_splits['X_test_traditional'] = scaler.transform(data_splits['X_test_traditional'])
    
    # Get mel-spectrogram shape
    mel_shape = data_splits['X_train_mel'][0].shape
    print(f"Mel-spectrogram shape: {mel_shape}")
    
    # Reshape for CNN
    data_splits['X_train_mel'] = data_splits['X_train_mel'].reshape(-1, mel_shape[0], mel_shape[1], 1)
    data_splits['X_test_mel'] = data_splits['X_test_mel'].reshape(-1, mel_shape[0], mel_shape[1], 1)
    
    results = {}
    
    # Train CNN model
    print("\nTraining CNN model...")
    cnn_model = model_builder.create_cnn_model(input_shape=(mel_shape[0], mel_shape[1], 1))
    cnn_model.fit(
        data_splits['X_train_mel'], data_splits['y_train'],
        epochs=10, batch_size=32, verbose=1,
        validation_split=0.2
    )
    
    # Evaluate CNN
    cnn_pred = cnn_model.predict(data_splits['X_test_mel'])
    cnn_pred_classes = np.argmax(cnn_pred, axis=1)
    cnn_accuracy = accuracy_score(data_splits['y_test'], cnn_pred_classes)
    results['CNN'] = cnn_accuracy
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    
    # Train traditional ML models
    print("\nTraining traditional ML models...")
    traditional_models = model_builder.create_traditional_ml_models()
    
    for name, model in traditional_models.items():
        print(f"Training {name}...")
        model.fit(data_splits['X_train_traditional'], data_splits['y_train'])
        pred = model.predict(data_splits['X_test_traditional'])
        accuracy = accuracy_score(data_splits['y_test'], pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save models
    os.makedirs('saved_models', exist_ok=True)
    cnn_model.save('saved_models/cnn_model.h5')
    joblib.dump(traditional_models['random_forest'], 'saved_models/random_forest_model.pkl')
    joblib.dump(traditional_models['svm'], 'saved_models/svm_model.pkl')
    preprocessor.save_preprocessor('saved_models/label_encoder.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    
    print("\n" + "="*50)
    print("TRAINING RESULTS:")
    print("="*50)
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    print("\nModels saved successfully!")
    return results

if __name__ == "__main__":
    quick_train()