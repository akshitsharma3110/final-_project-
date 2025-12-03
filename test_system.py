import os
import numpy as np
import librosa
from audio_preprocessor import AudioPreprocessor
from models import DisasterClassificationModels

def test_audio_preprocessing():
    """Test audio preprocessing functionality"""
    print("Testing Audio Preprocessing...")
    
    preprocessor = AudioPreprocessor()
    
    # Test with a sample audio file
    sample_files = []
    for disaster_type in ['cyclone', 'earthquake', 'explosion', 'fire', 'flood', 'landslide', 'thunderstorm']:
        disaster_path = os.path.join('audio_dataset', disaster_type)
        if os.path.exists(disaster_path):
            files = [f for f in os.listdir(disaster_path) if f.endswith('.wav')]
            if files:
                sample_files.append(os.path.join(disaster_path, files[0]))
    
    if sample_files:
        print(f"Testing with {len(sample_files)} sample files...")
        
        for i, file_path in enumerate(sample_files[:3]):  # Test first 3 files
            print(f"Processing {file_path}...")
            features = preprocessor.extract_features(file_path)
            
            if features is not None:
                print(f"  [OK] Features extracted successfully")
                print(f"  - MFCC shape: {features['mfcc_mean'].shape}")
                print(f"  - Mel spectrogram shape: {features['mel_spec'].shape}")
                print(f"  - Spectral centroid mean: {features['spectral_centroid_mean']:.2f}")
            else:
                print(f"  [ERROR] Failed to extract features")
    else:
        print("No audio files found for testing")
    
    print("Audio preprocessing test completed!\n")

def test_model_creation():
    """Test model creation"""
    print("Testing Model Creation...")
    
    model_builder = DisasterClassificationModels(num_classes=7)
    
    try:
        # Test CNN model
        cnn_model = model_builder.create_cnn_model()
        print("  [OK] CNN model created successfully")
        print(f"  - Total parameters: {cnn_model.count_params():,}")
        
        # Test RNN model
        rnn_model = model_builder.create_rnn_model()
        print("  [OK] RNN model created successfully")
        print(f"  - Total parameters: {rnn_model.count_params():,}")
        
        # Test Hybrid model
        hybrid_model = model_builder.create_hybrid_model()
        print("  [OK] Hybrid model created successfully")
        print(f"  - Total parameters: {hybrid_model.count_params():,}")
        
        # Test traditional ML models
        traditional_models = model_builder.create_traditional_ml_models()
        print(f"  [OK] Traditional ML models created: {list(traditional_models.keys())}")
        
    except Exception as e:
        print(f"  [ERROR] Model creation failed: {e}")
    
    print("Model creation test completed!\n")

def test_data_pipeline():
    """Test complete data pipeline"""
    print("Testing Data Pipeline...")
    
    try:
        preprocessor = AudioPreprocessor()
        
        # Test dataset preparation
        print("  Preparing dataset...")
        features_list, labels = preprocessor.prepare_dataset('audio_dataset')
        
        if features_list and labels:
            print(f"  [OK] Dataset prepared successfully")
            print(f"  - Total samples: {len(features_list)}")
            print(f"  - Classes found: {set(labels)}")
            print(f"  - Samples per class: {len(features_list) // len(set(labels))}")
            
            # Test data splitting
            print("  Splitting data...")
            data_splits = preprocessor.split_data(features_list)
            
            print(f"  [OK] Data split successfully")
            print(f"  - Training samples: {len(data_splits['y_train'])}")
            print(f"  - Validation samples: {len(data_splits['y_val'])}")
            print(f"  - Test samples: {len(data_splits['y_test'])}")
            print(f"  - Traditional features shape: {data_splits['X_train_traditional'].shape}")
            print(f"  - Mel spectrogram shape: {data_splits['X_train_mel'].shape}")
            
        else:
            print("  [ERROR] Dataset preparation failed")
            
    except Exception as e:
        print(f"  [ERROR] Data pipeline test failed: {e}")
    
    print("Data pipeline test completed!\n")

def test_feature_extraction_performance():
    """Test feature extraction performance"""
    print("Testing Feature Extraction Performance...")
    
    import time
    
    preprocessor = AudioPreprocessor()
    
    # Find a sample audio file
    sample_file = None
    for disaster_type in ['cyclone', 'earthquake', 'explosion']:
        disaster_path = os.path.join('audio_dataset', disaster_type)
        if os.path.exists(disaster_path):
            files = [f for f in os.listdir(disaster_path) if f.endswith('.wav')]
            if files:
                sample_file = os.path.join(disaster_path, files[0])
                break
    
    if sample_file:
        print(f"  Testing with: {sample_file}")
        
        # Time feature extraction
        start_time = time.time()
        features = preprocessor.extract_features(sample_file)
        end_time = time.time()
        
        if features is not None:
            processing_time = end_time - start_time
            print(f"  [OK] Feature extraction completed in {processing_time:.3f} seconds")
            
            # Check feature dimensions
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
            
            print(f"  - Traditional feature vector length: {len(feature_vector)}")
            print(f"  - Mel spectrogram shape: {features['mel_spec'].shape}")
            
            if processing_time < 1.0:
                print("  [OK] Performance: GOOD (< 1 second)")
            elif processing_time < 3.0:
                print("  [WARNING] Performance: ACCEPTABLE (1-3 seconds)")
            else:
                print("  [WARNING] Performance: SLOW (> 3 seconds)")
        else:
            print("  [ERROR] Feature extraction failed")
    else:
        print("  [ERROR] No sample file found for testing")
    
    print("Feature extraction performance test completed!\n")

def test_system_requirements():
    """Test system requirements and dependencies"""
    print("Testing System Requirements...")
    
    required_packages = [
        ('numpy', 'numpy'), ('pandas', 'pandas'), ('librosa', 'librosa'), 
        ('sklearn', 'scikit-learn'), ('tensorflow', 'tensorflow'), 
        ('matplotlib', 'matplotlib'), ('seaborn', 'seaborn'), ('flask', 'flask')
    ]
    
    missing_packages = []
    
    for import_name, display_name in required_packages:
        try:
            __import__(import_name)
            print(f"  [OK] {display_name} is installed")
        except ImportError:
            print(f"  [MISSING] {display_name} is missing")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n  Missing packages: {missing_packages}")
        print("  Run: pip install -r requirements.txt")
    else:
        print("  [OK] All required packages are installed")
    
    # Check dataset structure
    print("\n  Checking dataset structure...")
    expected_classes = ['cyclone', 'earthquake', 'explosion', 'fire', 'flood', 'landslide', 'thunderstorm']
    
    if os.path.exists('audio_dataset'):
        print("  [OK] audio_dataset directory exists")
        
        for class_name in expected_classes:
            class_path = os.path.join('audio_dataset', class_name)
            if os.path.exists(class_path):
                file_count = len([f for f in os.listdir(class_path) if f.endswith('.wav')])
                print(f"    [OK] {class_name}: {file_count} files")
            else:
                print(f"    [MISSING] {class_name}: directory missing")
    else:
        print("  [MISSING] audio_dataset directory not found")
    
    print("System requirements test completed!\n")

def main():
    """Run all tests"""
    print("="*60)
    print("DISASTER AUDIO CLASSIFICATION SYSTEM - TESTING")
    print("="*60)
    print()
    
    # Run all tests
    test_system_requirements()
    test_audio_preprocessing()
    test_model_creation()
    test_data_pipeline()
    test_feature_extraction_performance()
    
    print("="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Run 'python train_model.py' to train models")
    print("2. Run 'python evaluate_models.py' for comprehensive evaluation")
    print("3. Run 'python app.py' to start the web interface")

if __name__ == "__main__":
    main()