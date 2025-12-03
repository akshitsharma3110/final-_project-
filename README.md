# 🚨 Audio-Based Disaster Classification AI System

A comprehensive deep learning system that classifies audio recordings into different disaster categories using advanced Speech & Audio Processing techniques.

## 📋 Project Overview

This AI system can classify audio recordings into the following disaster categories:
- 🌪️ **Cyclone/Hurricane**
- 🏔️ **Earthquake** 
- 💥 **Explosion**
- 🔥 **Fire**
- 🌊 **Flood**
- ⛰️ **Landslide**
- ⛈️ **Thunderstorm**

## 🎯 Key Features

- **Multiple Model Architectures**: CNN, RNN, Hybrid, and WaveNet-inspired models
- **Advanced Feature Engineering**: MFCC, spectral features, chroma, mel-spectrograms
- **Ensemble Methods**: Voting classifiers and model stacking
- **Data Augmentation**: Time stretching, pitch shifting, noise addition
- **Real-time Prediction API**: Flask-based web interface
- **Comprehensive Evaluation**: ROC curves, confusion matrices, learning curves
- **Accuracy Improvement**: Hyperparameter tuning, cross-validation

## 📁 Project Structure

```
voice_note/
├── audio_dataset/              # Dataset directory
│   ├── cyclone/               # Cyclone audio samples
│   ├── earthquake/            # Earthquake audio samples
│   ├── explosion/             # Explosion audio samples
│   ├── fire/                  # Fire audio samples
│   ├── flood/                 # Flood audio samples
│   ├── landslide/             # Landslide audio samples
│   └── thunderstorm/          # Thunderstorm audio samples
├── templates/                 # HTML templates
│   └── index.html            # Web interface
├── saved_models/             # Trained models (generated)
├── audio_preprocessor.py     # Audio preprocessing & feature extraction
├── models.py                 # Model architectures
├── train_model.py           # Main training script
├── accuracy_improvement.py   # Advanced techniques for accuracy improvement
├── evaluate_models.py       # Comprehensive model evaluation
├── app.py                   # Flask deployment API
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd voice_note

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Your dataset is already organized in the `audio_dataset/` directory with 50 samples per class (350 total samples).

### 3. Train Models

```bash
# Train all models and generate evaluation reports
python train_model.py
```

This will:
- Extract features from all audio files
- Train CNN, RNN, Hybrid, and traditional ML models
- Generate confusion matrices and training plots
- Save trained models to `saved_models/`

### 4. Evaluate Models

```bash
# Run comprehensive evaluation
python evaluate_models.py
```

This generates:
- Model comparison charts
- ROC curves
- Detailed performance reports
- Accuracy improvement analysis

### 5. Deploy Web Interface

```bash
# Start the Flask API
python app.py
```

Visit `http://localhost:5000` to use the web interface for real-time audio classification.

## 🧠 Model Architectures

### 1. CNN Model
- **Input**: Mel-spectrograms (128 x 259 x 1)
- **Architecture**: 3 Conv2D layers with BatchNorm and Dropout
- **Features**: Spatial pattern recognition in spectrograms

### 2. RNN Model  
- **Input**: Sequential mel-spectrogram features (259 x 128)
- **Architecture**: 3 LSTM layers with dropout
- **Features**: Temporal pattern recognition

### 3. Hybrid Model
- **Inputs**: Mel-spectrograms + Traditional features
- **Architecture**: CNN branch + Dense branch with attention
- **Features**: Combines spatial and engineered features

### 4. Traditional ML
- **Models**: Random Forest, SVM with hyperparameter tuning
- **Features**: MFCC, spectral, chroma, rhythm features
- **Ensemble**: Voting classifier for improved accuracy

## 🔧 Feature Engineering

### Audio Features Extracted:
- **MFCC**: 13 coefficients (mean & std)
- **Spectral Features**: Centroid, rolloff, bandwidth, contrast
- **Temporal Features**: Zero-crossing rate, tempo, beats
- **Harmonic Features**: Chroma vectors, tonnetz
- **Energy Features**: Harmonic vs percussive energy
- **Mel-Spectrograms**: 128 mel bands for deep learning

### Data Augmentation:
- Time stretching (0.9x, 1.1x speed)
- Pitch shifting (±2 semitones)  
- Noise addition (Gaussian noise)
- Volume adjustment (0.8x, 1.2x amplitude)

## 📊 Performance Metrics

The system achieves approximately **80% accuracy** with the following evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores  
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve for multiclass

## 🎯 Accuracy Improvement Techniques

### 1. Hyperparameter Tuning
- Grid search for optimal parameters
- Cross-validation for robust evaluation
- Learning rate scheduling

### 2. Ensemble Methods
- Voting classifiers
- Model stacking
- Attention mechanisms

### 3. Advanced Architectures
- Dilated convolutions (WaveNet-inspired)
- Residual connections
- Batch normalization

### 4. Data Strategies
- Stratified sampling
- Data augmentation
- Feature selection

## 🌐 API Usage

### Prediction Endpoint

```python
import requests

# Upload audio file for prediction
files = {'audio': open('sample_audio.wav', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

print(f"Predicted disaster: {result['disaster_type']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Response Format

```json
{
  "disaster_type": "earthquake",
  "confidence": 0.87,
  "all_probabilities": {
    "cyclone": 0.05,
    "earthquake": 0.87,
    "explosion": 0.02,
    "fire": 0.01,
    "flood": 0.03,
    "landslide": 0.01,
    "thunderstorm": 0.01
  }
}
```

## 📈 Dataset Sources & Expansion

### Current Dataset
- **Size**: 350 audio samples (50 per class)
- **Format**: WAV files, 22.05 kHz sampling rate
- **Duration**: ~3 seconds per sample

### Recommended Dataset Sources for Expansion:
1. **Freesound.org**: Community-contributed audio samples
2. **AudioSet**: Google's large-scale audio dataset
3. **ESC-50**: Environmental Sound Classification dataset
4. **UrbanSound8K**: Urban sound classification dataset
5. **Custom Recording**: Field recordings of actual disaster events

### Data Collection Guidelines:
- Maintain consistent audio quality (22.05 kHz, mono)
- Ensure balanced class distribution
- Include diverse environmental conditions
- Validate audio labels for accuracy

## 🔧 Customization & Extension

### Adding New Disaster Classes:
1. Create new directory in `audio_dataset/`
2. Add audio samples to the directory
3. Update `disaster_classes` list in code
4. Retrain models with new data

### Improving Accuracy:
1. **More Data**: Collect additional samples per class
2. **Better Features**: Experiment with advanced audio features
3. **Model Tuning**: Optimize hyperparameters further
4. **Ensemble Methods**: Combine multiple model predictions

### Real-time Processing:
- Implement streaming audio processing
- Add voice activity detection
- Optimize inference speed for mobile deployment

## 🚀 Deployment Options

### Local Deployment
```bash
python app.py  # Flask development server
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### Cloud Deployment
- **AWS**: Deploy using EC2, Lambda, or SageMaker
- **Google Cloud**: Use Cloud Run or AI Platform
- **Azure**: Deploy with Container Instances or ML Studio

## 📋 Requirements

### System Requirements:
- **Python**: 3.7+
- **RAM**: 4GB+ recommended
- **Storage**: 2GB for models and data
- **CPU**: Multi-core recommended for training

### Key Dependencies:
- **TensorFlow**: 2.13.0 (Deep learning)
- **Librosa**: 0.10.1 (Audio processing)
- **Scikit-learn**: 1.3.0 (Traditional ML)
- **Flask**: 2.3.3 (Web API)
- **NumPy/Pandas**: Data manipulation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Librosa**: Excellent audio processing library
- **TensorFlow**: Powerful deep learning framework
- **Scikit-learn**: Comprehensive machine learning toolkit
- **Flask**: Lightweight web framework for deployment

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check existing documentation
- Review the evaluation reports for performance insights

---

**Built with ❤️ for disaster preparedness and emergency response systems**