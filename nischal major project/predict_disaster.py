import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

def predict_disaster(image_path, model_path, class_names_path):
    """
    Predict disaster type for a given image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        class_names_path: Path to class names JSON file
    
    Returns:
        Predicted class and confidence
    """
    # Load model and class names
    model = keras.models.load_model(model_path)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0], class_names


def visualize_prediction(image_path, model_path, class_names_path):
    """Visualize prediction with confidence scores."""
    predicted_class, confidence, all_predictions, class_names = predict_disaster(
        image_path, model_path, class_names_path
    )
    
    # Load image
    img = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}', fontsize=12)
    ax1.axis('off')
    
    # Display confidence scores
    colors = ['green' if i == np.argmax(all_predictions) else 'lightblue' 
              for i in range(len(class_names))]
    ax2.barh(class_names, all_predictions, color=colors)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Prediction Confidence by Class')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Predicted Disaster: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll predictions:")
    for class_name, pred in zip(class_names, all_predictions):
        print(f"  {class_name}: {pred:.2%}")


if __name__ == '__main__':
    model_path = r'b:\nischal major project\disaster_cnn_model.h5'
    class_names_path = r'b:\nischal major project\class_names.json'
    
    # Example: predict on a test image
    test_image = r'b:\nischal major project\disasters\flood\0.jpg'
    
    if os.path.exists(test_image):
        visualize_prediction(test_image, model_path, class_names_path)
    else:
        print(f"Test image not found: {test_image}")
