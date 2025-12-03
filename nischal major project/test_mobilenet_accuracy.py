import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (224, 224)
class_names = ['biological and chemical pandemic', 'cyclone', 'drought', 'earthquake', 'flood', 'landslide', 'tsunami', 'wildfire']

def load_test_data(data_dir):
    """Load all images from disaster folders."""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(class_idx)
            except:
                pass
    
    return np.array(images), np.array(labels)

def test_model():
    """Test model and display comprehensive accuracy metrics."""
    model_path = r"b:\nischal major project\disaster_mobilenet.h5"
    data_dir = r"b:\nischal major project\disasters"
    
    print("Loading model...")
    model = keras.models.load_model(model_path)
    
    print("Loading test data...")
    X_test, y_test = load_test_data(data_dir)
    print(f"Total test images: {len(X_test)}\n")
    
    print("Making predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*60)
    print("MODEL ACCURACY REPORT - MobileNetV2")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print("="*60)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_count = class_mask.sum()
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix (Overall Accuracy: {accuracy*100:.2f}%)', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(r"b:\nischal major project\confusion_matrix_mobilenet.png", dpi=100)
        print("\nConfusion matrix saved to confusion_matrix_mobilenet.png")
    except:
        print("\nConfusion matrix visualization skipped")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_accuracy': {}
    }
    
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            results['per_class_accuracy'][class_name] = float(class_acc)
    
    with open(r"b:\nischal major project\accuracy_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to accuracy_results.json")
    print(f"\nModel Performance Summary:")
    print(f"  - Overall Accuracy: {accuracy*100:.2f}%")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}")
    
    if accuracy >= 0.80:
        print(f"\n[SUCCESS] TARGET ACHIEVED: {accuracy*100:.2f}% >= 80%")
    else:
        print(f"\n[FAILED] Target not met: {accuracy*100:.2f}% < 80%")

if __name__ == '__main__':
    test_model()
