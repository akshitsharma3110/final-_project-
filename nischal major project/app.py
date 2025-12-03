import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL_PATH = r"b:\nischal major project\disaster_mobilenet.h5"
CLASS_NAMES_PATH = r"b:\nischal major project\class_names.json"
DISASTER_INFO_PATH = r"b:\nischal major project\disaster_info.json"

model = keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
with open(DISASTER_INFO_PATH, 'r') as f:
    disaster_info = json.load(f)

IMG_SIZE = (224, 224)

def get_google_images(query, num_images=2):
    """Fetch image URLs from Google Images"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = f"https://www.google.com/search?q={query}&tbm=isch"
        response = requests.get(url, headers=headers, timeout=5)
        
        images = []
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img', limit=num_images+1)
        
        for img in img_tags[1:num_images+1]:
            if 'src' in img.attrs and 'http' in str(img['src']):
                images.append(img['src'])
        
        return images[:num_images]
    except:
        return []

def get_disaster_images(disaster_type):
    """Get images for a specific disaster type"""
    search_queries = {
        'biological and chemical pandemic': 'pandemic safety health prevention',
        'cyclone': 'cyclone storm damage preparedness',
        'drought': 'drought dry land water scarcity prevention',
        'earthquake': 'earthquake damage building safety',
        'flood': 'flood water disaster prevention',
        'landslide': 'landslide mountain slope safety',
        'tsunami': 'tsunami wave ocean safety',
        'wildfire': 'wildfire forest fire prevention'
    }
    
    query = search_queries.get(disaster_type, disaster_type)
    images = get_google_images(query, 3)
    
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Get disaster information
        disaster_data = disaster_info.get(predicted_class, {})
        
        # Get related images
        related_images = get_disaster_images(predicted_class)
        
        results = {
            'predicted_class': predicted_class,
            'confidence': f'{confidence*100:.2f}%',
            'confidence_value': confidence,
            'all_predictions': {class_names[i]: f'{predictions[i]*100:.2f}%' for i in range(len(class_names))},
            'helpline': disaster_data.get('helpline', 'N/A'),
            'prevention_measures': disaster_data.get('prevention_measures', []),
            'image_keywords': disaster_data.get('image_keywords', []),
            'related_images': related_images
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
