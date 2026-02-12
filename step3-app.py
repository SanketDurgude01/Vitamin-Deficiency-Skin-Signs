"""
STEP 3: app.py
Save this file as: app.py
Description: Flask REST API server for predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = 'vitamin_deficiency_model_final.h5'
model = None

# Vitamin information database
CLASS_INFO = {
    'Vitamin_A_Deficiency': {
        'name': 'Vitamin A',
        'signs': ['Dry, scaly skin', 'Follicular hyperkeratosis', 'Rough patches'],
        'recommendations': [
            'Increase intake of carrots and sweet potatoes',
            'Consider vitamin A supplements (consult doctor)',
            'Eat liver, eggs, and dairy products'
        ],
        'sources': ['Carrots', 'Sweet potatoes', 'Spinach', 'Liver', 'Eggs'],
        'daily_requirement': '700-900 mcg'
    },
    'Vitamin_B_Deficiency': {
        'name': 'Vitamin B Complex',
        'signs': ['Cracked lips', 'Dermatitis', 'Inflammation'],
        'recommendations': [
            'Eat more whole grains and leafy greens',
            'Include eggs, meat, and dairy',
            'Consider B-complex supplements'
        ],
        'sources': ['Whole grains', 'Meat', 'Eggs', 'Legumes'],
        'daily_requirement': 'Varies by B vitamin'
    },
    'Vitamin_C_Deficiency': {
        'name': 'Vitamin C',
        'signs': ['Easy bruising', 'Poor wound healing', 'Dry skin'],
        'recommendations': [
            'Eat more citrus fruits daily',
            'Include bell peppers and strawberries',
            'Consider vitamin C supplements (500-1000mg)'
        ],
        'sources': ['Citrus fruits', 'Bell peppers', 'Strawberries', 'Broccoli'],
        'daily_requirement': '75-90 mg'
    },
    'Vitamin_D_Deficiency': {
        'name': 'Vitamin D',
        'signs': ['Pale skin', 'Slow wound healing', 'Psoriasis'],
        'recommendations': [
            'Get 15-20 minutes of sunlight daily',
            'Consume fortified dairy products',
            'Consider vitamin D3 supplements (1000-2000 IU)'
        ],
        'sources': ['Sunlight', 'Fatty fish', 'Fortified dairy', 'Egg yolks'],
        'daily_requirement': '600-800 IU'
    },
    'Vitamin_E_Deficiency': {
        'name': 'Vitamin E',
        'signs': ['Dry skin', 'Poor elasticity', 'Premature aging'],
        'recommendations': [
            'Consume nuts and seeds daily',
            'Use vitamin E oil topically',
            'Eat avocados and spinach'
        ],
        'sources': ['Nuts', 'Seeds', 'Vegetable oils', 'Avocado'],
        'daily_requirement': '15 mg'
    },
    'Normal_Skin': {
        'name': 'Normal/Healthy',
        'signs': ['No visible deficiency signs'],
        'recommendations': [
            'Maintain balanced diet',
            'Stay hydrated (8 glasses daily)',
            'Continue regular skin care'
        ],
        'sources': ['Varied diet'],
        'daily_requirement': 'N/A'
    }
}

def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            logger.info(f"✓ Model loaded: {MODEL_PATH}")
        else:
            logger.warning(f"⚠ Model not found: {MODEL_PATH}")
            logger.warning("Running in DEMO mode")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        logger.warning("Running in DEMO mode")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Preprocess image for prediction"""
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image_bytes):
    """Make prediction on image"""
    img_array = preprocess_image(image_bytes)
    
    if model is not None:
        # Real prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        class_names = list(CLASS_INFO.keys())
        predicted_class = class_names[predicted_class_idx]
        
        all_predictions = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
    else:
        # Demo mode - random prediction
        class_names = list(CLASS_INFO.keys())
        predicted_class = np.random.choice(class_names)
        confidence = np.random.uniform(0.65, 0.95)
        
        all_predictions = {
            name: np.random.uniform(0.05, 0.3) if name != predicted_class else confidence
            for name in class_names
        }
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'vitamin_info': CLASS_INFO[predicted_class],
        'all_predictions': all_predictions
    }
    
    return result

# API Routes
@app.route('/')
def home():
    """API home page"""
    return jsonify({
        'message': 'Vitamin Deficiency Detection API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'POST /predict': 'Upload image for prediction',
            'POST /predict_base64': 'Predict from base64 image',
            'GET /health': 'Health check',
            'GET /classes': 'Get all vitamin information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0'
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all vitamin class information"""
    return jsonify({
        'classes': CLASS_INFO,
        'total_classes': len(CLASS_INFO)
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Prediction endpoint for file upload"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed: png, jpg, jpeg'
            }), 400
        
        # Read image
        image_bytes = file.read()
        
        # Make prediction
        logger.info(f"Processing: {file.filename}")
        result = predict(image_bytes)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64_endpoint():
    """Prediction endpoint for base64 encoded images"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Make prediction
        logger.info("Processing base64 image")
        result = predict(image_bytes)
        
        logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({'error': 'File too large. Max size is 10MB'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Main execution
if __name__ == '__main__':
    print("\n" + "="*70)
    print("VITAMIN DEFICIENCY DETECTION API SERVER")
    print("="*70)
    
    print("\n[1] Loading model...")
    load_model()
    
    print("\n[2] Server Configuration:")
    print(f"    Host: 0.0.0.0")
    print(f"    Port: 5000")
    print(f"    Model Status: {'Loaded' if model else 'Demo Mode'}")
    print(f"    Max Upload: 10MB")
    
    print("\n[3] Available Endpoints:")
    print("    GET  /          - API information")
    print("    GET  /health    - Health check")
    print("    GET  /classes   - Vitamin information")
    print("    POST /predict   - Upload image")
    print("    POST /predict_base64 - Base64 image")
    
    print("\n[4] Starting server...")
    print("="*70)
    print("\n✓ Server running at: http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)