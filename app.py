from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn import datasets
import os
import json

app = Flask(__name__)

# Load the model and iris dataset for target names
print("Starting Iris Flower Classification Application...")
try:
    # Try loading the new model first, then fall back to the original model
    if os.path.exists('new_iris_model.pkl'):
        model = joblib.load('new_iris_model.pkl')
        print("Successfully loaded new_iris_model.pkl")
    else:
        model = joblib.load('iris_model.pkl')
        print("Successfully loaded iris_model.pkl")
    
    # Load iris dataset to get target names
    iris = datasets.load_iris()
    class_names = iris.target_names
    print(f"Class names: {class_names}")
    
except Exception as e:
    print(f"Error loading model or dataset: {e}")
    # Fallback to class names if model fails to load
    class_names = ['setosa', 'versicolor', 'virginica']
    model = None

# Additional flower information for enhanced display
flower_info = {
    'setosa': {
        'emoji': '🌸',
        'color': '#FFB6C1',
        'color_name': 'Pink and White',
        'description': 'Small, delicate petals with beautiful pink and white colors',
        'origin': 'North America and eastern Asia',
        'size': 'Small (petals < 2cm)',
        'habitat': 'Cooler climates, Arctic regions',
        'image': 'iris setosa.jpg'
    },
    'versicolor': {
        'emoji': '🌺',
        'color': '#8A2BE2',
        'color_name': 'Blue-Purple',
        'description': 'Medium-sized flowers with stunning blue-purple hues',
        'origin': 'Eastern North America',
        'size': 'Medium (petals 2-4cm)',
        'habitat': 'Wetlands and marshy areas',
        'image': 'iris versicolor.jpg'
    },
    'virginica': {
        'emoji': '🌷',
        'color': '#4B0082',
        'color_name': 'Deep Violet-Purple',
        'description': 'Large, magnificent blooms with deep violet-purple colors',
        'origin': 'Eastern North America (Virginia)',
        'size': 'Large (petals > 4cm)',
        'habitat': 'Various soil types, adaptable',
        'image': 'iris verginica.jpg'
    }
}

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/api/flower-info')
def get_flower_info():
    """API endpoint to get flower information"""
    return jsonify(flower_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise Exception("Model failed to load")
            
        # Get form values as float
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        
        # Validate input ranges (basic sanity check)
        for i, feature in enumerate(features):
            if feature < 0 or feature > 15:  # Reasonable limits for iris measurements
                raise ValueError(f"Feature {i+1} value {feature} is outside reasonable range (0-15 cm)")
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get prediction probabilities for confidence
        prediction_proba = model.predict_proba([features])[0]
        confidence = max(prediction_proba) * 100
        
        # Get the class name (flower species)
        species = class_names[prediction]
        
        # Capitalize the species name for display
        species_display = f"Iris {species.capitalize()}"
        
        # Get additional flower information
        additional_info = flower_info.get(species, {})
        
        # Print debug info
        print(f"Input features: {features}")
        print(f"Prediction: {prediction}, Species: {species_display}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Probabilities: {prediction_proba}")
        
        return render_template('result.html', 
                             prediction=species_display,
                             confidence=confidence,
                             features=features,
                             flower_info=additional_info)
    
    except ValueError as ve:
        error_message = f"Invalid input: {str(ve)}"
        print(error_message)
        return render_template('result.html', 
                             prediction="Error: Invalid input values", 
                             error=error_message)
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        print(error_message)
        return render_template('result.html', 
                             prediction="Error: Could not make prediction", 
                             error=error_message)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
