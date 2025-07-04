from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn import datasets
import os

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

@app.route('/')
def form():
    return render_template('form.html')

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
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get the class name (flower species)
        species = class_names[prediction]
        
        # Capitalize the species name for display
        species_display = f"Iris {species}"
        
        # Print debug info
        print(f"Input features: {features}")
        print(f"Prediction: {prediction}, Species: {species_display}")
        
        return render_template('result.html', prediction=species_display)
    
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        print(error_message)
        return render_template('result.html', prediction="Error: Could not make prediction", error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
