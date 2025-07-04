# Iris Flower Detection Web Application

Hare Checkout:=👉https://itsluckysharma01.github.io/Prediction_iris_Flower_Machine_Learning-Flask/👈🫡

This is a simple Flask web application that uses a machine learning model to predict the species of iris flowers based on measurements.

## Files and Structure

- `app.py` - The main Flask application
- `iris_model.pkl` / `new_iris_model.pkl` - The trained machine learning model
- `templates/` - Folder containing HTML templates
  - `form.html` - Input form for flower measurements
  - `result.html` - Page showing prediction results
- `create_new_model.py` - Script to create a fresh model if needed
- `test_app.py` - Script to test the application functionality
- `run_app.bat` - Windows batch file to easily run the application

## How to Run

1. Double-click on `run_app.bat` or run `python app.py` in your terminal
2. Open your web browser and go to http://127.0.0.1:5000
3. Enter the flower measurements and click "Predict Flower Species"

## Sample Measurements

### Iris Setosa

- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

### Iris Versicolor

- Sepal Length: 6.0 cm
- Sepal Width: 2.7 cm
- Petal Length: 4.2 cm
- Petal Width: 1.3 cm

### Iris Virginica

- Sepal Length: 6.8 cm
- Sepal Width: 3.0 cm
- Petal Length: 5.5 cm
- Petal Width: 2.1 cm

## Troubleshooting

If you encounter issues:

1. Run `python test_app.py` to verify the model is working correctly
2. Check that you have all the required Python packages installed:
   - Flask
   - scikit-learn
   - joblib
   - numpy
3. Try generating a new model with `python create_new_model.py`
