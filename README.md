# 🌸 Interactive Iris Flower Prediction Web Application 🌸

A beautiful, modern web application that uses Machine Learning to predict Iris flower species with an enhanced interactive user interface, animated backgrounds, and stunning visual effects.

**Live Demo**: https://itsluckysharma01.github.io/Prediction_iris_Flower_Machine_Learning-Flask/ 👈🫡

## ✨ New Enhanced Features

### 🎨 Interactive Design

- **Modern UI/UX**: Beautiful gradient backgrounds with glassmorphism effects
- **Animated Background Video**: Looping flower videos for immersive experience
- **Interactive Flower Cards**: Click-to-fill example values with hover effects
- **Floating Particles**: Dynamic flower emojis floating across the screen
- **Smooth Animations**: CSS keyframe animations for all elements

### 🌺 Flower Showcase

- **Real Flower Images**: Actual photographs of each iris species
- **Visual Flower Display**: High-quality images showing true flower colors
- **Detailed Information**: Comprehensive facts about each flower type with color names
- **Interactive Examples**: Click any flower card to auto-fill the form
- **Species-Specific Styling**: Unique colors and animations for each iris type
- **Dynamic Backgrounds**: Background colors change based on predicted flower type

### 🚀 Enhanced Functionality

- **Form Validation**: Real-time input validation with visual feedback
- **Number Inputs**: Proper numeric inputs with step controls
- **Confidence Scoring**: Display prediction confidence percentages
- **Error Handling**: Graceful error messages with helpful suggestions
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile

### 🎭 Visual Effects

- **Real Flower Photography**: High-quality images of actual iris flowers
- **Dynamic Background Colors**: Background changes based on predicted flower species
- **Background Videos**: Multiple fallback video sources for reliability
- **Particle System**: Dynamic floating flower animations
- **Confetti Effects**: Celebration animations for successful predictions
- **Glow Effects**: Smooth glowing animations throughout the interface
- **Hover Interactions**: Elements respond to user interactions
- **Custom Favicon**: Beautiful iris flower favicon for all devices and sizes
- **PWA Support**: Web app manifest for mobile installation
- **Color-Themed Results**: Each flower type displays with its natural color scheme

## 🎨 Favicon and Branding

The application now includes a complete set of favicon files for optimal display across all devices and platforms:

### 🌸 Design Elements

- **Gradient backgrounds**: Beautiful purple to pink gradients matching the app theme
- **Iris flower motifs**: Custom-designed flower shapes in the favicon
- **Consistent branding**: All icons follow the same color scheme and design language
- **Multiple sizes**: Optimized for different display contexts and resolutions

### 📱 PWA Features

- **Installable**: Users can install the app on their mobile devices
- **Standalone mode**: App runs in full-screen mode when installed
- **Custom theme colors**: Matches the application's visual design
- **Optimized icons**: Perfect display in app drawers and home screens

## 🛠️ Technical Features

### Machine Learning

- `app.py` - The main Flask application
- `iris_model.pkl` / `new_iris_model.pkl` - The trained machine learning model
- `templates/` - Folder containing HTML templates
  - `form.html` - Input form for flower measurements
  - `result.html` - Page showing prediction results
- `static/` - Folder containing static files

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

---

**Ready to explore the beautiful world of Iris flowers! 🌸🤖✨**
