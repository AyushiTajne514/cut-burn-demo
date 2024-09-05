from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = load_model('vgg19_model.h5')
 # Replace 'your_trained_model.h5' with your model's filename

# Mapping of class indices to labels (ensure these match your model's output classes)
class_labels = {0: "burn", 1: "cut"}  # Adjust according to your training labels

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image from the request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (250, 250))  # Resize to match your model's expected input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the numeric class to the label
    predicted_label = class_labels.get(predicted_class, "Unknown")

    return jsonify({'class': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
