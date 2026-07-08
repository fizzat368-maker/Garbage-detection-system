import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_garbage(img_path, model_path='garbage_model.h5'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train_model.py first.")
        return

    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalization
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Interpret results
    # Assuming class 0 = Clean, class 1 = Garbage (as per flow_from_directory alphabetical order)
    if prediction > 0.5:
        label = "Garbage Detected"
        confidence = prediction * 100
    else:
        label = "Clean Area"
        confidence = (1 - prediction) * 100
        
    print("-" * 30)
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_garbage(sys.argv[1])
