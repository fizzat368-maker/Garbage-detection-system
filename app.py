import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="AI Garbage Detection System",
    page_icon="🗑️",
    layout="centered"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #28a745;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-container {
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title & Header
st.markdown("<div class='title-container'><h1>🗑️ AI-Based Garbage Detection System</h1><p>Using Convolutional Neural Networks (CNN) for Environmental Cleanliness</p></div>", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("Project Details")
st.sidebar.info("""
**Course:** Artificial Intelligence  
**Model:** CNN (TensorFlow/Keras)  
**Dataset:** 300 Self-Collected Images  
**Members:**  
- Adan Zubair (BSCS)  
- Fizza Tahir (BSAI)
""")

# Load Model
MODEL_PATH = 'garbage_model.h5'
if os.path.exists(MODEL_PATH):
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_my_model()
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image to scan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Predict Button
        if st.button('Analyze Area'):
            with st.spinner('AI is analyzing the image...'):
                # Preprocess
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                # Prediction
                prediction = model.predict(img_array)[0][0]
                
                # Results Display
                st.markdown("<hr>", unsafe_allow_html=True)
                if prediction > 0.5:
                    st.error(f"###  Garbage Detected!")
                    st.progress(float(prediction))
                    st.write(f"Confidence Level: **{prediction*100:.2f}%**")
                else:
                    st.success(f"### Clean Area")
                    st.progress(float(1 - prediction))
                    st.write(f"Confidence Level: **{(1-prediction)*100:.2f}%**")
                    
else:
    st.warning(" Model file not found. Please run `train_model.py` first to train the CNN.")
    st.info("The training script will generate `garbage_model.h5` which is required for this app.")

# Footer
st.markdown("<br><hr><center>Artificial Intelligence Course Project | 2026</center>", unsafe_allow_html=True)
