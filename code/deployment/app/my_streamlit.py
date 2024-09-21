import streamlit as st
import requests
from PIL import Image
import numpy as np

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI
st.title("MNIST number Classifier")

# Input fields for the Iris flower data
with st.form(key='my_form'):
    input_data = st.file_uploader(label='Upload picture of number from 0 to 9')
    submit_button = st.form_submit_button(label='Predict')
    
    # Make prediction when the button is clicked
    if submit_button:
        # Prepare the data for the API request
        img = Image.open(input_data).resize((28, 28)).convert('L')
        img = np.asarray(img, dtype=np.float32)
        img /= 255.0
        input_data = {
            'image': str(img.tolist())
        }
        # Send a request to the FastAPI prediction endpoint
        response = requests.post(FASTAPI_URL, json=input_data)
        prediction = response.json()["prediction"]
        
        # Display the result
        st.success(f"The model predicts class: {prediction}")