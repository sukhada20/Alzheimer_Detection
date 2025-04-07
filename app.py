import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
@st.cache_resource
def load_vgg19_model():
    model = load_model("vgg19.h5")
    return model

model = load_vgg19_model()

# Define class labels
class_names = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# Title
st.title("🧠 Alzheimer's Detection from Brain Scans")
st.write("Upload an MRI brain scan and the model will predict the likelihood of Alzheimer's.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show results
    st.subheader("🧪 Prediction")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.bar_chart(prediction[0])
