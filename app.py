import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
@st.cache_resource
def load_alzheimer_model():
    model = load_model("alzheimer_detection.h5")
    return model

model = load_alzheimer_model()

# Define class names (update based on your model's classes)
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Streamlit UI
st.title("🧠 Alzheimer's Disease Detection")
st.write("Upload a brain MRI image and the model will predict the Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (update size if your model expects a different input shape)
    img = img.resize((150, 150))  # Change size according to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("🧾 Prediction:")
    st.write(f"**{predicted_class}**")

    # Optionally display confidence scores
    st.subheader("📊 Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {score:.2%}")
