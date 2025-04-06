import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_vgg19_model():
    return load_model("vgg19.h5")

model = load_vgg19_model()

# Class names (adjust based on your model training)
class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

st.title("🧠 Alzheimer's Stage Detection")
st.write("Upload a brain MRI image to detect the stage of Alzheimer's.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # VGG19 expects 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.markdown("### 🧪 Prediction Result")
    st.write(f"**Stage:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
