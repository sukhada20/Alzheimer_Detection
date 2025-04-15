import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

@st.cache_resource
def load_alzheimer_model():
    model = load_model('alzheimer_detection.h5')
    return model

model = load_alzheimer_model()
final_labels = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']

st.title("🧠 Alzheimer’s Disease Detection")
st.subheader("Using Transfer Learning Model for Early Diagnosis")

uploaded_file = st.file_uploader("Upload a brain MRI scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    st.write("Processing the image...")
    img_resized = img.resize((150, 150)) 
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    prediction = model.predict(img_array)
    predicted_class = final_labels[np.argmax(prediction)]

    st.success(f"🧬 **Predicted Diagnosis:** {predicted_class}")
    st.write("Prediction Probabilities:")
    for i, label in enumerate(final_labels):
        st.write(f"{label}: {prediction[0][i]:.4f}")

st.markdown("---")
st.info("This tool uses deep learning to assist with early detection of Alzheimer’s Disease.")
