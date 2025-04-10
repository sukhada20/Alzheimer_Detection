import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model only once
@st.cache_resource
def load_alzheimer_model():
    return load_model("alzheimer_detection.h5")

model = load_alzheimer_model()

# Define class names (must match model's output order)
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Streamlit UI
st.title("🧠 Alzheimer's Disease Detection")
st.write("Upload a brain MRI image and the model will predict the Alzheimer's stage.")

uploaded_file = st.file_uploader("🖼️ Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image (resize + normalize)
        img = img.resize((150, 150))  # Change based on your model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)[0]

        if len(predictions) != len(class_names):
            st.error("Prediction mismatch: Check if class_names match the model output.")
        else:
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = predictions[predicted_index]

            st.subheader("🧾 Prediction:")
            st.success(f"**{predicted_class}** ({confidence:.2%} confidence)")

            # Optional: show all class probabilities
            st.subheader("📊 Class Probabilities:")
            for i, score in enumerate(predictions):
                st.write(f"{class_names[i]}: {score:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error processing the image: {e}")
