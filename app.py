import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_vgg19_model():
    return load_model("vgg19.h5")

with st.spinner("Loading model..."):
    model = load_vgg19_model()

# Define class labels
class_names = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]

# App title
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
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    probabilities = tf.nn.softmax(prediction[0]).numpy()
    predicted_class = class_names[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100

    # Show prediction
    st.subheader("🧪 Prediction")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.bar_chart(probabilities)

# Show training history if file exists
if "vgg19_history.pkl" in os.listdir():
    st.subheader("📈 Model Training History")

    with open("vgg19_history.pkl", "rb") as f:
        history = pickle.load(f)

    # Accuracy Plot
    fig1, ax1 = plt.subplots()
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    st.pyplot(fig1)

    # Loss Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Training history not found. Please upload `vgg19_history.pkl` to view training performance.")
