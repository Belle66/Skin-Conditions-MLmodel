import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Dermatology AI Classifier", page_icon="Dermatology AI", layout="centered")

@st.cache_resource
def load_my_model():
model = tf.keras.models.load_model('skin_disease_model.keras')
return model

try:
model = load_my_model()
except Exception as e:
st.error(f"Error loading model: {e}")
st.stop()

class_names = ['Acne and Rosacea', 'Eczema']

st.title("Dermatology AI Classifier")
st.write("Upload a clear photo of the skin condition for an AI-powered prediction.")
st.markdown("---")

file = st.file_uploader("Choose a photo (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if file is not None:
image = Image.open(file).convert('RGB')
st.image(image, caption='Uploaded Image', use_container_width=True)

size = (224, 224)
image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

img_array = np.asarray(image_resized).astype(np.float32) / 255.0
img_reshape = np.expand_dims(img_array, axis=0)

with st.spinner("Analyzing image..."):
prediction = model.predict(img_reshape)

result_index = np.argmax(prediction)
result_label = class_names[result_index]
confidence = np.max(prediction) * 100

st.subheader("Result:")
if confidence > 75:
st.success(f"Identified as: {result_label}")
else:
st.warning(f"Likely: {result_label} (Low confidence)")

st.info(f"Confidence Level: {confidence:.2f}%")

st.markdown("---")
st.caption("Developed for Horus University (HUE) - AI Department Project")
