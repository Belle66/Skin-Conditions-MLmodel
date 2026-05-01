import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Dermatology AI Classifier", page_icon="🩺", layout="centered")

# 2. Load the Model
@st.cache_resource
def load_my_model():
    # Make sure this filename matches your file in the folder!
    # If using the native format, change to 'skin_disease_model.keras'
    model = tf.keras.models.load_model('skin_disease_model.h5')
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Define the categories
# Order must match how the model was trained (usually alphabetical)
class_names = ['Acne and Rosacea', 'Eczema']

# 4. The User Interface
st.title("🩺 Dermatology AI Classifier")
st.write("Upload a clear photo of the skin condition for an AI-powered prediction.")
st.markdown("---")

file = st.file_uploader("Choose a photo (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if file is not None:
    # --- Step A: Open and Convert ---
    # .convert('RGB') fixes the "shape=(1, 224, 224, 4)" error by removing transparency
    image = Image.open(file).convert('RGB')
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- Step B: Preprocessing ---
    # Resize to 224x224 to match MobileNetV2 input
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize (0 to 1)
    img_array = np.asarray(image_resized)
    img_array = img_array / 255.0 
    
    # Add the batch dimension (None, 224, 224, 3)
    img_reshape = img_array[np.newaxis, ...]
    
    # --- Step C: Prediction ---
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_reshape)
        
    result_index = np.argmax(prediction)
    result_label = class_names[result_index]
    confidence = np.max(prediction) * 100

    # --- Step D: Display Results ---
    st.subheader("Result:")
    if confidence > 75:
        st.success(f"**Identified as:** {result_label}")
    else:
        st.warning(f"**Likely:** {result_label} (Low confidence)")
        
    st.info(f"**Confidence Level:** {confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("Developed for Horus University (HUE) - AI Department Project")