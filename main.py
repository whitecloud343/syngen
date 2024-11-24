import streamlit as st
from fastai.vision.all import load_learner, PILImage
from pathlib import Path

# Load the trained model
MODEL_PATH = Path("tomato_fruit_model.pkl")
model = load_learner(MODEL_PATH)

# Streamlit app
st.title("Tomato Fruit Disease Prediction")
st.write("Upload an image of a tomato fruit to classify its disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    try:
        image = PILImage.create(uploaded_file)
        pred, pred_idx, probs = model.predict(image)

        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"Class: **{pred}**")
        st.write(f"Confidence: **{probs[pred_idx] * 100:.2f}%**")
    except Exception as e:
        st.error(f"Error in processing the image: {e}")
