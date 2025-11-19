import streamlit as st
import joblib
import re
import os

# 1. Page Configuration (Tab title and icon)
st.set_page_config(
    page_title="Property Classifier",
    page_icon="🏠",
    layout="centered"
)

# 2. Define Cleaning Function (MUST match training logic)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Load Model (Cached for performance)
# @st.cache_resource ensures the model loads only once, not every time you click a button
@st.cache_resource
def load_model():
    model_path = 'best_model/property_classifier_pipeline.pkl'
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please run the training notebook first.")
        return None
    return joblib.load(model_path)

model = load_model()

# 4. Mapping categories to Emojis for a better UI
category_emojis = {
    "flat": "🏢 Flat/Apartment",
    "houseorplot": "🏠 House or Plot",
    "landparcel": "🌳 Land Parcel",
    "commercial unit": "🏪 Commercial Unit",
    "others": "❓ Others"
}

# 5. The User Interface
st.title("📍 Property Address Classifier")
st.markdown("Enter a raw property address below, and the AI will categorize it.")

# Text Input Area
address_input = st.text_area("Property Address:", placeholder="e.g., Flat 402, Galaxy Apartments, MG Road...", height=100)

# Predict Button
if st.button("Classify Address", type="primary"):
    if not address_input.strip():
        st.warning("⚠️ Please enter an address first.")
    elif model is None:
        st.error("Model is not loaded.")
    else:
        # Processing
        with st.spinner("Analyzing..."):
            cleaned_address = clean_text(address_input)
            prediction = model.predict([cleaned_address])[0]
            
            # Get readable label
            display_label = category_emojis.get(prediction, prediction)
            
            # Display Result
            st.success("### Prediction Complete!")
            st.markdown(f"The address belongs to category: **{display_label}**")
            
            # Optional: Show what the model "saw" (Debugging info)
            with st.expander("See processed input"):
                st.text(f"Original: {address_input}")
                st.text(f"Cleaned:  {cleaned_address}")

# Footer
st.markdown("---")
st.caption("AI Intern Assignment Submission | Model: LinearSVC + TF-IDF")