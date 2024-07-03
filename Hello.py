import pandas as pd
import os
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

LOGGER = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Predictive Analytics App",
    page_icon="📊",
)

# Load pre-trained model
model = None
try:
    LOGGER.info("Attempting to load the model from 'model.pkl'.")
    model = joblib.load('model.pkl')
    LOGGER.info("Model loaded successfully.")
except FileNotFoundError as e:
    LOGGER.error(f"Model file not found: {e}")
    st.error("Model file not found. Please ensure 'model.pkl' is present in the directory.")
except Exception as e:
    LOGGER.error(f"Error loading model: {e}")
    st.error(f"An error occurred while loading the model: {e}")

def run():
    st.write("# Predictive Analytics App 📊")

    st.sidebar.header("User Input Features")
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
    """)

    # File uploader for user to upload CSV file
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        # Automatically load the example CSV file for testing
        input_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'penguins_example.csv'))

    # Ensure 'island' and 'sex' columns exist before encoding
    if 'island' not in input_df.columns or 'sex' not in input_df.columns:
        st.error("The input data must contain 'island' and 'sex' columns.")
        return

    # Encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['island', 'sex'], drop_first=True)

    # Ensure all expected categories are present in the encoded data
    expected_categories = {
        'island': ['Biscoe', 'Dream', 'Torgersen'],
        'sex': ['male', 'female']
    }
    for column, categories in expected_categories.items():
        for category in categories:
            col_name = f"{column}_{category}"
            if col_name not in input_df.columns:
                input_df[col_name] = 0

    # Define the expected feature order based on the model's training data
    expected_feature_order = [
        'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'
    ]

    # Reorder the columns of input_df to match the expected feature order
    input_df = input_df.reindex(columns=expected_feature_order, fill_value=0)

    # Displays the user input features
    st.subheader('User Input features')
    st.write(input_df)

    # Apply model to make predictions
    if model is not None:
        try:
            # Make predictions
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader('Prediction')
            st.write(prediction)

            st.subheader('Prediction Probability')
            st.write(prediction_proba)

            # Visualize prediction confidence
            st.subheader('Prediction Confidence Visualization')
            fig, ax = plt.subplots()
            ax.barh(np.arange(len(prediction_proba[0])), prediction_proba[0])
            ax.set_yticks(np.arange(len(prediction_proba[0])))
            ax.set_yticklabels(['Class 1', 'Class 2'])
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Confidence')
            st.pyplot(fig)
        except Exception as e:
            LOGGER.error(f"Error during prediction: {e}")
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model is not loaded. Predictions cannot be made.")

if __name__ == "__main__":
    run()
