import pandas as pd
# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# One-hot encode categorical variables
input_df = pd.get_dummies(input_df, columns=['island', 'sex'])

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
import os
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import subprocess
import os

LOGGER = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Predictive Analytics App",
    page_icon="📊",
)

# Run the create_dummy_model.py script to regenerate the model.pkl file
try:
    subprocess.run(["python3", "create_dummy_model.py"], check=True)
    LOGGER.info("Dummy model created successfully.")
    if os.path.exists('model.pkl'):
        LOGGER.info("model.pkl file is present in the directory.")
    else:
        LOGGER.error("model.pkl file is not present in the directory.")
except subprocess.CalledProcessError as e:
    LOGGER.error(f"Error creating dummy model: {e}")
    st.error(f"An error occurred while creating the dummy model: {e}")

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

    # Collects user input features into dataframe
    # uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # if uploaded_file is not None:
    #     input_df = pd.read_csv(uploaded_file)
    # else:
    def user_input_features():
        feature1 = st.sidebar.number_input('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
        feature2 = st.sidebar.number_input('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
        feature3 = st.sidebar.number_input('Feature 3', min_value=0.0, max_value=100.0, value=50.0)
        feature4 = st.sidebar.number_input('Feature 4', min_value=0.0, max_value=100.0, value=50.0)
        data = {'Feature 1': feature1,
                'Feature 2': feature2,
                'Feature 3': feature3,
                'Feature 4': feature4}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = pd.read_csv('/home/ubuntu/streamlit-hello/penguins_example.csv')

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['island', 'sex'])

    # Displays the user input features
    st.subheader('User Input features')
    st.write(input_df)

    # Apply model to make predictions
    if model is not None:
        try:
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
            st.error("An error occurred during prediction. Please check the input data and try again.")
    else:
        st.error("Model is not loaded. Predictions cannot be made.")

if __name__ == "__main__":
    run()

# Trivial change to trigger redeployment
# Another trivial change to trigger redeployment
