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

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

LOGGER = get_logger(__name__)

# Load pre-trained model
model = joblib.load('model.pkl')

def run():
    st.set_page_config(
        page_title="Predictive Analytics App",
        page_icon="📊",
    )

    st.write("# Predictive Analytics App 📊")

    st.sidebar.header("User Input Features")
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
    """)

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
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
        input_df = user_input_features()

    # Displays the user input features
    st.subheader('User Input features')
    if uploaded_file is not None:
        st.write(input_df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(input_df)

    # Apply model to make predictions
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

if __name__ == "__main__":
    run()
