import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model
loaded_model = pickle.load(open('cancer_prediction_model.sav', 'rb'))

# Create a function for the cancer prediction system
def predict(input_data):
    # Convert the input_data to a numpy array
    ip_to_numpy = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    ip_reshape = ip_to_numpy.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(ip_reshape)
    
    if prediction[0] == 0:
        return "The patient is fine."
    else:
        return "The patient has cancer."

# Define the main function for the Streamlit app
def main():
    # Custom CSS for title
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the image
    st.image('C:/Users/HP/Downloads/NoteBook of ML/cancer.png', use_column_width=True)
    
    # Display the title
    st.markdown('<h1 class="title">🩺📊 Cancer Prediction System🌟🔬</h1>', unsafe_allow_html=True)

    # Get user data as input
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Select your gender:", options=["Male", "Female"])
    gender = 0 if gender == "Male" else 1
    bmi = st.number_input("Enter your BMI value:", format="%.6f")
    sm = st.selectbox("Are you a smoker?", options=["Yes", "No"])
    sm = 1 if sm == "Yes" else 0
    gr = st.selectbox("Enter your Genetic Risk value:", options=[0, 1, 2])
    pa = st.number_input("Enter your Physical Activity value:", format="%.6f")
    al = st.number_input("Enter your Alcohol Consumption value:", format="%.6f")
    ch = st.selectbox("Do you have any cancer history?", options=["Yes", "No"])
    ch = 1 if ch == "Yes" else 0

    input_data = (age, gender, bmi, sm, gr, pa, al, ch)

    # Code for prediction
    result = ""

    # Create a button with custom color
    if st.button('Result'):
        result = predict(input_data)
        st.success(result)

if __name__ == '__main__':
    main()
