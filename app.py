import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load Model
model = joblib.load('Models/model.pkl')

st.title('🏥Medical Insurance Prediction')
st.markdown("""

In order to predict your Medical Insurance, select the information and click on predict


""")

st.sidebar.title('Medical Information')
st.sidebar.write('Select your details')

age = st.sidebar.number_input(
    
    'Age',
    min_value = 18,
    max_value = 90,
    value = 18
    
)

sex = st.sidebar.selectbox(
    
    'Gender',
    ['male', 'female']
    
)

bmi = st.sidebar.number_input(
    
    'BMI',
    min_value = 8,
    max_value = 90,
    value = 10
    
)

children = st.sidebar.number_input(
    'Number of Children',
    min_value = 0,
    max_value = 10,
    value = 0
    
)

smoker = st.sidebar.selectbox(
    
    'Smoking Status',
    ['no', 'yes']
    
)

region = st.sidebar.selectbox(
    
    'Region',
    ['southwest', 'southeast', 'northwest', 'northeast']
    
)


sampleData = pd.DataFrame({

    'age' : [age],
    'sex' :	[sex],
    'bmi' : [bmi],	
    'children' : [children],	
    'smoker' :	[smoker],
    'region' : [region],
        
})

if st.button('Predict Insurance Cost'):
    
    predictions = np.round(model.predict(sampleData), 2)
    
    st.success(predictions[0])
    st.write('ML Projects - Using Regression')
    
