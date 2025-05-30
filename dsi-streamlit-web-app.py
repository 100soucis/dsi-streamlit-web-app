
#import libraries

import streamlit as st 
import pandas as pd
import joblib

#load our podel pipeline object (from the pipeline advance code tutorial)

model = joblib.load('model.joblib')

st.title('Purchase Prediction Model')
st.subheader('Enter the customer information and submit for likelyhoo of purchase')

# we want to link streamlit to our code in spyder , so copy path link above and 
#go to prompt shell 

#Age input form

age=st.number_input(
    label="01,Enter the customer's age",
    min_value=18,
    max_value=120,
    value=35)

#Gender input form

gender=st.radio(
    label="02,Enter the customer's gender",
    options=['M','F'])


# Creditscore input

credit_score=st.number_input(
    label="03,Enter the customer's credit_score",
    min_value=0,
    max_value=1000,
    value=500)

#Submit inputs to model

if st.button('Submit for Prediction'):
    
    #store our data in a DataFrame for predictions
    new_data=pd.DataFrame({'age':[age],'gender':[gender],'credit_score':[credit_score]})
    
    # Apply model pipeline to the input data and extract probability prediction
    pred_proba=model.predict_proba(new_data)[0],[1]
    
    #Output prediction
    st.subheader(f"Based on these customer attributes, our model predict a purchase probabiltity of {pred_proba}")
    
    
    
    
    
    
    
    
    



