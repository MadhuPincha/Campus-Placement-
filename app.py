import streamlit as st
import pandas as pd
import numpy as np
import pickle 

st.set_page_config(layout="wide")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Campus Placement Prediction</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: white;'>Student Details</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: white;'>Please enter student details for placement prediction</h5><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    gender = st.selectbox("**Gender**", ("M","F"))
    if gender == 'M':
        gender = 0
        st.write("Gender:",'**Male**')
    else:
        gender = 1
        st.write("Gender:",'**Female**')
        
    ssc_p = st.number_input('**Secondary Education percentage- 10th Grade**', min_value=0, max_value=100)
    st.write('10th Grade Percentage:',ssc_p)
    
    ssc_b = st.selectbox("**Board of Education(10th grade)**", ('Central', 'Others'))
    if ssc_b == 'Central':
        st.write("Board in 10th grade:",'**Central**')
    else:
        st.write("Board in 10th grade:",'**Others**')
        
    hsc_p = st.number_input('**Higher Secondary Education percentage- 12th Grade**', min_value=0, max_value=100)
    st.write('12th Grade Percentage:',hsc_p)

with col2:    
    hsc_b = st.selectbox("**Board of Education(12th grade)**", ('Central', 'Others'))
    if hsc_b == 'Central':
        st.write("Board in 12th grade:",'**Central**')
    else:
        st.write("Board in 12th grade:",'**Others**')
    
    hsc_s = st.selectbox("**Specialization in Higher Secondary Education(12th Grade)**", ('Science', 'Commerce', 'Other'))
    if hsc_s == 'Science':
        st.write("Specialization in 12th Gorade:", '**Science**')
    elif hsc_s == 'Commerce':
        st.write("Specialization in 12th Gorade:", '**Commerce**')
    else :
        st.write("Specialization in 12th Gorade:", '**Arts**')
        
    degree_p = st.number_input('**Degree Percentage**', min_value=0, max_value=100)
    st.write('Degree Percentage:',degree_p)
  
    degree_t = st.selectbox("**Under Graduation(Degree type)- Field of Degree Education**", ('Sci&Tech', 'Comm&Mgmt', 'Others'))
    if degree_t == 'Sci&Tech':
        st.write("Degree Type:", '**Sci&Tech**')
    elif degree_t == 'Comm&Mgmt':
        st.write("Degree Type:", '**Comm&Mgmt**')
    else:
        st.write("Degree Type:", '**Others**')
        
    
with col3:       
    workex = st.selectbox("**Work Experience**", ('Yes', 'No'))
    if workex == 'Yes':
        st.write("Gender:",'**Male**')
    else:
        st.write("Gender:",'**Female**')
    
    etest_p = st.number_input('**Employability test percentage ( conducted by college)**', min_value=0, max_value=100)
    st.write('Employability test percentage:',etest_p)

    specialisation = st.selectbox("**Post Graduation(MBA)- Specialization**", ('Mkt&HR', 'Mkt&Fin'))
    if specialisation == 'Mkt&HR':
        st.write("MBA- Specialization:",'**Mkt&HR**')
    else:
        st.write("MBA- Specialization:",'**Mkt&Fin**')
            
# Store the user input in a data frame
df = pd.DataFrame(
    np.array([[gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation]]), 
    columns=['Gender', '10th%', '10th_board', '12th%', '12th_board', '12th_specialization', 'graduation%', 'graduation_deg', 'Work_experience', 'Employee_test%', 'MBA_specialization'])

st.caption("Your details are:")
st.table(df)

# Load the saved model
import pickle
model = open('model_pkl', 'rb')
pickle_model = pickle.load(model)

# Load the saved pipeline
pipe = open('pipeline.pkl', 'rb')
pipeline = pickle.load(pipe)

# Prediction
if st.button("Predict Placement"):
    # Transform the data
    data_transformed = pipeline.transform(df)
    prediction = pickle_model.predict(data_transformed)
    
    if prediction == 0:
        st.success('The student will Not {} get placed'.format(prediction))
    else:
        st.success('The student will {} get placed'.format(prediction))
        

st.markdown("<h5 style='text-align: center; color: white;'>About The Model</h5>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'>This model is built using Support Vector Classifier(SVC) algorithm</h6>", unsafe_allow_html=True)