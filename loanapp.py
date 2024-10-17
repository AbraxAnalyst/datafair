import streamlit as st
import pandas as pd
import pickle
import numpy as np

#import model 
pickle_in = open("xgb_model_.pkl","rb")
model = pickle.load(pickle_in)

df = pd.read_csv("credit.csv") 

#home_ownership
home = st.selectbox('Home Ownership', df["Home"].unique())

#loan_intent
loan_intent = st.selectbox('Loan Intent', df['Intent'].unique())

Age = int(st.number_input(label= "Customer Age", step=1))
customer_income = st.number_input(label= "Customer Income", step=500)
Emp_length = st.number_input(label= "Employment Duration", step=1)
Amount = st.number_input(label= "Loan Amount", step=1000)
Rate = st.number_input(label= "Rate")
Cred_length = st.number_input(label= "Duration", step=1)
Status = st.selectbox("Status", [1, 0])
Percent_income = st.number_input("Percent Income", min_value=0.0, max_value=1.0, step=0.1)

if st.button("Predict"):
    if home == 'RENT':
        home = 0
    elif home == 'OWN':
        home = 1
    elif home == 'MORTGAGE': 
        home = 2
    else:
        home = 3

    if loan_intent == 'PERSONAL':
        loan_intent = 0
    elif loan_intent == 'EDUCATION':
        loan_intent = 1
    elif loan_intent == 'MEDICAL':
        loan_intent = 2
    elif loan_intent == 'VENTURE':
        loan_intent = 3
    elif loan_intent == 'HOMEIMPROVEMENT':
        loan_intent = 4
    else:
        loan_intent = 5

    query = np.array([Age, customer_income, home, Emp_length, loan_intent, Amount, Rate, Status, Percent_income, Cred_length]).reshape(1, 10)

    result = model.predict_proba(query)[:, 1]  # Probability of class 1
    if result > 0.5:
        st.error("The applicant is likely to default on the loan.")
    else:
        st.success("The applicant is likely to repay the loan.")

