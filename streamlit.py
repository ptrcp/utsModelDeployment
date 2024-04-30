import streamlit as st
import pickle
import numpy as np
import pandas as pd

pickle_in = open("OOPmodel4.pkl", 'rb')
model = pickle.load(pickle_in)

def make_dataframe(data_arr, columns):
    dictionary = {
        'CreditScore': 0.0, 'Geography': 0, 'Gender': 0, 'Age': 0.0, 'Tenure': 0, 'Balance': 0.0,
       'NumOfProducts': 0, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 0.0
    }
    
    for i, val in enumerate(data_arr) :
        dictionary[columns[i]] = val

    df = pd.DataFrame(dictionary, index=[0])
    return df

def make_prediction(data):
    prediction = model.predict(data)
    return prediction

def main():
    #st.title(':red[Customer Churn Prediction]')
    st.title('Customer Churn Prediction')
    #st.image('/Users/patriciapepita/Documents/4th Semester/Model Deployment/Customer-churn-header.png')
    
    # INPUT 1
    st.text("")
    st.subheader('Customer Credit Score')
    credit_score = st.number_input('Input the value below', min_value=300.0,
                                   max_value=850.0, value=300.0)
    st.write('*Answer:* ', credit_score)
    
    # INPUT 2
    st.text("")
    st.subheader('Customer Location')
    st.write('0: France | 1: Germany | 2: Spain')
    geography = st.selectbox('Select the option below', [0, 1, 2])
    if geography == 0:
        st.write('*Answer: France*')
    elif geography == 1:
        st.write('*Answer: Germany*')
    else:
        st.write('*Answer: Spain*')
        
    # INPUT 3
    st.text("")
    st.subheader('Customer Gender')
    st.write('0: Female | 1: Male')
    gender = st.radio('Choose the option below', [0, 1])
    if gender == 0:
        st.write('*Answer: Female*')
    else:
        st.write('*Answer: Male*')
    
    # INPUT 4
    st.text("")
    st.subheader('Customer Age')
    st.write('0: Female | 1: Male')
    age = st.number_input('Input the number below', min_value=17, max_value=100, value=17)
    st.write('*Answer:*', age, '*years old*')

    # INPUT 5
    st.text("")
    st.subheader('Customer Tenure')
    tenure = st.slider('Choose the number below', min_value=0, max_value=20, value=0)
    st.write('*Answer:*', tenure)

    # INPUT 6
    st.text("")
    st.subheader('Customer Balance Value')
    balance = st.number_input('Input the value below', min_value=0)
    st.write('*Answer:* ', balance)
    
    # INPUT 7
    st.text("")
    st.subheader('Number of Products Owned')
    product = st.slider('Choose the number below', min_value=0, max_value=5, value=0)
    st.write('*Answer:*', product)

    # INPUT 8
    st.text("")
    st.subheader('Do They Have a Credit Card?')
    st.write('0: No | 1: Yes')
    crcard = st.radio('Choose the option below', [0, 1], key='cc')
    if crcard == 0:
        st.write('*Answer: No*')
    else:
        st.write('*Answer: Yes*')
        
    # INPUT 9
    st.text("")
    st.subheader('Do They Have an Active Membership?')
    st.write('0: No | 1: Yes')
    active = st.radio('Choose the option below', [0, 1], key='member')
    if active == 0:
        st.write('*Answer: No*')
    else:
        st.write('*Answer: Yes*')

    # INPUT 10
    st.text("")
    st.subheader('Customer Salary')
    salary = st.number_input('Input the estimation below', min_value=0.00,
                             max_value=None, value=0.00)
    st.write('*Answer:*', salary)

    #df = make_dataframe([credit_score,geography,gender,age,tenure, balance,product,crcard,active,salary], ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    
    #st.dataframe(df)
    
    st.text("")
    st.text("")
    if st.button('Create Prediction'):
        features = [credit_score,geography,gender,age,tenure,
                   balance,product,crcard,active,salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
        if result == 1:
            st.subheader(':red[**Churn Customer**]')
            #st.image('Documents/4th Semester/Model Deployment/churnCustomer.png')
        else:
            st.subheader(':green[**Loyal Customer**]')
            #st.image('Documents/4th Semester/Model Deployment/loyalCustomer.png')


def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
