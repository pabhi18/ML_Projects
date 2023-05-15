import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  
def prediction(R_D, admin, market):  
   
    prediction = classifier.predict(
        [[R_D, admin, market]])
    print(prediction)
    return prediction
      
def main():
    st.title('Estimated Profit of Any Startup:')
    st.markdown(" ")
    st.text('This model predict the profit of a Startup based on its investment in Research and ')
    st.text('Development (R&D), Administration, and Marketing Spend.')
    st.markdown(" ")


    R_D = st.number_input("Enter Amount Spend in Research and development:")
    admin = st.number_input("Enter Administration expenses:")
    market = st.number_input("Enter Marketing Spend Amount:")
    st.markdown(" ")
    result =" "
      

    if st.button("Predict"):
        result = prediction(R_D, admin, market)
        result = float(result)
    st.success('Estimated Profit: ₹ {} '.format(result))
     
if __name__=='__main__':
    main()