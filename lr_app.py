import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

from pyexpat import features

model=pickle.load(open('linear_regression_model.pkl','rb'))

st.title('Sales Ad Prediction ')
tv=st.text_input('Enter the Tv sales')
radio=st.text_input('Enter the radio sales')
newspaper=st.text_input('Enter the newspaper sales')

if st.button('Predict sales'):
    features=np.array([[tv,radio,newspaper]],dtype=np.float64)
    results=model.predict(features).reshape(1,-1)
    st.write('Predicted Sales is :',results[0])