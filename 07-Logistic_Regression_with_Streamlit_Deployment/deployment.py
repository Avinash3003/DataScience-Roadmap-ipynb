''' Requirements)
Terminal:- 
packages Installation - pip install streamlit
To run - streamlit run filename.py    (streamlit run deployment.py)
'''


import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


scaler = pickle.load(open('Scaler.sav','rb'))
loaded_model = pickle.load(open('LogisticRegression_model.sav','rb'))
dataset = pd.read_csv('Titanic_test.csv')


st.title('Titanic Survival Prediction App')

st.sidebar.header('User parameter Inputs')
st.sidebar.subheader('Enter Parameters to Predict')
def userInputParams():

    Pclass = st.sidebar.selectbox('Enter Pclass',[1,2,3])
    Sex = st.sidebar.selectbox('Enter Sex (Male=0,Female=1)',[0,1])
    Age = st.sidebar.number_input('Insert the Age',step=1)
    SibSp = st.sidebar.slider('SibSp',0,10)
    Parch = st.sidebar.slider('Parch',0,8)
    Fare = st.sidebar.number_input('Fare')
    Embarked = st.sidebar.selectbox("Port of Embarkation", ["Q", "S"])

    Embarked_Q,Embarked_S = 0,0

    if Embarked == 'Q':
        Embarked_Q = 1
    else:
        Embarked_S = 1

    return np.array([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked_Q,Embarked_S]])



Inputs = scaler.transform(userInputParams())


if st.button('Predict:)'):
    y_pred = loaded_model.predict(Inputs)
    y_pred_prob = loaded_model.predict_proba(Inputs)
    print(y_pred,y_pred_prob)
    st.subheader('Predicted Result')
    st.write('Survived' if y_pred[0]==1 else 'Did not Survived')
    st.write(f"**Probability of Not Surviving:** {y_pred_prob[0][0]:.2f}")
    st.write(f"**Probability of Surviving:** {y_pred_prob[0][1]:.2f}")














