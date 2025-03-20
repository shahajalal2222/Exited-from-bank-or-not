import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf
import pickle 

model=tf.keras.models.load_model('model.h5')

with open('Label.pickle','rb') as file:
 Label=pickle.load(file)
with open('onehot.pickle','rb') as file:
 onehot=pickle.load(file)
with open('scaler.pickle','rb') as file:
 scaler=pickle.load(file)

st.title('Customer churn prediction')
geopraphy=st.selectbox('Geography',onehot.categories_[0])
gender=st.selectbox('Gender',Label.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
number_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Has Active Member ',[0,1])

data={
 'Gender':[gender],
 'Age':[age],
 'Balance':[balance],
 'CreditScore':[credit_score],
 'EstimatedSalary':[estimated_salary],
 'Tenure':[tenure],
 'NumberOfProducts':[number_of_products],
 'HasCreditCard':[has_cr_card],
 'IsActiveMember':[is_active_member]
}
df=pd.DataFrame(data)
df['Gender']=Label.transform(df['Gender'])
encoded_df=onehot.transform([[geopraphy]]).toarray()
encoded_df=pd.DataFrame(encoded_df,columns=onehot.get_feature_names_out(['Geography']))
DF=pd.concat([df,encoded_df],axis=1)

predict_proba=model.predict(DF)
value=predict_proba[0][0]
if value>0.5:
 st.write('customer is like to churn')
else :
 st.write('customer is not is like to churn')