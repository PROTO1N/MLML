import pandas as pd 
import numpy as np 
import joblib 
import streamlit as st 
model=joblib.load('kath_lr.pkl')
features=joblib.load('kath_lr_cols.pkl')

# creating a function to predict values 
def pred(new_data): 
    x=model.predict(new_data)
    return x


# getting user data 
sepal_length= st.number_input("enter sepal length")
sepal_width=st.number_input("enter sepal width")
petal_length=st.number_input("petal length")
petal_width=st.number_input("petal width")

# Creating Data Frame 
if st.button("predict"): 
    input_data={
        'sepal_length':[sepal_length],
        'sepal_width':[sepal_width],
        'petal_length':[petal_length],
        'petal_width':[petal_width]
    }
    input_data=pd.DataFrame(input_data)

    pred=input_data[features]
    # Display prediction result
    st.write(f"The predicted flower species is: {pred[0]}")
