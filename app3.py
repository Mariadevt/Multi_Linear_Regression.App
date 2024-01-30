#Import the necessary librariies for the web app

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Load the dataset

df = pd.read_excel("Dropoutdataset.xlsx")

#Add image of the organisation/Logo

st.image("Africdsa.jpeg")

#Add the title to our app

st.title("Multiple Linear Regression App")

#Add the header

st.header("Dataset Concept.", divider="rainbow")

st.write("""The Dropout dataset is a comprehensive collection of information related to students' academic performance and various socio-economic factors, 
            aimed at understanding the factors influencing students decisions to either graduate, dropout, or remain enrolled in educational institutions.
            This dataset includes features such as socio-economic background, parental education, academic scores, attendance,and extracurricular activities.
            In the context of multi-linear regression, researchers and 
            data scientists utilize this dataset to build predictive models that can assess the likelihood of a student either graduating, 
            dropping out, or remaining enrolled based on a combination of these factors. By employing multi-linear regression techniques, 
            the dataset allows for the examination of the relationships and interactions among multiple independent variables simultaneously. 
            The model seeks to identify which specific factors play a significant role in predicting the educational outcomes of students, 
            providing valuable insights for educators, policymakers, and institutions to implement targeted interventions and support systems for at-risk students. 
            Through the analysis of the Dropout dataset, it becomes possible to develop more informed strategies to improve overall student success and reduce dropout rates.""")

#Display the header

st.header("Explanatory Data Analysis(EDA).", divider="rainbow")

#Use of the check box

if st.checkbox("Dataset Info"):
    st.write("Dataset Info" ,df.info())
    
if st.checkbox("Number of Rows"):
    st.write("Number of Rows" ,df.shape[0])
     
if st.checkbox("Number of Columns"):
    st.write("Number of Columns" ,df.shape[1])
    
if st.checkbox("Column Names"):
    st.write("Column Names",df.columns.tolist())
    
if st.checkbox("Data Types"):
    st.write("Data Types" ,df.dtypes)
    
if st.checkbox("Missing Values"):
    st.write("Missing Values" ,df.isnull().sum())
    
if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary" ,df.describe())
    
#Visualisation Part of It

st.header("Visusalisation of the Dataset(VIZ).", divider="rainbow")
    
#Bar Chart 

if st.checkbox("Inflation Rate Against GDP  Bar Chart"):
    st.write("Bar Chart of Inflation Rate Against the GDP")
    st.bar_chart(x="Inflation rate", y="GDP", data=df,color=["#008000"])

#Bar Chart

if st.checkbox("Gender Bar Chart"):
    st.write("Bar Chart for Gender Against GDP")
    st.bar_chart(x="Gender", y="GDP", data=df,color=["#008000"])
    
#Line Chart

if st.checkbox("Inflation Rate Line Chart"):
    st.write("Line Chart of Inflation Rate Against the GDP")
    st.line_chart(x="Inflation rate", y="GDP", data=df,color=["#008000"])

# Scatter Plot

if st.checkbox("Scatter Plot"):
    st.write("Scatter Chart of GDP Against Terget")
    st.scatter_chart(x="Target", y="GDP", data=df,color=["#008000"])
    


#Encoding our target column using labelEncoder

university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])

#Use of the OneHotEncoder to encode the categorical features

ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),['Target'])], remainder='passthrough')
X = df.iloc[:,: -1]
y = df.iloc[:,-1]
y_encoded = ct.fit_transform(df[["Target"]])

#split the data into training and testing 

X_train ,X_test, y_train,Y_test = train_test_split(X,y_encoded, test_size=0.2, random_state=0)

#Fit our regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#User input for independent variables

st.sidebar.header('Enter the values to be Predicted', divider="rainbow")

#Create the input boxes for each feature

user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st. sidebar.text_input(f"Enter {feature}")
     
#Button to triger the prediction

if st.sidebar.button('Predict'):

    #create a dataframe for the user input
    user_input_df = pd. DataFrame([user_input],dtype=float)
    
#Predict using the trained model

    y_pred = regressor.predict(user_input_df)

#inverse transfrom to get the original target values

    predicted_class = university.inverse_transform(np.argmax(y_pred,axis=1))

#Display the predicted class/target

    st.write("Predicted Result Outcome", divider="rainbow")
    st.write(predicted_class[0])
    