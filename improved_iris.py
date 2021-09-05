# Importing the necessary libraries.
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Loading the dataset.
iris_df=pd.read_csv('iris-species.csv')

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label']=iris_df['Species'].map({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2})

# Creating features and target DataFrames.
features=iris_df[['SepalLengthCm','PetalLengthCm','SepalWidthCm','PetalWidthCm']]
target=iris_df['Label']

# Splitting the dataset into train and test sets.
features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.3,random_state=42)

# Creating an SVC model. 
svc_model=SVC(kernel='linear')
svc_model.fit(features_train,target_train)

# Creating a Logistic Regression model. 
lr_model=LogisticRegression(n_jobs=-1)
lr_model.fit(features_train,target_train)

# Creating a Random Forest Classifier model.
rf_clf=RandomForestClassifier(n_jobs=-1,n_estimators=100)
rf_clf.fit(features_train,target_train)


# Create a function that accepts an ML mode object say 'model' and the four features of an Iris flower as inputs and returns its name.
@st.cache()
def prediction(model,sepal_length,sepal_width,petal_length,petal_width):
  species=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
  species=species[0]
  if species==0:
    return 'Iris-setosa'
  elif species==1:
    return 'Iris-virginica'
  else:
    return 'Iris-versicolor'


# Add title widget
st.sidebar.title("Iris Species Prediction App")      

# Add 4 sliders and store the value returned by them in 4 separate variables. 
s_len = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
# The 'float()' function converts the 'numpy.float' values to Python float values.
s_wid = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_len = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_wid = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

# Add a select box in the sidebar with the 'Classifier' label.
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button('Predict'):
  if classifier=='Support Vector Machine':
    output=prediction(svc_model,s_len,s_wid,p_len,p_wid)
    score=svc_model.score(features_train,target_train)
  elif classifier=='Logistic Regression':
    output=prediction(lr_model,s_len,s_wid,p_len,p_wid)
    score=lr_model.score(features_train,target_train)
  else:
    output=prediction(rf_clf,s_len,s_wid,p_len,p_wid)
    score=rf_clf.score(features_train,target_train)
  
  st.write('Species predicted: ',output)
  st.write('Accuracy score: ',score)