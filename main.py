import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write('''
# Prediction insufisance cardiaque
''')

st.sidebar.header("Informations Du Patient")

def user_input():
    Age=st.sidebar.slider('Age',1,25,100)
    #Sex=st.text_input('Sex', 'M',max_chars=1)
    Sex = st.selectbox('Sex:',('M', 'F'))

    #ChestPainType=st.text_input('ChestPainType', 'ASY',max_chars=3)
    ChestPainType = st.selectbox('ChestPainType:',('ASY', 'NAP','ATA','TA'))

    RestingBP=st.sidebar.slider('RestingBP',0,250,100)
    Cholesterol=st.sidebar.slider('Cholesterol',0,250,650)
    FastingBS= st.sidebar.slider('FastingBS',0,1,1)
    #RestingECG=st.text_input('RestingECG', 'Normal',max_chars=6)
    RestingECG = st.selectbox('RestingECG:',('Normal', 'LVH','ST'))

    MaxHR= st.sidebar.slider('MaxHR',50,250,100)
    #ExerciseAngina=st.text_input('ExerciseAngina', 'Y',max_chars=1)
    ExerciseAngina = st.selectbox('ExerciseAngina:',('Y', 'N'))

    Oldpeak= st.sidebar.slider('Oldpeak',-3,7,5)
    #ST_Slope=st.text_input('ST_Slope', 'Flat',max_chars=4)
    ST_Slope = st.selectbox('ST_Slope:',('Flat', 'Up','Down'))

    data={
    'Age':Age,
    'Sex':Sex,
    'ChestPainType':ChestPainType,
    'RestingBP':RestingBP,
    'Cholesterol':Cholesterol,
    'FastingBS':FastingBS,
    'RestingECG':RestingECG,
    'MaxHR':MaxHR,
    'ExerciseAngina':ExerciseAngina,
    'Oldpeak':Oldpeak,
    'ST_Slope':ST_Slope,
    }
    heart_parametres=pd.DataFrame(data,index=[0])
    return heart_parametres

df_pred=user_input()

st.subheader('Données saisies')
st.write(df_pred)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


path = "heart.csv"
df = pd.read_csv(path, encoding = "ISO-8859-1")

trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
categorical_cols= df.select_dtypes('object').columns

enc = OneHotEncoder(handle_unknown='ignore')
transformed_trainset = enc.fit_transform(trainset[categorical_cols]).toarray()
transformed_testset = enc.transform(testset[categorical_cols]).toarray()
# the above transformed_data is an array so convert it to dataframe
encoded_data_trainset = pd.DataFrame(transformed_trainset, index=trainset.index)
encoded_data_testset = pd.DataFrame(transformed_testset, index=testset.index)

# now concatenate the original data and the encoded data using pandas
trainset = pd.concat([trainset, encoded_data_trainset], axis=1)
trainset = trainset.drop(categorical_cols,axis=1)
  

testset = pd.concat([testset, encoded_data_testset], axis=1)
testset=testset.drop(categorical_cols,axis=1)  

X_train = trainset.drop('HeartDisease', axis=1)
y_train = trainset['HeartDisease']

X_test = testset.drop('HeartDisease', axis=1)
y_test = testset['HeartDisease']

preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=5))


RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
Linear_SVC=make_pipeline(StandardScaler(),LinearSVC())
Logistic_Regression=make_pipeline(StandardScaler(),LogisticRegression())

model=StackingClassifier([
                         ('RandomForest',RandomForest) ,   
                         ('LogisticRegression',Logistic_Regression),
                        ],
                        final_estimator= Linear_SVC)

model.fit(X_train,y_train)


transformed_pred = enc.transform(df_pred[categorical_cols]).toarray()
encoded_data_pred = pd.DataFrame(transformed_pred, index=df_pred.index)
df_pred = pd.concat([df_pred, encoded_data_pred], axis=1)
df_pred = df_pred.drop(categorical_cols,axis=1)



prediction=model.predict(df_pred)

st.subheader("Prediction")
if prediction==0:
    st.write('Le patient n''a pas une insufisance cardiaque')
else:
    st.write('Le patient a une insufisance cardiaque')


