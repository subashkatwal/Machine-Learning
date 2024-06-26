import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("C:/Users/Dell/Desktop/Machine Learning/PasswordStrengthChecker/data.csv")
# print(data.head())

data=data.dropna()
data["strength"]=data["strength"].map({0:"Weak",
                                       1:"Medium",
                                       2:"Strong"})
print(data.sample(3))

#tokenize the passwords

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
x=np.array(data["password"])
y=np.array(data["strength"])

tdif=TfidfVectorizer(tokenizer=word)
x=tdif.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.05,random_state=40)

model=RandomForestClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

#check strength of password using trained model 

import getpass
user=input("Enter password :")
data=tdif.transform([user]).toarray()
output=model.predict(data)
print(output)