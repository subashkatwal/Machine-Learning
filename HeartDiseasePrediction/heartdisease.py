import pandas as pd 
import pylab as plt 
import numpy as np 
import scipy.optimize as opt 
from sklearn.discriminant_analysis import StandardScaler
import statsmodels.api as sm 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns 


diseases_df = pd.read_csv('C:/Users/Dell/Desktop/ML/Machine-Learning/HeartDiseasePrediction/framingham.csv')

# print(diseases_df.head(5))

diseases_df.drop(['education'],inplace=True , axis=1)
diseases_df.rename(columns={'male':'Sex_male'},inplace=True)
# print(diseases_df)

# '''removing NaN / NULL values'''
diseases_df.dropna(axis=0,inplace=True)
print(diseases_df.head(),diseases_df.shape) #.shape determines the dimension of rows and columns
print(diseases_df.TenYearCHD.value_counts())# value_counts calculate the frequency of each unique value in column


# #splitting data into Dataset (Test and Train )

X=np.asarray(diseases_df[['age','Sex_male','cigsPerDay','totChol', 'sysBP', 'glucose',]])
y=np.asarray(diseases_df['TenYearCHD'])

x=preprocessing.StandardScaler().fit(X).transform(X)

#Train & Test 
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

print('Train set :',X_train.shape,Y_train.shape)
print('Test set :',X_test.shape,Y_test.shape)

plt.figure(figsize=(7,5))
sns.countplot(x='TenYearCHD',data=diseases_df)
plt.show()

'''Counting no of people affected bu CHD where 0--> not affected & 1-> affected'''
laste= diseases_df['TenYearCHD'].plot()
plt.show()

from sklearn.linear_model import LogisticRegression 
logReg=LogisticRegression()
logReg.fit(X_train,Y_train)
y_Pred=logReg.predict(X_test)

from sklearn.metrics import accuracy_score 
print('Accuracy of the model is = ',accuracy_score(Y_test,y_Pred))


# Confusing matrix
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(Y_test,y_Pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicated : 0',
                                          'Predicted :1 '],
                         index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="Greens")
plt.show()

print('The details for confusion matrix is = ')
print(classification_report(Y_test,y_Pred))

age = int(input("Enter age: "))
sex = int(input("Enter Sex (1 = Male, 0 = Female): "))
cigs = float(input("Cigs per day: "))
chol = float(input("Total cholesterol: "))
bp = float(input("Systolic BP: "))
glucose = float(input("Glucose level: "))

input_df = pd.DataFrame([{
    "age": age,
    "Sex_male": sex,
    "cigsPerDay": cigs,
    "totChol": chol,
    "sysBP": bp,
    "glucose": glucose
}])

new_data= pd.DataFrame(input_df)
scaler=preprocessing.StandardScaler().fit(X)
scaled_input= scaler.transform(new_data)

prediction=logReg.predict(scaled_input)

print(" Prediction for new user : \n ")
print("Input :",new_data)

print("Predicted Risk (TenYearCHD):", int(prediction[0]), "→", "Risk" if prediction[0] == 1 else "No Risk")
