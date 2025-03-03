# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Lakshmen Prashanth R
RegisterNumber:  212224230137
*/
```
```py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


```

## Output:
![image](https://github.com/user-attachments/assets/98fcdf32-8242-48b5-9569-706179650726)

![image](https://github.com/user-attachments/assets/4962e5e3-502b-4413-a707-5e8f1cd24068)

![image](https://github.com/user-attachments/assets/a47cd3ad-8c01-4b32-9710-8531b4305891)

![image](https://github.com/user-attachments/assets/76ae1030-a9b6-4402-b32f-4a3b42568b4b)

![image](https://github.com/user-attachments/assets/1fdf08bc-5ad7-4b7d-ad01-c830dfad80b9)

![image](https://github.com/user-attachments/assets/c0d7bf91-61da-4504-b378-43f015b5d10a)

![image](https://github.com/user-attachments/assets/1d9d16a5-c474-493b-89c0-34987829259f)

![image](https://github.com/user-attachments/assets/6dd65f08-6f1f-4ada-891b-f59b25d74078)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
