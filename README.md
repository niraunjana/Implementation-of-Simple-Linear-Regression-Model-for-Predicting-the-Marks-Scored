# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the needed packages.
2. Assigning hours To X and Scores to Y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formmula to find the values.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NIRAUNJANA GAYATHRI G.R.
RegisterNumber:  22008369
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')
print('df.head')

df.head()

print("df.tail")
df.tail()

X=df.iloc[:,:-1].values
print("Array of X")
X

Y=df.iloc[:,1].values
print("Array of Y")
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

![image](https://user-images.githubusercontent.com/119395610/229364495-07eb95b0-a7dc-423c-b784-a5e821f05e3e.png)
![image](https://user-images.githubusercontent.com/119395610/229364510-ebeb1033-aa53-4efb-ad32-34cc49f65c4a.png)
![image](https://user-images.githubusercontent.com/119395610/229364521-a2ceadd2-d1f6-4353-91d4-eb0e05243239.png)
![image](https://user-images.githubusercontent.com/119395610/229364528-3c63a134-0733-46a8-9ad0-0ef3f7c5766e.png)
![image](https://user-images.githubusercontent.com/119395610/229364536-8f5a929e-1353-4c35-9fda-5516b77c6567.png)
![image](https://user-images.githubusercontent.com/119395610/229364550-3c668f61-6b50-4a1a-8754-6f9475c595fb.png)
![image](https://user-images.githubusercontent.com/119395610/229364568-ce538746-1769-45a2-abd1-279cddbf8941.png)
![image](https://user-images.githubusercontent.com/119395610/229364579-f81830ce-8f3e-45a5-a3ed-ff1aad58ccba.png)
![image](https://user-images.githubusercontent.com/119395610/229364590-703676f7-b33b-4885-9d64-4771d9411e7a.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
