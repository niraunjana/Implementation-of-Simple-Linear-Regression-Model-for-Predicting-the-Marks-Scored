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

![image](https://user-images.githubusercontent.com/119395610/229364153-6c063580-2abc-47af-bc4a-0f6d491d42e5.png)

![image](https://user-images.githubusercontent.com/119395610/229364166-d11da734-aadf-4cbb-be3e-b90311e20fcf.png)
![image](https://user-images.githubusercontent.com/119395610/229364186-e6ca66c3-cdd2-4171-9f6f-5e4f499d40e0.png)
![image](https://user-images.githubusercontent.com/119395610/229364204-c7b8fe42-97c6-4158-a251-6acddaebf7fa.png)
![image](https://user-images.githubusercontent.com/119395610/229364223-166cafb3-6b44-4ad2-a20d-cf41938b160f.png)
![image](https://user-images.githubusercontent.com/119395610/229364272-bf893e77-1da0-4cda-bddf-d04f5e76eada.png)
![image](https://user-images.githubusercontent.com/119395610/229364284-fa1a79eb-f160-45cc-990f-249a19312bec.png)
![image](https://user-images.githubusercontent.com/119395610/229364297-ed3363ee-c87b-4271-b796-83c4b6ee5d30.png)
![image](https://user-images.githubusercontent.com/119395610/229364320-c0a1f593-85b8-4eaa-a952-1ec80e981ffa.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
