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

![WhatsApp Image 2023-04-02 at 21 55 39](https://user-images.githubusercontent.com/119395610/229366335-9cc7b718-6d6e-4461-bd62-1ba130da19dc.jpg)

![WhatsApp Image 2023-04-02 at 22 11 34](https://user-images.githubusercontent.com/119395610/229366706-bdb4ac2d-fff8-41c9-9325-b3bfd965755d.jpg)


![WhatsApp Image 2023-04-02 at 21 57 53](https://user-images.githubusercontent.com/119395610/229366382-c6623bd1-c1d2-45fe-8eed-097e51e4377d.jpg)

![WhatsApp Image 2023-04-02 at 21 59 04](https://user-images.githubusercontent.com/119395610/229366392-1a088fe3-5d91-4be7-b633-a4099d093e27.jpg)

![WhatsApp Image 2023-04-02 at 22 00 06](https://user-images.githubusercontent.com/119395610/229366409-9cd96c06-cc84-4554-a310-314b472c8aa3.jpg)

![WhatsApp Image 2023-04-02 at 22 01 46](https://user-images.githubusercontent.com/119395610/229366427-12194773-8f66-49bc-b0a2-0dda094d4598.jpg)

![WhatsApp Image 2023-04-02 at 22 02 46](https://user-images.githubusercontent.com/119395610/229366450-47119646-2ac7-4952-966a-9f9eca54ec57.jpg)

![WhatsApp Image 2023-04-02 at 22 08 14](https://user-images.githubusercontent.com/119395610/229366507-6c7b4359-8623-4eb1-802f-35cffc84d067.jpg)

![WhatsApp Image 2023-04-02 at 22 04 50](https://user-images.githubusercontent.com/119395610/229366518-c390a13b-a4ea-454a-9212-38b05af440bc.jpg)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
