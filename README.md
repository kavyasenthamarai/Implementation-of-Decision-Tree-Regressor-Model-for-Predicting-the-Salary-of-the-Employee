# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Prepare your data

-Collect and clean data on employee salaries and features
-Split data into training and testing sets
2.Define your model

-Use a Decision Tree Regressor to recursively partition data based on input features
-Determine maximum depth of tree and other hyperparameters
3.Train your model

-Fit model to training data
-Calculate mean salary value for each subset
4.Evaluate your model

-Use model to make predictions on testing data
-Calculate metrics such as MAE and MSE to evaluate performance
5.Tune hyperparameters

-Experiment with different hyperparameters to improve performance
6.Deploy your model

Use model to make predictions on new data in real-world application.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KAVYA K
RegisterNumber:  212222230065
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2
```
dt.predict([[5,6]])
## Output:

Initial dataset:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/f70046fe-ca6f-41cc-ac40-afe6fc7ecbf7)

Data Info:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/5970e62f-78fa-4581-84f1-cdf80d38ef63)

Optimization of null values:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/003d24e0-b188-488e-af1a-79bd1e228306)

Converting string literals to numericl values using label encoder:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/35688cea-4e17-4182-9359-d4e30a1c526f)

Assigning x and y values:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/8ee5ed4c-bcb6-42dd-b6a7-fe1468c64e90)

Mean Squared Error:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/84eeb7d8-0aa7-491b-bce0-e4b64951d4d0)

R2 (variance):

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/c6619ed3-421f-4f5a-9211-6eef041a63e5)

Prediction:

![image](https://github.com/kavyasenthamarai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118668727/b078b440-de87-4314-9b2c-9e0d2a67a804)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
