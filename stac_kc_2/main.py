import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pandas.read_csv('extended_salary_data.csv')
print(data)

stats=data.describe()
stats=stats.transpose()

stats['median']=data.median()
stats['variance']=data.var()

print(stats)

plt.scatter(data.YearsExperience,data.Salary,alpha=0.5,color='green')
plt.title('Years of Exp v/s Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

X=data[['YearsExperience']] #sklearn requires 2D not series
y=data['Salary']

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=42)
# print(f'{X_train.shape[0]}, {X_test.shape[0]}') # in how many rows its divided

model=LinearRegression()
model.fit(X_train,y_train)

slope=model.coef_[0]
intercept=model.intercept_
print(f'{slope:.2f}, <> ,{intercept:.2f}')

plt.scatter(X_train, y_train,color='blue',label='Training Data')
plt.scatter(X_test, y_test,color='green',label='Testing Data')

X_range= np.linspace(X.min(),X.max(),100).reshape(-1,1) #start,stop,100 evenly spaced values

y_prediction=model.predict(X_range)

plt.plot(X_range, y_prediction, color='red', label='Regression Line') #line plot

plt.title('Linear Regression_train')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

y_prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, y_prediction)
mse = mean_squared_error(y_test, y_prediction)

print(f"{mae:.2f}")
print(f"{mse:.2f}")

plt.scatter(data['YearsExperience'], data['Salary'], color='blue', alpha=0.7, label='Data Points')

X_range = np.linspace(X['YearsExperience'].min(), X['YearsExperience'].max(), 100).reshape(-1, 1)  # Generate X values
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, color='red', label='Regression Line')

plt.title('Linear Regression_MAE_MSE_test')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()