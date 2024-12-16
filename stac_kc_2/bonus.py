from main import *

np.random.seed(42)
data['ProjectsCompleted'] = np.random.randint(1, 20, size=len(data))

print(data)

X_simple = data[['YearsExperience']]  # Simple linear regression with only Years of Experience

X_multiple = data[['YearsExperience', 'ProjectsCompleted']]  # Multiple linear regression

y = data['Salary'] # Target variable

X_simple_train, X_simple_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_multiple_train, X_multiple_test = train_test_split(X_multiple, test_size=0.2, random_state=42)


# Train Simple
simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

# Predict simple
y_simple_pred = simple_model.predict(X_simple_test)

# Evaluate simple
simple_mae = mean_absolute_error(y_test, y_simple_pred)
simple_mse = mean_squared_error(y_test, y_simple_pred)

print(f"Simple MAE: {simple_mae:.2f}, MSE: {simple_mse:.2f}")

# Train Multiple
multiple_model = LinearRegression()
multiple_model.fit(X_multiple_train, y_train)

# Predict multiple
y_multiple_pred = multiple_model.predict(X_multiple_test)

# Evaluate multiple
multiple_mae = mean_absolute_error(y_test, y_multiple_pred)
multiple_mse = mean_squared_error(y_test, y_multiple_pred)

print(f"Multiple MAE: {multiple_mae:.2f}, MSE: {multiple_mse:.2f}")

plt.scatter(y_test, y_simple_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.scatter(y_test, y_multiple_pred, color='green')
plt.title('Simple LR-blue, Multiple LR-green')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.show()
