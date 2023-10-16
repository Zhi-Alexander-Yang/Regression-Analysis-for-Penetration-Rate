from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Define the base models and the final estimator
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('svr', SVR())
]
final_model = LinearRegression()

# Train the Stacking Regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_model)
stacking_model.fit(X_train, y_train)

# Evaluate the Stacking Regressor
y_pred_stack = stacking_model.predict(X_test)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
r2_stack = r2_score(y_test, y_pred_stack)
