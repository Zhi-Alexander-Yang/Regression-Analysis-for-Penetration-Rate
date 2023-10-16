from sklearn.ensemble import RandomForestRegressor

# Define the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Fit the model to the training data
rf_model.fit(X_train_preprocessed, y_train)

# Predict on the training set
y_train_pred_rf = rf_model.predict(X_train_preprocessed)

# Predict on the test set
y_test_pred_rf = rf_model.predict(X_test_preprocessed)

# Evaluate the training and test performance
train_rmse_rf = mean_squared_error(y_train, y_train_pred_rf, squared=False)
test_rmse_rf = mean_squared_error(y_test, y_test_pred_rf, squared=False)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

train_rmse_rf, test_rmse_rf, train_r2_rf, test_r2_rf

# RESULT
# (0.09496765705386424,
#  0.1348074366036578,
#  0.9813417434748433,
#  0.9566427935308215)

# Training RMSE: 0.09497
# Test RMSE: 0.13481
# Training R2: 0.98134
# Test R2: 0.95664
