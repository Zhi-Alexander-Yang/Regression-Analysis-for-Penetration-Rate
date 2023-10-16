from sklearn.ensemble import RandomForestRegressor

# Initialize the RandomForestRegressor model
rf_model_original = RandomForestRegressor(random_state=42)

# Train the model on the preprocessed training data
rf_model_original.fit(X_train_preprocessed_original, y_train_original)

# Predict on the test set
y_test_pred_original = rf_model_original.predict(X_test_preprocessed_original)

# Calculate and print the RMSE and R^2 scores
rmse_original = mean_squared_error(y_test_original, y_test_pred_original, squared=False)
r2_original = r2_score(y_test_original, y_test_pred_original)

rmse_original, r2_original

# RESULT
# (0.135910434023043, 0.9559303917225166)
