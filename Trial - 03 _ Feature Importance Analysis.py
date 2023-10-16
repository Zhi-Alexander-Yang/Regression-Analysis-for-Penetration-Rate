# Get feature importances
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Remove the least important features
least_important_features = importance_df[importance_df['Importance'] < 0.01]['Feature']
X_train = X_train.drop(least_important_features, axis=1)
X_test = X_test.drop(least_important_features, axis=1)
