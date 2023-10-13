from sklearn.model_selection import GridSearchCV
# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_rf = pd.read_csv(file_path)
df_rf = df_rf.dropna()
df_rf["Hole id"] = df_rf["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Selecting original features and target variable
X_rf = df_rf[features_unified]
y_rf = df_rf[target_unified]

# Splitting the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Standardizing the features
scaler_rf = StandardScaler()
X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
X_test_rf_scaled = scaler_rf.transform(X_test_rf)

# Defining a specific hyperparameter grid for Grid Search
param_grid_specific = {
    'n_estimators': [100, 150],
    'max_depth': [None, 20],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2]
}

# Setting up the Grid Search with specific values
grid_search_specific = GridSearchCV(estimator=model_rf, param_grid=param_grid_specific, scoring='neg_root_mean_squared_error', cv=3, verbose=2, n_jobs=-1)

# Performing the Grid Search for hyperparameter tuning with specific values
grid_search_specific.fit(X_train_rf_scaled, y_train_rf)

# RESULT
# GridSearchCV(cv=3, estimator=RandomForestRegressor(random_state=42), n_jobs=-1,
#              param_grid={'max_depth': [None, 20], 'min_samples_leaf': [1, 2],
#                          'min_samples_split': [2, 3],
#                          'n_estimators': [100, 150]},
#              scoring='neg_root_mean_squared_error', verbose=2)

# Getting the best hyperparameters from the specific Grid Search
best_params_specific = grid_search_specific.best_params_

# Training and evaluating the Random Forest model with the best hyperparameters from specific Grid Search
model_rf_specific = RandomForestRegressor(**best_params_specific, random_state=42)
model_rf_specific.fit(X_train_rf_scaled, y_train_rf)
y_pred_rf_specific = model_rf_specific.predict(X_test_rf_scaled)
rmse_rf_specific = mean_squared_error(y_test_rf, y_pred_rf_specific, squared=False)
r2_rf_specific = r2_score(y_test_rf, y_pred_rf_specific)

# Displaying the results for the Random Forest model with specific Grid Search
rmse_rf_specific, r2_rf_specific, best_params_specific

# RESULT
# (0.4157211236857813,
#  0.5876767955060284,
#  {'max_depth': 20,
#   'min_samples_leaf': 1,
#   'min_samples_split': 2,
#   'n_estimators': 150})
