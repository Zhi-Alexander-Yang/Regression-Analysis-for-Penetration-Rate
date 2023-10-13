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

# Defining a reduced hyperparameter grid for Random Search
param_grid_reduced = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setting up the Random Search with reduced search space
random_search_reduced = RandomizedSearchCV(estimator=model_rf, param_distributions=param_grid_reduced, n_iter=10, scoring='neg_root_mean_squared_error', cv=3, verbose=2, random_state=42, n_jobs=-1)

# Performing the Random Search for hyperparameter tuning with reduced search space
random_search_reduced.fit(X_train_rf_scaled, y_train_rf)

# RESULT
# RandomizedSearchCV(cv=3, estimator=RandomForestRegressor(random_state=42),
#                    n_jobs=-1,
#                    param_distributions={'max_depth': [None, 20, 30],
#                                         'min_samples_leaf': [1, 2],
#                                         'min_samples_split': [2, 5],
#                                         'n_estimators': [100, 200]},
#                    random_state=42, scoring='neg_root_mean_squared_error',
#                    verbose=2)

# Getting the best hyperparameters from the reduced Random Search
best_params_reduced = random_search_reduced.best_params_

# Training and evaluating the Random Forest model with the best hyperparameters from reduced Random Search
model_rf_reduced = RandomForestRegressor(**best_params_reduced, random_state=42)
model_rf_reduced.fit(X_train_rf_scaled, y_train_rf)
y_pred_rf_reduced = model_rf_reduced.predict(X_test_rf_scaled)
rmse_rf_reduced = mean_squared_error(y_test_rf, y_pred_rf_reduced, squared=False)
r2_rf_reduced = r2_score(y_test_rf, y_pred_rf_reduced)

# Displaying the results for the Random Forest model with reduced Random Search
rmse_rf_reduced, r2_rf_reduced, best_params_reduced

# RESULT
# (0.41735625736190535,
#  0.5844268790450401,
#  {'n_estimators': 200,
#   'min_samples_split': 2,
#   'min_samples_leaf': 1,
#   'max_depth': 20})
