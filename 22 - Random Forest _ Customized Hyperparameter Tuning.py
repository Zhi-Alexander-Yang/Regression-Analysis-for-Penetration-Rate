from sklearn.model_selection import RandomizedSearchCV

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

# Hyperparameters grid for Random Search
random_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Function to apply Random Search for a specific hole
def random_search_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified]
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, _, y_train_hole, _ = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler_hole = StandardScaler()
    X_train_hole_scaled = scaler_hole.fit_transform(X_train_hole)

    # Applying Random Search for hyperparameter tuning
    rf_random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                          param_distributions=random_grid,
                                          n_iter=10, cv=3, random_state=42, n_jobs=-1)
    rf_random_search.fit(X_train_hole_scaled, y_train_hole)

    return rf_random_search.best_params_

# Applying customized hyperparameter tuning for all holes
customized_hyperparameters = {hole_id: random_search_for_hole(hole_id) for hole_id in df_rf['Hole id'].unique()}
customized_hyperparameters

# RESULT
# {2: {'n_estimators': 50,
#   'min_samples_split': 2,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  6: {'n_estimators': 150,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  15: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  0: {'n_estimators': 150,
#   'min_samples_split': 5,
#   'min_samples_leaf': 4,
#   'max_depth': 20},
#  7: {'n_estimators': 50,
#   'min_samples_split': 2,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  4: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  12: {'n_estimators': 100,
#   'min_samples_split': 2,
#   'min_samples_leaf': 4,
#   'max_depth': 30},
#  3: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  8: {'n_estimators': 50,
#   'min_samples_split': 2,
#   'min_samples_leaf': 4,
#   'max_depth': None},
#  9: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  5: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  14: {'n_estimators': 50,
#   'min_samples_split': 10,
#   'min_samples_leaf': 2,
#   'max_depth': 20},
#  11: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  10: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  1: {'n_estimators': 200,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10},
#  13: {'n_estimators': 150,
#   'min_samples_split': 5,
#   'min_samples_leaf': 2,
#   'max_depth': 10}}
