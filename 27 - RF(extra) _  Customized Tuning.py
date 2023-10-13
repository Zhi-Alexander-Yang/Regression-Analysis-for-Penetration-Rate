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

# Customized hyperparameters grid for Random Search tailored to Hole ID 10
custom_random_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3]
}

# Filtering the data for the specific challenging hole (Hole ID 10)
hole_data_10 = df_rf[df_rf['Hole id'] == 10]
X_hole_10 = hole_data_10[features_unified]
y_hole_10 = hole_data_10[target_unified]

# Splitting the data into training and testing sets
X_train_hole_10, _, y_train_hole_10, _ = train_test_split(X_hole_10, y_hole_10, test_size=0.2, random_state=42)

# Standardizing the features
scaler_hole_10 = StandardScaler()
X_train_hole_10_scaled = scaler_hole_10.fit_transform(X_train_hole_10)

# Applying Random Search for customized hyperparameter tuning for Hole ID 10
rf_random_search_custom = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                             param_distributions=custom_random_grid,
                                             n_iter=20, cv=3, random_state=42, n_jobs=-1)
rf_random_search_custom.fit(X_train_hole_10_scaled, y_train_hole_10)

# Best hyperparameters obtained for Hole ID 10
custom_hyperparameters_hole_10 = rf_random_search_custom.best_params_
custom_hyperparameters_hole_10

# RESULT
# {'n_estimators': 150,
#  'min_samples_split': 3,
#  'min_samples_leaf': 1,
#  'max_depth': 15}
