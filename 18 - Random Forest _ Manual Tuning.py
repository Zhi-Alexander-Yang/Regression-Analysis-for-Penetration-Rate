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

# Defining manually tuned hyperparameters for Random Forest
manual_params = {
    'n_estimators': 150,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Training and evaluating the Random Forest model with manually tuned hyperparameters
model_rf_manual = RandomForestRegressor(**manual_params, random_state=42)
model_rf_manual.fit(X_train_rf_scaled, y_train_rf)
y_pred_rf_manual = model_rf_manual.predict(X_test_rf_scaled)
rmse_rf_manual = mean_squared_error(y_test_rf, y_pred_rf_manual, squared=False)
r2_rf_manual = r2_score(y_test_rf, y_pred_rf_manual)

# Displaying the results for the Random Forest model with manual tuning
rmse_rf_manual, r2_rf_manual, manual_params

# RESULT
# (0.4157211236857813,
#  0.5876767955060284,
#  {'n_estimators': 150,
#   'max_depth': 20,
#   'min_samples_split': 2,
#   'min_samples_leaf': 1})
