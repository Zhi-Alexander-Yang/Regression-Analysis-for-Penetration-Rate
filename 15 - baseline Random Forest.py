from sklearn.ensemble import RandomForestRegressor

# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_polynomial_features = pd.read_csv(file_path)
df_polynomial_features = df_polynomial_features.dropna()
df_polynomial_features["Hole id"] = df_polynomial_features["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Selecting original features and target variable
X_rf = df_polynomial_features[features_unified]
y_rf = df_polynomial_features[target_unified]

# Splitting the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Standardizing the features
scaler_rf = StandardScaler()
X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
X_test_rf_scaled = scaler_rf.transform(X_test_rf)

# Training and evaluating the Random Forest model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train_rf_scaled, y_train_rf)
y_pred_rf = model_rf.predict(X_test_rf_scaled)
rmse_rf = mean_squared_error(y_test_rf, y_pred_rf, squared=False)
r2_rf = r2_score(y_test_rf, y_pred_rf)

# Displaying the results for Random Forest
rmse_rf, r2_rf

# RESULT
# (0.41439279689604946, 0.5903075251576828)
