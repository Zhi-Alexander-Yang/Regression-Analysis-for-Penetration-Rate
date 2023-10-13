# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_polynomial_features = pd.read_csv(file_path)
df_polynomial_features = df_polynomial_features.dropna()
df_polynomial_features["Hole id"] = df_polynomial_features["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Selecting features and target variable
X_poly = df_polynomial_features[features_unified]
y_poly = df_polynomial_features[target_unified]

# Splitting the data into training and testing sets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)

# Creating polynomial features (degree=2)
poly_transformer = PolynomialFeatures(degree=2)
X_train_poly_expanded = poly_transformer.fit_transform(X_train_poly)
X_test_poly_expanded = poly_transformer.transform(X_test_poly)

# Standardizing the polynomial features
scaler_poly = StandardScaler()
X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly_expanded)
X_test_poly_scaled = scaler_poly.transform(X_test_poly_expanded)

# Training and evaluating the model with polynomial features
model_poly = DecisionTreeRegressor(random_state=42)
model_poly.fit(X_train_poly_scaled, y_train_poly)
y_pred_poly = model_poly.predict(X_test_poly_scaled)
rmse_poly = mean_squared_error(y_test_poly, y_pred_poly, squared=False)
r2_poly = r2_score(y_test_poly, y_pred_poly)

# Displaying the results for polynomial features
rmse_poly, r2_poly

# RESULT
# (0.5805752658536832, 0.1958248460093056)
