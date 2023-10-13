from sklearn.preprocessing import PolynomialFeatures

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

# Function to apply Polynomial Features transformation and evaluate the model for a specific hole
def polynomial_features_for_hole(hole_id, degree):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified]
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Applying Polynomial Features transformation
    poly = PolynomialFeatures(degree=degree)
    X_train_hole_poly = poly.fit_transform(X_train_hole)
    X_test_hole_poly = poly.transform(X_test_hole)

    # Standardizing the transformed features
    scaler_hole = StandardScaler()
    X_train_hole_poly_scaled = scaler_hole.fit_transform(X_train_hole_poly)
    X_test_hole_poly_scaled = scaler_hole.transform(X_test_hole_poly)

    # Training and evaluating the Random Forest model with customized hyperparameters
    model_rf_hole = RandomForestRegressor(**customized_hyperparameters[hole_id], random_state=42)
    model_rf_hole.fit(X_train_hole_poly_scaled, y_train_hole)
    y_pred_hole = model_rf_hole.predict(X_test_hole_poly_scaled)
    rmse_hole = mean_squared_error(y_test_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Applying Polynomial Features transformation (degree=2) for all holes and summarizing the results
polynomial_results = []
for hole_id in df_rf['Hole id'].unique():
    rmse_hole, r2_hole = polynomial_features_for_hole(hole_id, degree=2)
    polynomial_results.append((hole_id, rmse_hole, r2_hole))

# Creating a DataFrame to represent the results for Polynomial Features transformation
polynomial_results_df = pd.DataFrame(polynomial_results, columns=['Hole ID', 'RMSE', 'R-Squared'])
polynomial_results_df.sort_values(by='R-Squared', ascending=False)

# RESULT
#     Hole ID      RMSE  R-Squared
# 3         0  0.341951   0.925527
# 4         7  0.301651   0.884090
# 2        15  0.237476   0.672773
# 15       13  0.510495   0.662119
# 9         9  0.355464   0.637833
# 6        12  0.253446   0.593611
# 12       11  0.326585   0.539302
# 5         4  0.244046   0.499359
# 10        5  0.343806   0.477658
# 14        1  0.340858   0.469689
# 0         2  0.380823   0.404720
# 8         8  0.369746   0.371035
# 1         6  0.335403   0.351838
# 11       14  0.475514   0.293967
# 7         3  0.672624   0.287275
# 13       10  0.456084  -0.127032
