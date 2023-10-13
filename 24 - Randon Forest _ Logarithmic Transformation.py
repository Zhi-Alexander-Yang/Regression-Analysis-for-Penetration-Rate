import numpy as np

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

# Function to apply Logarithmic transformation and evaluate the model for a specific hole
def logarithmic_transformation_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified].apply(np.log1p) # Applying logarithmic transformation
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Standardizing the transformed features
    scaler_hole = StandardScaler()
    X_train_hole_log_scaled = scaler_hole.fit_transform(X_train_hole)
    X_test_hole_log_scaled = scaler_hole.transform(X_test_hole)

    # Training and evaluating the Random Forest model with customized hyperparameters
    model_rf_hole = RandomForestRegressor(**customized_hyperparameters[hole_id], random_state=42)
    model_rf_hole.fit(X_train_hole_log_scaled, y_train_hole)
    y_pred_hole = model_rf_hole.predict(X_test_hole_log_scaled)
    rmse_hole = mean_squared_error(y_test_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Applying Logarithmic transformation for all holes and summarizing the results
logarithmic_results = []
for hole_id in df_rf['Hole id'].unique():
    rmse_hole, r2_hole = logarithmic_transformation_for_hole(hole_id)
    logarithmic_results.append((hole_id, rmse_hole, r2_hole))

# Creating a DataFrame to represent the results for Logarithmic transformation
logarithmic_results_df = pd.DataFrame(logarithmic_results, columns=['Hole ID', 'RMSE', 'R-Squared'])
logarithmic_results_df.sort_values(by='R-Squared', ascending=False)


# RESULT
#     Hole ID      RMSE  R-Squared
# 4         7  0.259577   0.914169
# 3         0  0.578051   0.787185
# 6        12  0.224676   0.680636
# 9         9  0.353976   0.640859
# 2        15  0.250546   0.635764
# 12       11  0.296874   0.619315
# 15       13  0.593040   0.544017
# 0         2  0.337077   0.533627
# 5         4  0.248466   0.481058
# 10        5  0.343092   0.479823
# 14        1  0.355691   0.422529
# 1         6  0.323839   0.395761
# 8         8  0.375589   0.350998
# 7         3  0.662776   0.307992
# 11       14  0.482440   0.273250
# 13       10  0.490639  -0.304282
