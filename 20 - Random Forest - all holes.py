# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_rf = pd.read_csv(file_path)
df_rf = df_rf.dropna()
df_rf["Hole id"] = df_rf["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Function to evaluate the Random Forest model for a specific hole
def evaluate_model_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified]
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler_hole = StandardScaler()
    X_train_hole_scaled = scaler_hole.fit_transform(X_train_hole)
    X_test_hole_scaled = scaler_hole.transform(X_test_hole)

    # Training and evaluating the Random Forest model for the specific hole
    model_rf_hole = RandomForestRegressor(**manual_params, random_state=42)
    model_rf_hole.fit(X_train_hole_scaled, y_train_hole)
    y_pred_hole = model_rf_hole.predict(X_test_hole_scaled)
    rmse_hole = mean_squared_error(y_test_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Evaluating the Random Forest model for each hole and summarizing the results
validation_results = []
for hole_id in df_rf['Hole id'].unique():
    rmse_hole, r2_hole = evaluate_model_for_hole(hole_id)
    validation_results.append((hole_id, rmse_hole, r2_hole))

# Creating a DataFrame to represent the validation results for each hole
validation_df = pd.DataFrame(validation_results, columns=['Hole ID', 'RMSE', 'R-Squared'])
validation_df.sort_values(by='R-Squared', ascending=False)

# RESULT
#     Hole ID      RMSE  R-Squared
# 4         7  0.246411   0.922655
# 3         0  0.426640   0.884070
# 2        15  0.244900   0.651995
# 12       11  0.290186   0.636273
# 6        12  0.249413   0.606442
# 9         9  0.371902   0.603562
# 15       13  0.559943   0.593494
# 0         2  0.330469   0.551734
# 5         4  0.241224   0.510870
# 10        5  0.350644   0.456674
# 14        1  0.361368   0.403951
# 1         6  0.322204   0.401849
# 8         8  0.363703   0.391425
# 7         3  0.654129   0.325931
# 11       14  0.500361   0.218253
# 13       10  0.495669  -0.331162
