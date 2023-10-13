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

# Identifying challenging holes based on R-Squared values (e.g., R-Squared < 0.1)
challenging_holes_ids = logarithmic_results_df[logarithmic_results_df['R-Squared'] < 0.1]['Hole ID'].values

# Analyzing the features and target for the challenging holes
challenging_holes_analysis = []
for hole_id in challenging_holes_ids:
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    hole_summary = hole_data.describe().loc[['mean', 'std'], features_unified + [target_unified]]
    hole_summary['Hole ID'] = hole_id
    challenging_holes_analysis.append(hole_summary)

# Combining the analysis results for challenging holes
challenging_holes_analysis_df = pd.concat(challenging_holes_analysis, axis=0).reset_index()
challenging_holes_analysis_df.rename(columns={'index': 'Statistic'}, inplace=True)

# Function to apply Exponential transformation and evaluate the model for the specific challenging hole (Hole ID 10)
def exponential_transformation_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified].apply(np.exp) # Applying exponential transformation
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Standardizing the transformed features
    scaler_hole = StandardScaler()
    X_train_hole_exp_scaled = scaler_hole.fit_transform(X_train_hole)
    X_test_hole_exp_scaled = scaler_hole.transform(X_test_hole)

    # Training and evaluating the Random Forest model with customized hyperparameters
    model_rf_hole = RandomForestRegressor(**customized_hyperparameters[hole_id], random_state=42)
    model_rf_hole.fit(X_train_hole_exp_scaled, y_train_hole)
    y_pred_hole = model_rf_hole.predict(X_test_hole_exp_scaled)
    rmse_hole = mean_squared_error(y_test_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Applying Exponential transformation for Hole ID 10 and summarizing the result
exponential_result = exponential_transformation_for_hole(hole_id=10)
exponential_result_dict = {'Hole ID': 10, 'RMSE': exponential_result[0], 'R-Squared': exponential_result[1]}
exponential_result_dict

# RESULT
# {'Hole ID': 10, 'RMSE': 0.4664416710193534, 'R-Squared': -0.17880351101574266
