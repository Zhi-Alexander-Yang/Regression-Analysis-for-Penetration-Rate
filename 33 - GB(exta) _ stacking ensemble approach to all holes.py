from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Loading the newly provided database
file_path_new = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_gb = pd.read_csv(file_path_new)

# Updating the feature and target columns based on the new dataset
features_gb = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
target_gb = 'Penetration rate'

# Function to perform stacking for a specific hole
def stacking_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_gb[df_gb['Hole id'] == hole_id]
    X_hole = hole_data[features_gb]
    y_hole = hole_data[target_gb]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Defining the base models
    base_models = [
        ('Linear Regression', LinearRegression()),
        ('SVR', SVR(kernel='linear')),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]

    # Initializing Stacking Regressor
    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

    # Training the stacking model
    stacking_regressor.fit(X_train_hole, y_train_hole)

    # Predicting on the test set
    y_pred_hole = stacking_regressor.predict(X_test_hole)

    # Evaluating the model
    rmse_hole = np.sqrt(mean_squared_error(y_test_hole, y_pred_hole))
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Function to apply stacking for all holes and evaluate the performance
def stacking_for_all_holes():
    stacking_evaluation_results = {}
    for hole_id in df_gb['Hole id'].unique():
        stacking_rmse, stacking_r2 = stacking_for_hole(hole_id)
        stacking_evaluation_results[hole_id] = {'RMSE': stacking_rmse, 'R-Squared': stacking_r2}

    return stacking_evaluation_results

# Applying stacking for all holes and getting the evaluation results
stacking_evaluation_results = stacking_for_all_holes()

# Stacking evaluation results for all holes
stacking_evaluation_results_df = pd.DataFrame.from_dict(stacking_evaluation_results, orient='index')
stacking_evaluation_results_df.reset_index(inplace=True)
stacking_evaluation_results_df.rename(columns={'index': 'Hole ID'}, inplace=True)
stacking_evaluation_results_df
