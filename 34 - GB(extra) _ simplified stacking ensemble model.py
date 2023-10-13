# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Loading the newly provided database
file_path_new = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_gb = pd.read_csv(file_path_new)

# Updating the feature and target columns based on the new dataset
features_gb = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
target_gb = 'Penetration rate'

# Function to perform simplified stacking for a specific hole
def simplified_stacking_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_gb[df_gb['Hole id'] == hole_id]
    X_hole = hole_data[features_gb]
    y_hole = hole_data[target_gb]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Defining the base models (simplified selection)
    base_models_simplified = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=50, random_state=42))
    ]

    # Initializing Stacking Regressor with simplified base models
    stacking_regressor_simplified = StackingRegressor(estimators=base_models_simplified, final_estimator=LinearRegression())

    # Training the stacking model
    stacking_regressor_simplified.fit(X_train_hole, y_train_hole)

    # Predicting on the test set
    y_pred_hole = stacking_regressor_simplified.predict(X_test_hole)

    # Evaluating the model
    rmse_hole = np.sqrt(mean_squared_error(y_test_hole, y_pred_hole))
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Selecting a subset of holes for analysis
subset_of_holes = df_gb['Hole id'].unique()[:5]

# Applying simplified stacking for the subset of holes and getting the evaluation results
stacking_results_subset = {}
for hole_id in subset_of_holes:
    stacking_rmse_simplified, stacking_r2_simplified = simplified_stacking_for_hole(hole_id)
    stacking_results_subset[hole_id] = {'RMSE': stacking_rmse_simplified, 'R-Squared': stacking_r2_simplified}

# Stacking evaluation results for the subset of holes
stacking_results_subset_df = pd.DataFrame.from_dict(stacking_results_subset, orient='index')
stacking_results_subset_df.reset_index(inplace=True)
stacking_results_subset_df.rename(columns={'index': 'Hole ID'}, inplace=True)
stacking_results_subset_df

# RESULT
#       Hole ID      RMSE  R-Squared
# 0  29be0312e2  0.329285   0.554940
# 1  53034ece37  0.334135   0.356729
# 2  db95d6684b  0.250819   0.634970
# 3  07b4bd5a9d  0.421011   0.887110
# 4  5b88eb1e23  0.282957   0.898011
