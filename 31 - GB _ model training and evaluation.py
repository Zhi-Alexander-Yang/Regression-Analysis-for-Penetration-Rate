from sklearn.metrics import mean_squared_error, r2_score

# Loading the newly provided database
file_path_new = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_gb = pd.read_csv(file_path_new)

# Updating the feature and target columns based on the new dataset
features_gb = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
target_gb = 'Penetration rate'

# Function to perform hyperparameter tuning for Gradient Boosting for a specific hole
def gradient_boosting_tuning_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_gb[df_gb['Hole id'] == hole_id]
    X_hole = hole_data[features_gb]
    y_hole = hole_data[target_gb]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Hyperparameter grid for Gradient Boosting
    gb_hyperparameter_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3]
    }

    # Randomized Search for hyperparameter tuning
    gb_random_search = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                                          param_distributions=gb_hyperparameter_grid,
                                          n_iter=10, cv=3, random_state=42, n_jobs=-1)
    gb_random_search.fit(X_train_hole, y_train_hole)

    # Best hyperparameters obtained for the specific hole
    best_hyperparameters_hole = gb_random_search.best_params_

    return best_hyperparameters_hole

# Performing hyperparameter tuning for Gradient Boosting for all holes and summarizing the best hyperparameters
gb_best_hyperparameters = {}
for hole_id in df_gb['Hole id'].unique():
    gb_best_hyperparameters[hole_id] = gradient_boosting_tuning_for_hole(hole_id)

# Function to train and evaluate Gradient Boosting model for a specific hole
def gradient_boosting_model_for_hole(hole_id, hyperparameters):
    # Filtering the data for the specific hole
    hole_data = df_gb[df_gb['Hole id'] == hole_id]
    X_hole = hole_data[features_gb]
    y_hole = hole_data[target_gb]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Initializing Gradient Boosting model with optimal hyperparameters
    gb_model = GradientBoostingRegressor(n_estimators=hyperparameters['n_estimators'],
                                         learning_rate=hyperparameters['learning_rate'],
                                         max_depth=hyperparameters['max_depth'],
                                         min_samples_split=hyperparameters['min_samples_split'],
                                         min_samples_leaf=hyperparameters['min_samples_leaf'],
                                         random_state=42)

    # Training the model
    gb_model.fit(X_train_hole, y_train_hole)

    # Predicting on the test set
    y_pred_hole = gb_model.predict(X_test_hole)

    # Evaluating the model
    rmse_hole = np.sqrt(mean_squared_error(y_test_hole, y_pred_hole))
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Training and evaluating Gradient Boosting model for all holes
gb_evaluation_results = {}
for hole_id, hyperparameters in gb_best_hyperparameters.items():
    gb_rmse, gb_r2 = gradient_boosting_model_for_hole(hole_id, hyperparameters)
    gb_evaluation_results[hole_id] = {'RMSE': gb_rmse, 'R-Squared': gb_r2}

# Gradient Boosting evaluation results for all holes
gb_evaluation_results_df = pd.DataFrame.from_dict(gb_evaluation_results, orient='index')
gb_evaluation_results_df.reset_index(inplace=True)
gb_evaluation_results_df.rename(columns={'index': 'Hole ID'}, inplace=True)
gb_evaluation_results_df

# RESULT
#        Hole ID      RMSE  R-Squared
# 0   29be0312e2  0.376743   0.417407
# 1   53034ece37  0.298545   0.486468
# 2   db95d6684b  0.235458   0.678313
# 3   07b4bd5a9d  0.975915   0.393410
# 4   5b88eb1e23  0.256023   0.916503
# 5   432e88547b  0.256449   0.447179
# 6   c69005e9d6  0.309182   0.395217
# 7   35f0f14168  0.672784   0.286935
# 8   6a7cb0d5af  0.431365   0.143930
# 9   b8960bac07  0.416079   0.503787
# 10  519ca0a683  0.405235   0.274325
# 11  d6abc83da1  0.591039  -0.090764
# 12  c4e502a4b2  0.297265   0.618310
# 13  be8b244c5c  0.561696  -0.709424
# 14  14c2f806ff  0.336192   0.484110
# 15  c92afa1018  0.659515   0.436064
