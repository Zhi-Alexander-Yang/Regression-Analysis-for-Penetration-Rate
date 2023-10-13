# Load the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_decision_tree = pd.read_csv(file_path)

# Drop any rows with missing values
df_decision_tree = df_decision_tree.dropna()

# Defining the further reduced hyperparameter grid for Grid Search
further_reduced_param_grid = {
    'criterion': ['squared_error'],  # Using squared_error based on the warning
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Defining the hyperparameter grid for Random Search
random_param_grid = {
    'criterion': ['squared_error'],  # Using squared_error based on the warning
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Apply reduced Grid Search to the first hole (without specifying n_iter)
best_params_grid, rmse_grid, r2_grid = tune_decision_tree(first_hole_data, GridSearchCV, further_reduced_param_grid)

# Apply Random Search to the first hole (specifying n_iter=10)
best_params_random, rmse_random, r2_random = tune_decision_tree(first_hole_data, RandomizedSearchCV, random_param_grid, n_iter=10)

# Results for the first hole
first_hole_results = {
    "Grid Search": {"Best Parameters": best_params_grid, "RMSE": rmse_grid, "R-squared": r2_grid},
    "Random Search": {"Best Parameters": best_params_random, "RMSE": rmse_random, "R-squared": r2_random}
}

first_hole_results

# RESULT
# {'Grid Search': {'Best Parameters': {'criterion': 'squared_error',
#    'max_depth': 20,
#    'min_samples_leaf': 2,
#    'min_samples_split': 2,
#    'splitter': 'random'},
#   'RMSE': 0.6304950263921739,
#   'R-squared': -0.6316912902363636},
#  'Random Search': {'Best Parameters': {'splitter': 'random',
#    'min_samples_split': 10,
#    'min_samples_leaf': 4,
#    'max_depth': 30,
#    'criterion': 'squared_error'},
#   'RMSE': 0.5008843428435341,
#   'R-squared': -0.029792152171237873}}
