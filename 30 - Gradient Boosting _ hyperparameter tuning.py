from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

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

# Best hyperparameters obtained for Gradient Boosting for all holes
gb_best_hyperparameters
