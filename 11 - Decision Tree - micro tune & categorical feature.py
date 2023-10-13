# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_unified_model = pd.read_csv(file_path)
df_unified_model = df_unified_model.dropna()

# Convert the "Hole id" column to a categorical variable and encode it
df_unified_model["Hole id"] = df_unified_model["Hole id"].astype('category').cat.codes


# Defining the feature and target variable names
features = ["Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target = "Penetration rate"

# Results DataFrame for individual models using reduced Grid Search and Random Search
individual_grid_search_results = pd.DataFrame(columns=["Hole id", "Best Parameters", "RMSE", "R-squared"])
individual_random_search_results = pd.DataFrame(columns=["Hole id", "Best Parameters", "RMSE", "R-squared"])

# Loop through each unique hole id and apply hyperparameter tuning using both methods
for hole_id in df_decision_tree["Hole id"].unique():
    hole_data = df_decision_tree[df_decision_tree["Hole id"] == hole_id]

    # Apply reduced Grid Search
    best_params_grid, rmse_grid, r2_grid = tune_decision_tree(hole_data, GridSearchCV, further_reduced_param_grid)
    individual_grid_search_results = individual_grid_search_results.append({"Hole id": hole_id, "Best Parameters": best_params_grid, "RMSE": rmse_grid, "R-squared": r2_grid}, ignore_index=True)

    # Apply Random Search
    best_params_random, rmse_random, r2_random = tune_decision_tree(hole_data, RandomizedSearchCV, random_param_grid, n_iter=10)
    individual_random_search_results = individual_random_search_results.append({"Hole id": hole_id, "Best Parameters": best_params_random, "RMSE": rmse_random, "R-squared": r2_random}, ignore_index=True)

# Display the results for both methods
individual_grid_search_results, individual_random_search_results

# RESULT
# (       Hole id                                    Best Parameters      RMSE  \
#  0   29be0312e2  {'criterion': 'squared_error', 'max_depth': 20...  0.630495
#  1   53034ece37  {'criterion': 'squared_error', 'max_depth': 10...  0.445115
#  2   db95d6684b  {'criterion': 'squared_error', 'max_depth': 10...  0.284934
#  3   07b4bd5a9d  {'criterion': 'squared_error', 'max_depth': 10...  1.122870
#  4   5b88eb1e23  {'criterion': 'squared_error', 'max_depth': 10...  0.269460
#  5   432e88547b  {'criterion': 'squared_error', 'max_depth': 10...  0.430839
#  6   c69005e9d6  {'criterion': 'squared_error', 'max_depth': 10...  0.212745
#  7   35f0f14168  {'criterion': 'squared_error', 'max_depth': 10...  0.661604
#  8   6a7cb0d5af  {'criterion': 'squared_error', 'max_depth': No...  0.517738
#  9   b8960bac07  {'criterion': 'squared_error', 'max_depth': 10...  0.455623
#  10  519ca0a683  {'criterion': 'squared_error', 'max_depth': 10...  0.526739
#  11  d6abc83da1  {'criterion': 'squared_error', 'max_depth': 10...  0.585336
#  12  c4e502a4b2  {'criterion': 'squared_error', 'max_depth': 10...  0.333730
#  13  be8b244c5c  {'criterion': 'squared_error', 'max_depth': 10...  0.606112
#  14  14c2f806ff  {'criterion': 'squared_error', 'max_depth': 10...  0.369736
#  15  c92afa1018  {'criterion': 'squared_error', 'max_depth': No...  0.787309
#
#      R-squared
#  0   -0.631691
#  1   -0.141545
#  2    0.528919
#  3    0.196973
#  4    0.907509
#  5   -0.560321
#  6    0.713654
#  7    0.310437
#  8   -0.233220
#  9    0.404984
#  10  -0.226082
#  11  -0.069816
#  12   0.518924
#  13  -0.990457
#  14   0.376024
#  15   0.196344  ,
#         Hole id                                    Best Parameters      RMSE  \
#  0   29be0312e2  {'splitter': 'random', 'min_samples_split': 5,...  0.499631
#  1   53034ece37  {'splitter': 'random', 'min_samples_split': 5,...  0.374843
#  2   db95d6684b  {'splitter': 'best', 'min_samples_split': 10, ...  0.274581
#  3   07b4bd5a9d  {'splitter': 'best', 'min_samples_split': 5, '...  0.851373
#  4   5b88eb1e23  {'splitter': 'best', 'min_samples_split': 10, ...  0.258195
#  5   432e88547b  {'splitter': 'random', 'min_samples_split': 5,...  0.430839
#  6   c69005e9d6  {'splitter': 'random', 'min_samples_split': 10...  0.214478
#  7   35f0f14168  {'splitter': 'best', 'min_samples_split': 10, ...  0.767393
#  8   6a7cb0d5af  {'splitter': 'random', 'min_samples_split': 2,...  0.475798
#  9   b8960bac07  {'splitter': 'best', 'min_samples_split': 5, '...  0.432186
#  10  519ca0a683  {'splitter': 'random', 'min_samples_split': 10...  0.364611
#  11  d6abc83da1  {'splitter': 'random', 'min_samples_split': 2,...  0.525458
#  12  c4e502a4b2  {'splitter': 'best', 'min_samples_split': 10, ...  0.402023
#  13  be8b244c5c  {'splitter': 'random', 'min_samples_split': 2,...  0.522953
#  14  14c2f806ff  {'splitter': 'best', 'min_samples_split': 5, '...  0.578147
#  15  c92afa1018  {'splitter': 'random', 'min_samples_split': 10...  0.578722
#
#      R-squared
#  0   -0.024646
#  1    0.190444
#  2    0.562530
#  3    0.538352
#  4    0.915081
#  5   -0.560321
#  6    0.708971
#  7    0.072287
#  8   -0.041517
#  9    0.464625
#  10   0.412526
#  11   0.137865
#  12   0.301888
#  13  -0.481740
#  14  -0.525670
#  15   0.565770  )

# Re-importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Reload the dataset
file_path = '/mnt/data/1-s2.0-S1365160920308121-mmc1.csv'
df_unified_model = pd.read_csv(file_path)
df_unified_model = df_unified_model.dropna()

# Convert the "Hole id" column to a categorical variable and encode it
df_unified_model["Hole id"] = df_unified_model["Hole id"].astype('category').cat.codes

# Display the first few rows to verify the encoding
df_unified_model.head()

# RESULT
#    Hole id  Depth                 Time  Penetration rate  Percussion pressure  \
# 0        2  0.025  2017-02-10T08:42:16             1.821              124.280
# 1        2  0.050  2017-02-10T08:42:16             3.126              136.658
# 2        2  0.077  2017-02-10T08:42:17             2.958              140.218
# 3        2  0.106  2017-02-10T08:42:17             2.720              142.401
# 4        2  0.131  2017-02-10T08:42:18             2.602              139.546
#
#    Feed pressure  Flush air pressure  Rotation pressure  Dampening pressure  \
# 0         23.702               4.621             38.586              38.914
# 1         25.768               5.064             38.364              39.837
# 2         26.415               5.355             37.695              40.437
# 3         26.852               5.630             37.928              40.752
# 4         26.939               5.845             37.751              41.061
#
#    Hardness  Salve  HoleNo
# 0     4.216   1938      13
# 1     8.418   1938      13
# 2     7.732   1938      13
# 3     6.842   1938      13

# Selecting features and target variable
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Splitting the data into training and testing sets
X_unified = df_unified_model[features_unified]
y_unified = df_unified_model[target_unified]
X_train_unified, X_test_unified, y_train_unified, y_test_unified = train_test_split(X_unified, y_unified, test_size=0.2, random_state=42)

# Standardizing the features
scaler_unified = StandardScaler()
X_train_unified_scaled = scaler_unified.fit_transform(X_train_unified)
X_test_unified_scaled = scaler_unified.transform(X_test_unified)

# Displaying the first few rows of scaled training features
pd.DataFrame(X_train_unified_scaled, columns=features_unified).head()

# RESULT
#     Hole id  Percussion pressure  Feed pressure  Flush air pressure  \
# 0  0.352801             0.504412       0.433446            1.334596
# 1  0.352801            -2.219131      -1.581906           -0.148386
# 2 -1.371462             0.464463       0.288602           -0.212424
# 3  1.645998             0.410817       0.526671           -0.844376
# 4 -0.078265             0.476344       0.434260            0.251008
#
#    Rotation pressure  Dampening pressure
# 0          -0.101961            0.036029
# 1          -0.510637           -1.643786
# 2           1.258862            0.531890
# 3          -0.326621            0.676037  
# 4          -0.076236            0.061332

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Defining hyperparameter grids for unified model
unified_grid_param_grid = {
    'criterion': ['squared_error'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
unified_random_param_grid = {
    'criterion': ['squared_error'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Hyperparameter tuning function for unified model
def tune_unified_model(search_method, param_grid, n_iter=None):
    model = DecisionTreeRegressor(random_state=42)
    params = {'estimator': model, 'param_distributions' if search_method == RandomizedSearchCV else 'param_grid': param_grid,
              'cv': 5, 'scoring': 'neg_mean_squared_error'}
    if n_iter:
        params['n_iter'] = n_iter
    search = search_method(**params)
    search.fit(X_train_unified_scaled, y_train_unified)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_unified_scaled)
    rmse = mean_squared_error(y_test_unified, y_pred, squared=False)
    r2 = r2_score(y_test_unified, y_pred)
    return search.best_params_, rmse, r2

# Apply Grid Search and Random Search to unified model
best_params_unified_grid, rmse_unified_grid, r2_unified_grid = tune_unified_model(GridSearchCV, unified_grid_param_grid)
best_params_unified_random, rmse_unified_random, r2_unified_random = tune_unified_model(RandomizedSearchCV, unified_random_param_grid, n_iter=10)

# Displaying the results for both methods
best_params_unified_grid, rmse_unified_grid, r2_unified_grid, best_params_unified_random, rmse_unified_random, r2_unified_random

# RESULT
# ({'criterion': 'squared_error',
#   'max_depth': None,
#   'min_samples_leaf': 2,
#   'min_samples_split': 10,
#   'splitter': 'random'},
#  0.5865398075541914,
#  0.17921657783589406,
#  {'splitter': 'random',
#   'min_samples_split': 10,
#   'min_samples_leaf': 2,
#   'max_depth': 20,
#   'criterion': 'squared_error'},
#  0.5924968473548781,
#  0.16245976571854537)
