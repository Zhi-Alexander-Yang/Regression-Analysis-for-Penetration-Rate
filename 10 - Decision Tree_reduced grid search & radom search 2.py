# Load the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_decision_tree = pd.read_csv(file_path)

# Drop any rows with missing values
df_decision_tree = df_decision_tree.dropna()

# Results DataFrame for reduced Grid Search and Random Search
grid_search_results = pd.DataFrame(columns=["Hole id", "Best Parameters", "RMSE", "R-squared"])
random_search_results = pd.DataFrame(columns=["Hole id", "Best Parameters", "RMSE", "R-squared"])

# Loop through each unique hole id (excluding the first hole) and apply hyperparameter tuning using both methods
for hole_id in df_decision_tree["Hole id"].unique()[1:]:
    hole_data = df_decision_tree[df_decision_tree["Hole id"] == hole_id]

    # Apply reduced Grid Search
    best_params_grid, rmse_grid, r2_grid = tune_decision_tree(hole_data, GridSearchCV, further_reduced_param_grid)
    grid_search_results = grid_search_results.append({"Hole id": hole_id, "Best Parameters": best_params_grid, "RMSE": rmse_grid, "R-squared": r2_grid}, ignore_index=True)

    # Apply Random Search
    best_params_random, rmse_random, r2_random = tune_decision_tree(hole_data, RandomizedSearchCV, random_param_grid, n_iter=10)
    random_search_results = random_search_results.append({"Hole id": hole_id, "Best Parameters": best_params_random, "RMSE": rmse_random, "R-squared": r2_random}, ignore_index=True)

# Display the results for both methods
grid_search_results, random_search_results

# RESULT
# (       Hole id                                    Best Parameters      RMSE  \
#  0   53034ece37  {'criterion': 'squared_error', 'max_depth': 10...  0.445115
#  1   db95d6684b  {'criterion': 'squared_error', 'max_depth': 10...  0.284934
#  2   07b4bd5a9d  {'criterion': 'squared_error', 'max_depth': 10...  1.122870   
#  3   5b88eb1e23  {'criterion': 'squared_error', 'max_depth': 10...  0.269460
#  4   432e88547b  {'criterion': 'squared_error', 'max_depth': 10...  0.430839
#  5   c69005e9d6  {'criterion': 'squared_error', 'max_depth': 10...  0.212745
#  6   35f0f14168  {'criterion': 'squared_error', 'max_depth': 10...  0.661604
#  7   6a7cb0d5af  {'criterion': 'squared_error', 'max_depth': No...  0.517738
#  8   b8960bac07  {'criterion': 'squared_error', 'max_depth': 10...  0.455623
#  9   519ca0a683  {'criterion': 'squared_error', 'max_depth': 10...  0.526739
#  10  d6abc83da1  {'criterion': 'squared_error', 'max_depth': 10...  0.585336
#  11  c4e502a4b2  {'criterion': 'squared_error', 'max_depth': 10...  0.333730
#  12  be8b244c5c  {'criterion': 'squared_error', 'max_depth': 10...  0.606112
#  13  14c2f806ff  {'criterion': 'squared_error', 'max_depth': 10...  0.369736
#  14  c92afa1018  {'criterion': 'squared_error', 'max_depth': No...  0.787309
#
#      R-squared
#  0   -0.141545
#  1    0.528919
#  2    0.196973
#  3    0.907509
#  4   -0.560321
#  5    0.713654
#  6    0.310437
#  7   -0.233220
#  8    0.404984
#  9   -0.226082
#  10  -0.069816
#  11   0.518924
#  12  -0.990457
#  13   0.376024
#  14   0.196344  ,
#         Hole id                                    Best Parameters      RMSE  \
#  0   53034ece37  {'splitter': 'best', 'min_samples_split': 2, '...  0.448155
#  1   db95d6684b  {'splitter': 'best', 'min_samples_split': 10, ...  0.279259
#  2   07b4bd5a9d  {'splitter': 'best', 'min_samples_split': 10, ...  0.854878
#  3   5b88eb1e23  {'splitter': 'best', 'min_samples_split': 2, '...  0.265360
#  4   432e88547b  {'splitter': 'random', 'min_samples_split': 5,...  0.359271
#  5   c69005e9d6  {'splitter': 'random', 'min_samples_split': 10...  0.214478
#  6   35f0f14168  {'splitter': 'random', 'min_samples_split': 2,...  0.728446
#  7   6a7cb0d5af  {'splitter': 'random', 'min_samples_split': 10...  0.451923
#  8   b8960bac07  {'splitter': 'best', 'min_samples_split': 10, ...  0.434158
#  9   519ca0a683  {'splitter': 'random', 'min_samples_split': 5,...  0.378074
#  10  d6abc83da1  {'splitter': 'random', 'min_samples_split': 2,...  0.585336
#  11  c4e502a4b2  {'splitter': 'best', 'min_samples_split': 10, ...  0.319313
#  12  be8b244c5c  {'splitter': 'best', 'min_samples_split': 5, '...  0.606623
#  13  14c2f806ff  {'splitter': 'best', 'min_samples_split': 10, ...  0.502583
#  14  c92afa1018  {'splitter': 'random', 'min_samples_split': 2,...  0.593081
#
#      R-squared
#  0   -0.157191
#  1    0.547496
#  2    0.534543
#  3    0.910302
#  4   -0.084999
#  5    0.708971
#  6    0.164066
#  7    0.060385
#  8    0.459728
#  9    0.368342
#  10  -0.069816
#  11   0.559592
#  12  -0.993810
#  13  -0.152918
#  14   0.543954  )
