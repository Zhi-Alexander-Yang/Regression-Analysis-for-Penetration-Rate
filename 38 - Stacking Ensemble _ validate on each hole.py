# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset again
data_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
data_all_holes = pd.read_csv(data_path)

# Selecting features and target variable
X_all_holes = data_all_holes.drop(columns=['Penetration rate', 'Hole id', 'Time'])
y_all_holes = data_all_holes['Penetration rate']

# Splitting the data into training and validation sets
X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(X_all_holes, y_all_holes, test_size=0.2, random_state=42)

# Creating interaction features using PolynomialFeatures
poly_interaction_all = PolynomialFeatures(degree=2, interaction_only=True)
X_train_interaction_all = poly_interaction_all.fit_transform(X_train_all)
X_val_interaction_all = poly_interaction_all.transform(X_val_all)

# Reducing the complexity of the base models and enabling parallel computation
base_models_reduced = [
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
]

# Creating a Stacking Regressor with Linear Regression as the meta-model and reduced complexity base models
stacking_regressor_reduced = StackingRegressor(estimators=base_models_reduced, final_estimator=LinearRegression(), n_jobs=-1)

# Fitting the Stacking Regressor to the training data with interaction features
stacking_regressor_reduced.fit(X_train_interaction_all, y_train_all)

# RESULT
# StackingRegressor(estimators=[('rf',
#                                RandomForestRegressor(max_depth=10,
#                                                      n_estimators=50, n_jobs=-1,
#                                                      random_state=42)),
#                               ('gb',
#                                GradientBoostingRegressor(max_depth=5,
#                                                          n_estimators=50,
#                                                          random_state=42))],
#                   final_estimator=LinearRegression(), n_jobs=-1)

# List to store performance metrics for each hole
performance_metrics = []

# Looping over each hole
for hole_id in data_all_holes['Hole id'].unique():

    # Selecting data for the current hole
    data_hole = data_all_holes[data_all_holes['Hole id'] == hole_id]

    # Selecting features and target variable
    X_hole = data_hole.drop(columns=['Penetration rate', 'Hole id', 'Time'])
    y_hole = data_hole['Penetration rate']

    # Creating interaction features using PolynomialFeatures
    X_hole_interaction = poly_interaction_all.transform(X_hole)

    # Predicting penetration rate using the Stacking Regressor
    y_pred_hole = stacking_regressor_reduced.predict(X_hole_interaction)

    # Calculating RMSE and R^2 score for the predictions
    rmse_hole = mean_squared_error(y_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_hole, y_pred_hole)

    # Storing the performance metrics for the current hole
    performance_metrics.append([hole_id, rmse_hole, r2_hole])

# Creating a DataFrame with the performance metrics for each hole
performance_df = pd.DataFrame(performance_metrics, columns=['Hole id', 'RMSE', 'R^2'])
performance_df

# RESULT
#        Hole id      RMSE       R^2
# 0   29be0312e2  0.098063  0.981384
# 1   53034ece37  0.041237  0.987982
# 2   db95d6684b  0.036610  0.993490
# 3   07b4bd5a9d  0.215809  0.971488
# 4   5b88eb1e23  0.124127  0.978431
# 5   432e88547b  0.028571  0.994764
# 6   c69005e9d6  0.048588  0.989546
# 7   35f0f14168  0.143864  0.970974
# 8   6a7cb0d5af  0.066178  0.980547
# 9   b8960bac07  0.051444  0.991804
# 10  519ca0a683  0.067854  0.987224
# 11  d6abc83da1  0.084323  0.972738
# 12  c4e502a4b2  0.122388  0.942599
# 13  be8b244c5c  0.050556  0.989411
# 14  14c2f806ff  0.046636  0.990243
# 15  c92afa1018  0.076119  0.993046
