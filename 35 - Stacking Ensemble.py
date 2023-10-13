# Loading the dataset again
data_path = '/mnt/data/1-s2.0-S1365160920308121-mmc1.csv'
data_all_holes = pd.read_csv(data_path)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# Selecting a specific "Hole id" for analysis
unique_hole_ids = data_all_holes['Hole id'].unique()
selected_hole_id = unique_hole_ids[0]  # Choosing the first unique "Hole id"

# Filtering the data for the selected "Hole id"
hole_data_gb = data_all_holes[data_all_holes['Hole id'] == selected_hole_id]

# Splitting the data into features (X) and target (y)
X_gb = hole_data_gb.drop(columns=['Penetration rate', 'Hole id', 'Time'])
y_gb = hole_data_gb['Penetration rate']

# Splitting the data into training and validation sets
X_train_gb, X_val_gb, y_train_gb, y_val_gb = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)

# Creating interaction features using PolynomialFeatures
poly_interaction_gb = PolynomialFeatures(degree=2, interaction_only=True)
X_train_interaction_gb = poly_interaction_gb.fit_transform(X_train_gb)
X_val_interaction_gb = poly_interaction_gb.transform(X_val_gb)

# Creating a Gradient Boosting Regressor
gb_regressor_gb = GradientBoostingRegressor(random_state=42)

# Fitting the Gradient Boosting Regressor to the training data with interaction features
gb_regressor_gb.fit(X_train_interaction_gb, y_train_gb)

# Predicting on the validation set with interaction features using Gradient Boosting
y_val_pred_gb_gb = gb_regressor_gb.predict(X_val_interaction_gb)

# Calculating RMSE and R^2 score for Gradient Boosting on the validation set
rmse_gb_gb = mean_squared_error(y_val_gb, y_val_pred_gb_gb, squared=False)
r2_gb_gb = r2_score(y_val_gb, y_val_pred_gb_gb)

rmse_gb_gb, r2_gb_gb

# RESULT
# (0.0958678611189253, 0.9622757389583964)

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Defining base models for stacking
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42))
]

# Creating a Stacking Regressor with Linear Regression as the meta-model
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Fitting the Stacking Regressor to the training data with interaction features
stacking_regressor.fit(X_train_interaction_gb, y_train_gb)

# Predicting on the validation set with interaction features using Stacking Regressor
y_val_pred_stacking = stacking_regressor.predict(X_val_interaction_gb)

# Calculating RMSE and R^2 score for Stacking Regressor on the validation set
rmse_stacking = mean_squared_error(y_val_gb, y_val_pred_stacking, squared=False)
r2_stacking = r2_score(y_val_gb, y_val_pred_stacking)

rmse_stacking, r2_stacking

# RESULT
# (0.07912322728644294, 0.9743029865828227)
