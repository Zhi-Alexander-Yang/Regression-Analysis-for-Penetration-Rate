# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Reloading the dataset
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

# Data shape after preprocessing
X_train_interaction_all.shape, X_val_interaction_all.shape

# RESULT
# ((8482, 46), (2121, 46))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Defining base models for stacking
base_models_all = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42))
]

# Creating a Stacking Regressor with Linear Regression as the meta-model
stacking_regressor_all = StackingRegressor(estimators=base_models_all, final_estimator=LinearRegression())

# Fitting the Stacking Regressor to the training data with interaction features
stacking_regressor_all.fit(X_train_interaction_all, y_train_all)

# Predicting on the validation set with interaction features using Stacking Regressor
y_val_pred_stacking_all = stacking_regressor_all.predict(X_val_interaction_all)

# Calculating RMSE and R^2 score for Stacking Regressor on the validation set
rmse_stacking_all = mean_squared_error(y_val_all, y_val_pred_stacking_all, squared=False)
r2_stacking_all = r2_score(y_val_all, y_val_pred_stacking_all)

rmse_stacking_all, r2_stacking_all
