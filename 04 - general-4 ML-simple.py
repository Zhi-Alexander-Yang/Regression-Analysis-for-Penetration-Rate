from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

# Load the data
df = pd.read_csv('data/1-s2.0-S1365160920308121-mmc1.csv')


# Initialize the models with default parameters
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)
nn = MLPRegressor(random_state=42, max_iter=500)

models = {'Decision Tree': dt, 'Random Forest': rf, 'Gradient Boosting': gb, 'Neural Network': nn}
results = {}


# Select data for the hole with id 'c69005e9d6'
df_single_hole = df[df['Hole id'] == 'c69005e9d6']


X_single_hole = df_single_hole[["Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]]
Y_single_hole = df_single_hole['Penetration rate']




# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_single_hole, Y_single_hole, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    r2 = r2_score(Y_test, Y_pred)
    
    results[model_name] = {'RMSE': rmse, 'R-squared': r2}

# Convert the results to a DataFrame for easier viewing
results_df = pd.DataFrame(results).T

results_df

#RESULT
#                       RMSE  R-squared
#Decision Tree      0.302575   0.420789
#Random Forest      0.249455   0.606307
#Gradient Boosting  0.263026   0.562307
#Neural Network     0.346799   0.239098

