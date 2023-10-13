# Function to apply Decision Tree model to a single hole and return the results
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# File path
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'

# Reload the dataset
df_decision_tree = pd.read_csv(file_path)
# Define the features and target variable
features = ["Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target = "Penetration rate"


def apply_decision_tree_to_hole(hole_data):
    X = hole_data[features]
    y = hole_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model's performance using RMSE and R-squared
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return rmse, r2

# Initialize a DataFrame to store the results for each hole
individual_decision_tree_results = pd.DataFrame(columns=["Hole id", "RMSE", "R-squared"])

# Loop through each unique hole id
for hole_id in df_decision_tree["Hole id"].unique():
    hole_data = df_decision_tree[df_decision_tree["Hole id"] == hole_id]
    rmse, r2 = apply_decision_tree_to_hole(hole_data)
    individual_decision_tree_results = individual_decision_tree_results.append({"Hole id": hole_id, "RMSE": rmse, "R-squared": r2}, ignore_index=True)

# Display the results for each hole
individual_decision_tree_results


# RESULT
#        Hole id      RMSE  R-squared
# 0   29be0312e2  0.439236   0.208099
# 1   53034ece37  0.448155  -0.157191
# 2   db95d6684b  0.366337   0.221302
# 3   07b4bd5a9d  0.551040   0.806609
# 4   5b88eb1e23  0.334409   0.857549
# 5   432e88547b  0.382692  -0.231072
# 6   c69005e9d6  0.302575   0.420789
# 7   35f0f14168  0.778181   0.046021
# 8   6a7cb0d5af  0.390825   0.297277
# 9   b8960bac07  0.453450   0.410647
# 10  519ca0a683  0.614821  -0.670420
# 11  d6abc83da1  0.744805  -0.732143
# 12  c4e502a4b2  0.340410   0.499472
# 13  be8b244c5c  0.652874  -1.309431
# 14  14c2f806ff  0.610714  -0.702390
# 15  c92afa1018  0.906652  -0.065765
