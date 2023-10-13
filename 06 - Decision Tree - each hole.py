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

# Initialize a DataFrame to store the results for each hole
decision_tree_results = pd.DataFrame(columns=["Hole id", "RMSE", "R-squared"])

# Loop through each unique hole id
for hole_id in df_decision_tree["Hole id"].unique():
    # Extract the data for the hole
    hole_data = df_decision_tree[df_decision_tree["Hole id"] == hole_id]
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

    # Store the results
    decision_tree_results = decision_tree_results.append({"Hole id": hole_id, "RMSE": rmse, "R-squared": r2}, ignore_index=True)

# Display the results for each hole
decision_tree_results.head()


# RESULT
#       Hole id      RMSE  R-squared
# 0  29be0312e2  0.439236   0.208099
# 1  53034ece37  0.448155  -0.157191
# 2  db95d6684b  0.366337   0.221302
# 3  07b4bd5a9d  0.551040   0.806609
# 4  5b88eb1e23  0.334409   0.857549
