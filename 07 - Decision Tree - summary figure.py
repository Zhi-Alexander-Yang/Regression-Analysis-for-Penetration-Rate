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

# Summary statistics for RMSE and R-squared values across all holes
decision_tree_summary = decision_tree_results[["RMSE", "R-squared"]].describe()

# Plotting the distribution of RMSE and R-squared values
decision_tree_results[["RMSE", "R-squared"]].plot(kind='box', subplots=True, layout=(1,2), figsize=(12, 5), title="Distribution of RMSE and R-squared for Decision Trees")

decision_tree_summary


# RESULT
#             RMSE  R-squared
# count  16.000000  16.000000
# mean    0.519823  -0.006291
# std     0.181065   0.602174
# min     0.302575  -1.309431
# 25%     0.378603  -0.340909
# 50%     0.450802   0.127060
# 75%     0.624334   0.413182
# max     0.906652   0.857549
