# Step 1.1: Load the Data
# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
data = pd.read_csv(file_path)

# Step 1.2: Check for Missing Values
# Check for missing values in each column
missing_values = data.isnull().sum()

# Step 1.3: Encode Categorical Features
# Selecting categorical features
categorical_features = ['Hole id', 'Time']

# Applying one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_features)

# Step 1.5: Split the Data (Revised)
# Separate the features (X) and the target variable (y)
X_original = data_original.drop('Penetration rate', axis=1)
y_original = data_original['Penetration rate']

# Split the data into training and testing sets (80% training, 20% testing)
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42)

# Step 1.4: Scale the Features (Revised)
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training set
X_train_scaled = scaler.fit_transform(X_train_original)

# Transform the test set
X_test_scaled = scaler.transform(X_test_original)

# Display the shape of the scaled training and test sets
X_train_scaled.shape, X_test_scaled.shape

#RESULT
#((8482, 14), (2121, 14))

# Step 2.1: Build and Train the SVM Model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the SVM model with RBF kernel
svm_model = SVR(kernel='rbf')

# Train the model on the scaled training data
svm_model.fit(X_train_scaled, y_train_original)

# Predict on the test set
y_test_pred_svm = svm_model.predict(X_test_scaled)

# Calculate and print the RMSE and R^2 scores
rmse_svm = mean_squared_error(y_test_original, y_test_pred_svm, squared=False)
r2_svm = r2_score(y_test_original, y_test_pred_svm)

# Compute permutation importance on the test set using the pipeline (including preprocessing and SVM model)
perm_importance_svm_pipeline = permutation_importance(svm_pipeline, X_test, y_test, n_repeats=10, random_state=42)

# Create a DataFrame to display the feature importance using the original feature names
svm_perm_importance_pipeline_df = pd.DataFrame({'Feature': feature_names, 'Importance': perm_importance_svm_pipeline.importances_mean})
svm_perm_importance_pipeline_df = svm_perm_importance_pipeline_df.sort_values(by='Importance', ascending=False)

# Correcting the names of the features for interaction terms
interaction_features = [
    "Hardness",
    "Feed pressure",
    "Dampening pressure",
    "Percussion pressure",
    "Rotation pressure",
]

# Creating interaction terms for these features
for i, feature1 in enumerate(interaction_features):
    for feature2 in interaction_features[i + 1:]:
        interaction_term = f"{feature1}_x_{feature2}"
        X_train[interaction_term] = X_train[feature1] * X_train[feature2]
        X_test[interaction_term] = X_test[feature1] * X_test[feature2]

# Previewing the updated training data with interaction terms
X_train.head()

#RESULT
         Hole id   Depth                 Time  Percussion pressure  \
#6717  b8960bac07   8.237  2017-02-10T13:03:34              192.115   
#6974  b8960bac07  14.535  2017-02-10T13:07:33              139.620   
#9928  14c2f806ff  12.320  2017-02-10T14.51.56              191.345   
#1482  db95d6684b   1.407  2017-02-10T09:04:07              190.311   
#3413  5b88eb1e23  14.269  2017-02-10T11:56:57              191.574   
#
#      Feed pressure  Flush air pressure  Rotation pressure  \
#6717         88.666               9.563             52.842   
#6974         54.034               7.803             50.459   
#9928         86.177               7.727             60.777   
#1482         90.268               6.977             51.532   
#3413         88.680               8.277             52.992   
#
#      Dampening pressure  Hardness  Salve  ...  Hardness_x_Feed pressure  \
#6717              66.235     3.781   1940  ...                335.246146   
#6974              50.899     1.772   1940  ...                 95.748248   
#9928              70.762     3.548   1941  ...                305.755996   
#1482              72.078     3.580   1938  ...                323.159440   
#3413              66.466     3.703   1939  ...                328.382040   
#
#     Hardness_x_Dampening pressure  Hardness_x_Percussion pressure  \
#6717                     250.434535                      726.386815   
#6974                      90.193028                      247.406640   
#9928                     251.063576                      678.892060   
#1482                     258.039240                      681.313380   
#3413                     246.123598                      709.398522   
#
#      Hardness_x_Rotation pressure  Feed pressure_x_Dampening pressure  \
#6717                    199.795602                         5872.792510   
#6974                     89.413348                         2750.276566   
#9928                    215.636796                         6098.056874   
#1482                    184.484560                         6506.336904   
#3413                    196.229376                         5894.204880   
#
#      Feed pressure_x_Percussion pressure  Feed pressure_x_Rotation pressure  \
#6717                         17034.068590                        4685.288772   
#6974                          7544.227080                        2726.501606   
#9928                         16489.538065                        5237.579529   
#1482                         17178.993348                        4651.690576   
#3413                         16988.782320                        4699.330560   
#
#      Dampening pressure_x_Percussion pressure  \
#6717                              12724.737025   
#6974                               7106.518380   
#9928                              13539.954890   
#1482                              13717.236258   
#3413                              12733.157484   
#
#      Dampening pressure_x_Rotation pressure  \
#6717                             3499.989870   
#6974                             2568.312641   
#9928                             4300.702074   
#1482                             3714.323496   
#3413                             3522.166272   
#
#      Percussion pressure_x_Rotation pressure  
#6717                             10151.740830  
#6974                              7045.085580  
#9928                             11629.375065  
#1482                              9807.106452  
#3413                             10151.889408  
#
#[5 rows x 21 columns]

# Retraining the SVM model with interaction terms
svm_model_with_interactions = SVR(kernel='linear', C=1)
svm_model_with_interactions.fit(X_train_scaled, y_train)

# Predicting the test data
y_pred_with_interactions = svm_model_with_interactions.predict(X_test_scaled)

# Evaluating the model's performance
rmse_with_interactions = mean_squared_error(y_test, y_pred_with_interactions, squared=False)
r2_score_with_interactions = r2_score(y_test, y_pred_with_interactions)

rmse_with_interactions, r2_score_with_interactions

#RESULT
#(0.6798403404788429, -0.10267481124581779)