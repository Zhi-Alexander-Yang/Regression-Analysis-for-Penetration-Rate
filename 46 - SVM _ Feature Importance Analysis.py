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

# Show the top features based on permutation importance
svm_perm_importance_pipeline_df.head()

#RESULT
#               Feature  Importance
#8             Hardness    1.044204
#4        Feed pressure    0.465603
#7   Dampening pressure    0.244541
#3  Percussion pressure    0.192447
#6    Rotation pressure    0.190542