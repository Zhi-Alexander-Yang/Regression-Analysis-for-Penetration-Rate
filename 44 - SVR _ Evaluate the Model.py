# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Loading the data
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df = pd.read_csv(file_path)

# Define the target variable and features
target_column = 'Penetration rate'
features = df.drop(columns=[target_column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target_column], test_size=0.2, random_state=42)

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())]) # Scaling is important for SVR

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Preprocessed training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define the SVR model
svr_model = SVR()

# Defining a separate numerical transformer that includes standard scaling
numerical_transformer_with_scaling = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler(with_mean=False)) # Standard scaling without mean centering
])

# Creating a preprocessor that applies the numerical transformer with scaling to numerical features
# and keeps the categorical features unchanged after one-hot encoding
preprocessor_svr = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_with_scaling, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating SVR pipeline with the updated preprocessor and SVR model
svr_pipeline = make_pipeline(
    preprocessor_svr,
    SVR()
)

# Creating the GridSearchCV object for SVR with reduced search space
grid_search_svr_reduced = GridSearchCV(estimator=svr_pipeline, param_grid=param_grid_svr_reduced,
                                       cv=3, n_jobs=-1, verbose=1)

from sklearn.metrics import mean_squared_error, r2_score

# Predicting on the test set using the best model
y_test_pred_svr_reduced = grid_search_svr_reduced.predict(X_test)

# Calculating and printing the RMSE and R^2 scores for SVR with reduced search space
rmse_svr_reduced = mean_squared_error(y_test, y_test_pred_svr_reduced, squared=False)
r2_svr_reduced = r2_score(y_test, y_test_pred_svr_reduced)

best_params_svr_reduced, rmse_svr_reduced, r2_svr_reduced

# RESULT
# ({'svr__C': 0.1, 'svr__epsilon': 1, 'svr__kernel': 'linear'},
#  0.6032231665854936,
#  0.13186030250353897)
