# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans

# Loading the dataset again
data_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
data_all_holes = pd.read_csv(data_path)

# Selecting features for clustering
features_for_clustering = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
X_clustering = data_all_holes[features_for_clustering]

# Training a KMeans model with an arbitrary number of clusters (let's start with 5)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_clustering)

# Adding the cluster labels as a new feature to the data
data_all_holes['Cluster'] = kmeans.labels_

# Importing necessary libraries
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Selecting features and target variable
features = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure', 'Cluster']
target = 'Penetration rate'

X = data_all_holes[features]
y = data_all_holes[target]

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Polynomial features for numerical features, and one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PolynomialFeatures(degree=2, include_bias=False), features[:-1]),
        ('cat', 'passthrough', [features[-1]])
    ])

# Defining the base estimators
base_estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Defining the final estimator
final_estimator = LinearRegression()

# Creating a Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, n_jobs=-1)

# Creating a pipeline with the preprocessor and the Stacking Regressor
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', stacking_regressor)])

# Training the pipeline on the training set
pipeline.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred = pipeline.predict(X_val)

# Calculating RMSE and R^2 score for the predictions
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

# Importing necessary libraries
from sklearn.decomposition import PCA

# Creating a PCA object
pca = PCA()

# Fitting the PCA object to the training data and transforming it
X_train_pca = pca.fit_transform(X_train)

# Transforming the validation data using the PCA object
X_val_pca = pca.transform(X_val)

# Training the Stacking Regressor on the PCA-transformed training data
stacking_regressor.fit(X_train_pca, y_train)

# Predicting on the PCA-transformed validation set
y_val_pred_pca = stacking_regressor.predict(X_val_pca)

# Calculating RMSE and R^2 score for the predictions
rmse_pca = mean_squared_error(y_val, y_val_pred_pca, squared=False)
r2_pca = r2_score(y_val, y_val_pred_pca)

rmse_pca, r2_pca

# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans

# Loading the dataset again
data_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
data_all_holes = pd.read_csv(data_path)

# Selecting features for clustering
features_for_clustering = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
X_clustering = data_all_holes[features_for_clustering]

# Training a KMeans model with an arbitrary number of clusters (let's start with 5)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_clustering)

# Adding the cluster labels as a new feature to the data
data_all_holes['Cluster'] = kmeans.labels_

# Importing necessary libraries
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Selecting features and target variable
features = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure', 'Cluster']
target = 'Penetration rate'

X = data_all_holes[features]
y = data_all_holes[target]

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Polynomial features for numerical features, and one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PolynomialFeatures(degree=2, include_bias=False), features[:-1]),
        ('cat', 'passthrough', [features[-1]])
    ])

# Defining the base estimators
base_estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Defining the final estimator
final_estimator = LinearRegression()

# Creating a Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, n_jobs=-1)

# Creating a pipeline with the preprocessor and the Stacking Regressor
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', stacking_regressor)])

# Training the pipeline on the training set
pipeline.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred = pipeline.predict(X_val)

# Calculating RMSE and R^2 score for the predictions
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
r2 = r2_score(y_val, y_val_pred)

# Importing necessary libraries for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Defining the hyperparameters to tune
param_grid = {
    'regressor__rf__n_estimators': [50, 100, 200],
    'regressor__rf__max_depth': [10, 20, None],
    'regressor__gb__n_estimators': [50, 100, 200],
    'regressor__gb__max_depth': [3, 5, 7],
    'regressor__gb__learning_rate': [0.1, 0.01, 0.001]
}

# Creating a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fitting the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Getting the best parameters
best_params = grid_search.best_params_

# Creating a pipeline with the preprocessor and the Stacking Regressor with the best parameters
pipeline_tuned = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', StackingRegressor(estimators=[
                                     ('rf', RandomForestRegressor(n_estimators=best_params['regressor__rf__n_estimators'], max_depth=best_params['regressor__rf__max_depth'], random_state=42, n_jobs=-1)),
                                     ('gb', GradientBoostingRegressor(n_estimators=best_params['regressor__gb__n_estimators'], max_depth=best_params['regressor__gb__max_depth'], learning_rate=best_params['regressor__gb__learning_rate'], random_state=42))
                                 ], final_estimator=LinearRegression(), n_jobs=-1))])

# Training the pipeline on the training set
pipeline_tuned.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred_tuned = pipeline_tuned.predict(X_val)

# Calculating RMSE and R^2 score for the predictions
rmse_tuned = mean_squared_error(y_val, y_val_pred_tuned, squared=False)
r2_tuned = r2_score(y_val, y_val_pred_tuned)

best_params, rmse_tuned, r2_tuned
