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

rmse, r2

# RESULT
# (0.4648045377198907, 0.4845645198989308)
