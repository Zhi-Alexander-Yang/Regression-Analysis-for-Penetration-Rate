# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_polynomial_features = pd.read_csv(file_path)
df_polynomial_features = df_polynomial_features.dropna()
df_polynomial_features["Hole id"] = df_polynomial_features["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Creating interaction features
df_interaction_features = df_polynomial_features.copy()
df_interaction_features["percussion_feed_interaction"] = df_interaction_features["Percussion pressure"] * df_interaction_features["Feed pressure"]
df_interaction_features["percussion_rotation_interaction"] = df_interaction_features["Percussion pressure"] * df_interaction_features["Rotation pressure"]
df_interaction_features["feed_flush_interaction"] = df_interaction_features["Feed pressure"] * df_interaction_features["Flush air pressure"]
df_interaction_features["flush_dampening_interaction"] = df_interaction_features["Flush air pressure"] * df_interaction_features["Dampening pressure"]

import numpy as np

# Creating normalized features
df_normalized_features = df_interaction_features.copy()
df_normalized_features["percussion_pressure_log"] = np.log1p(df_normalized_features["Percussion pressure"])
df_normalized_features["feed_pressure_sqrt"] = np.sqrt(df_normalized_features["Feed pressure"])
df_normalized_features["flush_air_pressure_exp"] = np.exp(df_normalized_features["Flush air pressure"])

# Selecting features including normalized features
features_normalized = features_interaction + ["percussion_pressure_log", "feed_pressure_sqrt", "flush_air_pressure_exp"]
X_normalized = df_normalized_features[features_normalized]
y_normalized = df_normalized_features[target_unified]

# Splitting the data into training and testing sets
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Standardizing the normalized features
scaler_normalized = StandardScaler()
X_train_normalized_scaled = scaler_normalized.fit_transform(X_train_normalized)
X_test_normalized_scaled = scaler_normalized.transform(X_test_normalized)

# Training and evaluating the model with normalized features
model_normalized = DecisionTreeRegressor(random_state=42)
model_normalized.fit(X_train_normalized_scaled, y_train_normalized)
y_pred_normalized = model_normalized.predict(X_test_normalized_scaled)
rmse_normalized = mean_squared_error(y_test_normalized, y_pred_normalized, squared=False)
r2_normalized = r2_score(y_test_normalized, y_pred_normalized)

# Displaying the results for normalized features
rmse_normalized, r2_normalized

# RESULT
# (0.5870354772749631, 0.17782874571131968)
