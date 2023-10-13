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

# Selecting features including interaction features
features_interaction = features_unified + ["percussion_feed_interaction", "percussion_rotation_interaction", "feed_flush_interaction", "flush_dampening_interaction"]
X_interaction = df_interaction_features[features_interaction]
y_interaction = df_interaction_features[target_unified]

# Splitting the data into training and testing sets
X_train_interaction, X_test_interaction, y_train_interaction, y_test_interaction = train_test_split(X_interaction, y_interaction, test_size=0.2, random_state=42)

# Standardizing the interaction features
scaler_interaction = StandardScaler()
X_train_interaction_scaled = scaler_interaction.fit_transform(X_train_interaction)
X_test_interaction_scaled = scaler_interaction.transform(X_test_interaction)

# Training and evaluating the model with interaction features
model_interaction = DecisionTreeRegressor(random_state=42)
model_interaction.fit(X_train_interaction_scaled, y_train_interaction)
y_pred_interaction = model_interaction.predict(X_test_interaction_scaled)
rmse_interaction = mean_squared_error(y_test_interaction, y_pred_interaction, squared=False)
r2_interaction = r2_score(y_test_interaction, y_pred_interaction)

# Displaying the results for interaction features
rmse_interaction, r2_interaction

# RESULT
# (0.5861655000077538, 0.1802638295147071)
