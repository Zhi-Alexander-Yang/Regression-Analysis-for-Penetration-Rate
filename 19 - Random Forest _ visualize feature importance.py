import matplotlib.pyplot as plt

# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_rf = pd.read_csv(file_path)
df_rf = df_rf.dropna()
df_rf["Hole id"] = df_rf["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Extracting feature importance from the manually tuned Random Forest model
feature_importance = model_rf_manual.feature_importances_

# Creating a DataFrame to represent feature importance
feature_importance_df = pd.DataFrame({'Feature': features_unified, 'Importance': feature_importance})

# Sorting features based on importance
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df_sorted['Feature'], feature_importance_df_sorted['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Inverting y-axis to show the most important feature at the top
plt.show()

# Displaying the sorted feature importance
feature_importance_df_sorted

# RESULT
#                Feature  Importance
# 5   Dampening pressure    0.290271
# 4    Rotation pressure    0.182725
# 3   Flush air pressure    0.152928
# 1  Percussion pressure    0.140819
# 2        Feed pressure    0.133885
# 0              Hole id    0.099372
