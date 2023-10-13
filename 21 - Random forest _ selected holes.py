# Reload the dataset
file_path = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_rf = pd.read_csv(file_path)
df_rf = df_rf.dropna()
df_rf["Hole id"] = df_rf["Hole id"].astype('category').cat.codes

# Defining features and target variable names
features_unified = ["Hole id", "Percussion pressure", "Feed pressure", "Flush air pressure", "Rotation pressure", "Dampening pressure"]
target_unified = "Penetration rate"

# Function to evaluate the Random Forest model for a specific hole
def evaluate_model_for_hole(hole_id):
    # Filtering the data for the specific hole
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    X_hole = hole_data[features_unified]
    y_hole = hole_data[target_unified]

    # Splitting the data into training and testing sets
    X_train_hole, X_test_hole, y_train_hole, y_test_hole = train_test_split(X_hole, y_hole, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler_hole = StandardScaler()
    X_train_hole_scaled = scaler_hole.fit_transform(X_train_hole)
    X_test_hole_scaled = scaler_hole.transform(X_test_hole)

    # Training and evaluating the Random Forest model for the specific hole
    model_rf_hole = RandomForestRegressor(**manual_params, random_state=42)
    model_rf_hole.fit(X_train_hole_scaled, y_train_hole)
    y_pred_hole = model_rf_hole.predict(X_test_hole_scaled)
    rmse_hole = mean_squared_error(y_test_hole, y_pred_hole, squared=False)
    r2_hole = r2_score(y_test_hole, y_pred_hole)

    return rmse_hole, r2_hole

# Selecting specific holes for site-specific analysis
selected_holes = [7, 5, 10]

# Analyzing the features and target for the selected holes
site_specific_analysis = []
for hole_id in selected_holes:
    hole_data = df_rf[df_rf['Hole id'] == hole_id]
    hole_summary = hole_data.describe().loc[['mean', 'std'], features_unified + [target_unified]]
    hole_summary['Hole ID'] = hole_id
    site_specific_analysis.append(hole_summary)

# Combining the analysis results for selected holes
site_specific_analysis_df = pd.concat(site_specific_analysis, axis=0).reset_index()
site_specific_analysis_df.rename(columns={'index': 'Statistic'}, inplace=True)
site_specific_analysis_df

# RESULT
#   Statistic  Hole id  Percussion pressure  Feed pressure  Flush air pressure  \
# 0      mean      7.0           184.806161      81.712832            7.630844
# 1       std      0.0            17.194886      16.888402            0.885011
# 2      mean      5.0           179.326531      78.084568            9.254547
# 3       std      0.0            22.338963      20.899398            1.405893
# 4      mean     10.0           182.138040      81.158288            6.627359
# 5       std      0.0            19.767131      18.391709            0.669039
#
#    Rotation pressure  Dampening pressure  Penetration rate  Hole ID
# 0          56.048508           67.025483          2.345987        7
# 1           4.627629            8.458985          0.845775        7
# 2          55.390133           64.471657          2.305408        5
# 3           6.198990           10.276423          0.600794        5
# 4          53.026263           65.437804          1.915715       10
# 5           6.927983            8.890068          0.491717       10
