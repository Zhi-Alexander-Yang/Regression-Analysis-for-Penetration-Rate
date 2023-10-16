# Identify numerical and categorical features in the original dataset
numerical_features_original = df.select_dtypes(include=['int64', 'float64']).columns.difference(["Penetration rate"])
categorical_features_original = df.select_dtypes(include=['object']).columns

# Combine transformers for the original dataset
preprocessor_original = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_original),
        ('cat', categorical_transformer, categorical_features_original)
    ])

# Separate the features (X) and the target variable (y) in the original dataset
X_original = df.drop('Penetration rate', axis=1)
y_original = df['Penetration rate']

# Split the data into training and testing sets
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# Preprocess the data
X_train_preprocessed_original = preprocessor_original.fit_transform(X_train_original)
X_test_preprocessed_original = preprocessor_original.transform(X_test_original)

# Show the shape of the preprocessed data
X_train_preprocessed_original.shape, X_test_preprocessed_original.shape

# RESULT
# ((8482, 6594), (2121, 6594))
# The preprocessing is successful, and we now have training and testing sets with 6594 features each. These features include both numerical and encoded categorical variables.
