from sklearn.preprocessing import PolynomialFeatures

# Selecting features for polynomial transformation
polynomial_features_cols = ["Hardness", "Feed pressure", "Dampening pressure", "Percussion pressure"]

# Creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
polynomial_features = poly.fit_transform(df[polynomial_features_cols])

# Converting the polynomial features into a DataFrame
polynomial_features_df = pd.DataFrame(polynomial_features, columns=poly.get_feature_names(polynomial_features_cols))

# Concatenating the polynomial features with the original DataFrame
df_with_polynomial = pd.concat([df, polynomial_features_df], axis=1)

# Displaying the new features
polynomial_features_df.head()

# RESULT
#    Hardness  Feed pressure  Dampening pressure  Percussion pressure  \
# 0     4.216         23.702              38.914              124.280
# 1     8.418         25.768              39.837              136.658
# 2     7.732         26.415              40.437              140.218
# 3     6.842         26.852              40.752              142.401
# 4     6.512         26.939              41.061              139.546
#
#    Hardness^2  Hardness Feed pressure  Hardness Dampening pressure  \
# 0   17.774656               99.927632                   164.061424
# 1   70.862724              216.915024                   335.347866
# 2   59.783824              204.240780                   312.658884
# 3   46.812964              183.721384                   278.825184
# 4   42.406144              175.426768                   267.389232
#
#    Hardness Percussion pressure  Feed pressure^2  \
# 0                    523.964480       561.784804
# 1                   1150.387044       663.989824
# 2                   1084.165576       697.752225
# 3                    974.307642       721.029904
# 4                    908.723552       725.709721
#
#    Feed pressure Dampening pressure  Feed pressure Percussion pressure  \
# 0                        922.339628                        2945.684560
# 1                       1026.519816                        3521.403344
# 2                       1068.143355                        3703.858470
# 3                       1094.272704                        3823.751652
# 4                       1106.142279                        3759.229694
#
#    Dampening pressure^2  Dampening pressure Percussion pressure  \
# 0           1514.299396                             4836.231920
# 1           1586.986569                             5444.044746
# 2           1635.150969                             5669.995266
# 3           1660.725504                             5803.125552
# 4           1686.005721                             5729.898306
#
#    Percussion pressure^2
# 0           15445.518400
# 1           18675.408964
# 2           19661.087524
# 3           20278.044801
# 4           19473.086116
