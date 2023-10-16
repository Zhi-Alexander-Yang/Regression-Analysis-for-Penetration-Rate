# Calculate the correlation matrix
correlation_matrix = df.corr()

# Show the correlation with the target variable "Penetration rate"
correlation_with_target = correlation_matrix["Penetration rate"].sort_values(ascending=False)
correlation_with_target

# RESULT
# Penetration rate       1.000000
# Hardness               0.264004
# Salve                  0.113985
# HoleNo                 0.072359
# Rotation pressure     -0.048222
# Flush air pressure    -0.061479
# Depth                 -0.133901
# Percussion pressure   -0.205730
# Dampening pressure    -0.244864
# Feed pressure         -0.287867
# Name: Penetration rate, dtype: float64
# The correlation analysis reveals the following relationships between 
