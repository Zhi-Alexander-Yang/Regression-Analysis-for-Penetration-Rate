# Loading the newly provided database
file_path_new = '/data/1-s2.0-S1365160920308121-mmc1.csv'
df_gb = pd.read_csv(file_path_new)

# Updating the feature and target columns based on the new dataset
features_gb = ['Percussion pressure', 'Feed pressure', 'Flush air pressure', 'Rotation pressure', 'Dampening pressure']
target_gb = 'Penetration rate'

# Summary of the data for Gradient Boosting analysis
df_gb_summary = df_gb[features_gb + [target_gb, 'Hole id']].describe().transpose()
df_gb_summary['Hole ID'] = df_gb_summary.index
df_gb_summary.reset_index(drop=True, inplace=True)
df_gb_summary

# RESULT
#      count        mean        std     min       25%      50%       75%  \
# 0  10603.0  182.329205  19.256709   0.213  188.3305  190.605  191.4290
# 1  10603.0   81.150078  17.277436   0.342   85.3385   88.834   89.0910
# 2  10603.0    7.972507   1.188331   2.173    7.1150    7.925    8.7360
# 3  10603.0   53.407251   5.823058  29.829   50.6995   53.840   56.8045
# 4  10603.0   65.867379   9.157478  35.467   65.2035   69.120   71.4205
# 5  10603.0    2.122995   0.685980   0.350    1.8470    2.076    2.2410
#
#        max              Hole ID
# 0  201.141  Percussion pressure
# 1   93.844        Feed pressure
# 2   11.460   Flush air pressure
# 3  165.846    Rotation pressure
# 4   81.053   Dampening pressure
# 5   19.046     Penetration rate  
