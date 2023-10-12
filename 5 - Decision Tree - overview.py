# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# File path
file_path = '/mnt/data/1-s2.0-S1365160920308121-mmc1.csv'

# Reload the dataset
df_decision_tree = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
df_decision_tree.head()


# RESULT
#       Hole id  Depth                 Time  Penetration rate  \
# 0  29be0312e2  0.025  2017-02-10T08:42:16             1.821
# 1  29be0312e2  0.050  2017-02-10T08:42:16             3.126
# 2  29be0312e2  0.077  2017-02-10T08:42:17             2.958
# 3  29be0312e2  0.106  2017-02-10T08:42:17             2.720
# 4  29be0312e2  0.131  2017-02-10T08:42:18             2.602
#
#    Percussion pressure  Feed pressure  Flush air pressure  Rotation pressure  \
# 0              124.280         23.702               4.621             38.586
# 1              136.658         25.768               5.064             38.364
# 2              140.218         26.415               5.355             37.695
# 3              142.401         26.852               5.630             37.928
# 4              139.546         26.939               5.845             37.751
#
#    Dampening pressure  Hardness  Salve  HoleNo
# 0              38.914     4.216   1938      13
# 1              39.837     8.418   1938      13
# 2              40.437     7.732   1938      13
# 3              40.752     6.842   1938      13
# 4              41.061     6.512   1938      13  
