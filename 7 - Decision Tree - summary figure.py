# Summary statistics for RMSE and R-squared values across all holes
decision_tree_summary = decision_tree_results[["RMSE", "R-squared"]].describe()

# Plotting the distribution of RMSE and R-squared values
decision_tree_results[["RMSE", "R-squared"]].plot(kind='box', subplots=True, layout=(1,2), figsize=(12, 5), title="Distribution of RMSE and R-squared for Decision Trees")

decision_tree_summary


# RESULT
#             RMSE  R-squared
# count  16.000000  16.000000
# mean    0.519823  -0.006291
# std     0.181065   0.602174
# min     0.302575  -1.309431
# 25%     0.378603  -0.340909
# 50%     0.450802   0.127060
# 75%     0.624334   0.413182
# max     0.906652   0.857549
