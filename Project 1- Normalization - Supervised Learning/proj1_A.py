"""
Created on Fri Jun 12 15:52:00 2020

File A: Statistical analysis
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"data_banknote_authentication.txt", header=None)

# display covariance matrix
print("Covariance matrix: \n", df.cov())

#co-variance plot
pd.plotting.scatter_matrix(df, figsize=(13, 13))
plt.show()
