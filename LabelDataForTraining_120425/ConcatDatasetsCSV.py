import pandas as pd

# Read individual CSV files.
df1 = pd.read_csv("LabelledData_csv\comp3PCA\FelixA1_070325Labelled3PCA.csv")
df2 = pd.read_csv("LabelledData_csv\comp3PCA\FelixB1_070325Labelled3PCA.csv")
df3 = pd.read_csv("LabelledData_csv\comp3PCA\FelixC1_070325Labelled3PCA.csv")
df4 = pd.read_csv("LabelledData_csv\comp3PCA\FelixD1_070325Labelled3PCA.csv")
df5 = pd.read_csv("LabelledData_csv\comp3PCA\FelixE1_070325Labelled3PCA.csv")

# Concatenate them vertically (stack one on top of the other).
df_all = pd.concat([df1, df2, df3, df4, df5], axis=0)

# Optionally, reset the index.
df_all = df_all.reset_index(drop=True)

# Save the combined DataFrame to a new CSV file.
df_all.to_csv("CombinedDatasets3PCA_Felix070325_ABCDE.csv", index=False)
