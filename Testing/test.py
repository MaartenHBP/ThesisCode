# Importing pandas package
import pandas as pd
import numpy as np

# Creating two dictionaries
d1 = {
    'iris':[19000,178922,18327,19337],
    'test':[9221,29711,128312,201831]
}

d2 = {
    'iris':[97849,74272,1973,193713],
    'test':[974923,13719,18391,28841]
}



# Creating DataFrames
df1 = pd.DataFrame(d1)

df2 = pd.DataFrame(d2)

a = pd.concat([df1, df2], keys=['algo1', 'algo2'])
a = a.swaplevel(0, 1)
a.sort_index(axis=0, level=0, inplace=True)
a = a.stack()
print(a)

k = []
pand = []
for i in range(4):
    pand.append(pd.DataFrame(np.argsort(a.loc[i])).unstack())

a = pd.concat(pand)
print(a)



df = (
    # combine dataframes into a single dataframe
    pd.concat([df1, df2])
    # replace 0 values with nan to exclude them from mean calculation
    # group by the row within the original dataframe
    .groupby(level=0)
    # calculate the mean
    # .mean()
)

# print(df)

# Display Original DataFrames
# print("Created DataFrame 1:\n",df1,"\n")
# print("Created DataFrame: 2\n",df2,"\n")

# Finding mean of concatenated DataFrames
df_concat = pd.concat((df1, df2))

# Finding mean of concatenated DataFrame
result = df_concat.mean()

# Display result
# print("Result:\n",result)