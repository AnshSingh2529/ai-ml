import warnings
import pandas as pd
import seaborn as sns
import requests
import os
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

sns.set()

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv"

os.makedirs("data", exist_ok=True)
file_path = "./data/Ames_Housing_Data.tsv"

if not os.path.isfile(file_path):
    print("Downloading dataset...")
    response = requests.get(URL)

    with open(file_path, "wb") as f:
        f.write(response.content)
    print("Download completed.")

else:
    print("Dataset already exists. Skipping download.")


# load the dataset into a pandas dataframe
df = pd.read_csv(file_path, sep="\t")
print("Dataframe loaded successfully.")

# df.info()

df = df.loc[df["Gr Liv Area"] <= 4000, :]
# print("Number of the rows in the data", df.shape[0])
# print("Number of the columns in the data", df.shape[1])

data = df.copy()  # keeps Copy of the data

# print(df.head())


# Get a Pd.Series consisting of all the string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
# print(one_hot_encode_cols)
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
# print(one_hot_encode_cols)
df[one_hot_encode_cols].head().T
# print(df[one_hot_encode_cols].head().T)

# One-Hot Encoding of categorical(dummy) variables
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
# print(df.describe().T)

# --------------- Log transforming skew variables -------------
# --------------- using np.log() or np.log1p() ----------------
mask = data.dtypes == float  # boolean mask for float columns
# print(mask)

float_cols = data.columns[mask]
# print(float_cols)  # list of float columns

"""
Skewness measures how asymmetric a distribution is.

    1.A perfectly symmetrical (normal) distribution has skewness ≈ 0.
    2.Positive skew → tail extends to the right (e.g., house prices).
    3.Negative skew → tail extends to the left.
"""
skew_limit = 0.75  # threshold for skewness

"""
data[float_cols] - selects all the float-type columns.
.skew() - computes Pearson's skewness coefficient for each column.
"""
skew_vals = data[float_cols].skew()
# print(skew_vals)

"""
Converts the Series (skew_vals) into a DataFrame (a table-like structure).
    After to_frame() - there is a new column in the table named as 0
    .rename() - used to rename that 0 named column to 'Skew'

This filters the table to keep only columns whose absolute skew is greater than the threshold (skew_limit = 0.75).

    abs(Skew) handles both positive and negative skew.
    The .query() method is a convenient, SQL-like filter syntax.
    (If a column had Skew = 0.40, it'd be filtered out.)
"""
skew_cols = (
    skew_vals.sort_values(ascending=False)
    .to_frame()
    .rename(columns={0: "Skew"})
    .query("abs(Skew) > {}".format(skew_limit))
)
# print(skew_cols)

field = "BsmtFin SF 1"

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
# ---- Before using Log transformation
df[field].hist(ax=ax_before)
# ---- After using log transformation
df[field].apply(np.log1p).hist(ax=ax_after)

ax_before.set(title="before np.log1p", ylabel="frequency", xlabel="values")
ax_after.set(title="after np.log1p", ylabel="frequency", xlabel="values")

fig.suptitle('Field "{}"'.format(field))

# for showing the visuals
# plt.show()
# print("Visuals shown successfully!!")

#--------------------- Perform the skew transformation ----------------------

for col in skew_cols.index.values:
    if col == 'SalePrice':    #--- Skips this column and iterate next.
        continue
    df[col] = df[col].apply(np.log1p)    # --- apply log transformations

df.shape

# df = data
# data.isnull().sum().sort_values()  #Aggregation on data -- sum count of nulls in the data


# smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
#                       'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
#                       'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
#                       'Garage Cars','SalePrice']]

# smaller_df.describe().T
# smaller_df.info()

# smaller_df = smaller_df.fillna(0)   #--- finds n/a value and fill it with 0.
# smaller_df.info()

# sns.pairplot(smaller_df, plot_kws=dict(alpha=.1, edgecolor='none'))

#Separate our features from our target  -------- MANUAL VERSION OF USING POLYNOMIAL FEATURES

# X = smaller_df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
#                       'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
#                       'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
#                       'Garage Cars']]

# y = smaller_df['SalePrice']

# X.info()

# X2 = X.copy()

# X2['OQ2'] = X2['Overall Qual'] ** 2
# X2['GLA2'] = X2['Gr Liv Area'] ** 2

# X3 = X2.copy()

# # multiplicative interaction
# X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built']

# # division interaction
# X3['OQ_/_LA'] = X3['Overall Qual'] / X3['Lot Area']

# data['House Style'].value_counts()
# pd.get_dummies(df['House Style'], drop_first=True).head()

# nbh_counts = df.Neighborhood.value_counts()
# nbh_counts

# other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
# other_nbhs

# X4 = X3.copy()
# X4['Neighborhood'] = df['Neighborhood'].replace(other_nbhs, 'Other')



# def add_deviation_feature(X, feature, category):
    
#     # temp groupby object
#     category_gb = X.groupby(category)[feature]
    
#     # create category means and standard deviations for each observation
#     category_mean = category_gb.transform(lambda x: x.mean())
#     category_std = category_gb.transform(lambda x: x.std())
    
#     # compute stds from category mean for each feature value,
#     # add to X as new feature
#     deviation_feature = (X[feature] - category_mean) / category_std 
#     X[feature + '_Dev_' + category] = deviation_feature  



# X5 = X4.copy()
# X5['House Style'] = df['House Style']
# add_deviation_feature(X5, 'Year Built', 'House Style')
# add_deviation_feature(X5, 'Overall Qual', 'Neighborhood')


# AUTOMATIC VERSION OF USING POLYNOMIAL FEATURE OF Scikit-learn
from sklearn.preprocessing import PolynomialFeatures

#Instantiate and provide desired degree; 
#   Note: degree=2 also includes intercept, degree 1 terms, and cross-terms

pf = PolynomialFeatures(degree=2)

features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])

pf.get_feature_names_out()  #Must add input_features = features for appropriate names

feat_array = pf.transform(df[features])
df = pd.DataFrame(feat_array, columns = pf.get_feature_names_out(input_features=features))
print(df.head())