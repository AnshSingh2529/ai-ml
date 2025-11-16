import warnings
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests


warnings.filterwarnings("ignore")
sns.set()

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv"

filepath = "../data/Ames_Housing_Data.tsv"

if not os.path.isfile(filepath):
    print("Downloading....")
    response = requests.get(URL)
    with open(filepath, "wb") as f:
        f.write(response.content)
        print("Download completed")
else:
    print("Dataset already exists, Skipped Download!!")

df = pd.read_csv(filepath, sep="\t")
print("DataFrame loaded successfully")

# print(df.head())
# print(df.info())

# ------------- Simple EDA --------------

# df['Gr Liv Area'].hist()
# plt.show()

df = df.loc[df["Gr Liv Area"] <= 4000, :]
# print("Number of rows: ", df.shape[0])
# print("Number of columns: ", df.shape[1])

data = df.copy()  # ------- keep a copy of original data

# print(df.head())

# checking if all the columns uniques so there is no duplicates
# print(len(df.PID.unique()))  # Output : 2925 which is same as rows number;

df.drop(["PID", "Order"], axis=1, inplace=True)
# print(df.head()) # checking of dropped or not ? YES

# print(df.select_dtypes('number').head())
# print(df.select_dtypes('number').columns)

# ----------- Log Transformations -------------

num_cols = df.select_dtypes("number").columns
skew_limit = 0.75  # define a limit above which we will log transform
skew_vals = df[num_cols].skew()
# print(skew_vals)

skew_cols = skew_vals[abs(skew_vals) > skew_limit].sort_values(ascending=False)
# print(skew_cols)


# ------------ {Single Field}Visually See the difference after applying Log transform ------------

field = "SalePrice"

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
df[field].hist(ax=ax_before)
df[field].apply(np.log1p).hist(ax=ax_after)

ax_before.set(title="before np.log1p", ylabel="frequency", xlabel="value")
ax_after.set(title="after np.log1p", ylabel="frequency", xlabel="value")
fig.suptitle('Field "{}"'.format(field))
# plt.show()

# ------------- {Each Column} --------------

for col in skew_cols.index.values:
    if col == "SalePrice":
        continue
    df[col] = df[col].apply(np.log1p)


# ------------- How many missing values are here in your original data improve it for model

null_val = data.isnull().sum().sort_values()
# print(null_val)

smaller_df = df.loc[
    :,
    [
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Gr Liv Area",
        "Full Bath",
        "Bedroom AbvGr",
        "Fireplaces",
        "Garage Cars",
        "SalePrice",
    ],
]

# print(smaller_df.describe())
smaller_df = smaller_df.fillna(0)
# print(smaller_df.info())


# ---------- Pair Plot of Features -----------

# sns.pairplot(smaller_df, plot_kws=dict(alpha=0.1, edgecolor="none"))
# plt.show()

# --------- Separate features and target variable ---------

X = smaller_df.loc[:, smaller_df.columns != "SalePrice"]
Y = smaller_df["SalePrice"]
# print(X.info())

# ---------- Manual Polynomial Features ---------

X2 = X.copy()
X2["OQ2"] = X2["Overall Qual"] ** 2
X2["GLA2"] = X2["Gr Liv Area"] ** 2

# ----------- Interaction Features ---------
X3 = X2.copy()
# multiplicative interaction
X3["OQ_x_YB"] = X3["Overall Qual"] * X3["Year Built"]
# division interaction
X3["OQ_/_LA"] = X3["Overall Qual"] / X3["Lot Area"]

# ------------ Applying One-Hot Encoding & Categorical Encoding --------------
# -----------for all categorical columns -----------

# one_hot_enc_cols = df.dtypes.include('object').columns
# one_hot_enc_cols = one_hot_enc_cols.tolist()
# print(one_hot_enc_cols)
# df = pd.get_dummies(df, columns=one_hot_enc_cols, drop_first=True)
# print(df.head())

# ----------- for a single categorical column -----------

# print(df['House Style'].value_counts())
single_field = "House Style"
df[single_field].value_counts()
pd.get_dummies(df[single_field], drop_first=True).head()
# print(df)

# ----------- Neighborhood column value counts -----------

nbh_counts = df.Neighborhood.value_counts()
# print(nbh_counts)

other_nbh = list(nbh_counts[nbh_counts <= 8].index)
# print(other_nbh)

X4 = X3.copy()
X4["Neighborhood"] = df["Neighborhood"].replace(other_nbh, "Other")
# print(X4['Neighborhood'].value_counts())


# print(X4.groupby("Neighborhood")["Overall Qual"].transform(lambda x: x.mean()))
# print(X4.groupby("Neighborhood")["Overall Qual"].transform(lambda x: x.std()))


def add_deviation_feature(X, feature, category):

    category_gp = X.groupby(category)[feature]
    category_mean = category_gp.transform(lambda x: x.mean())
    category_std = category_gp.transform(lambda x: x.std())

    deviation_feature = (X[feature] - category_mean) / category_std
    new_feature_name = "{}_dev_from_{}".format(feature, category)
    X[new_feature_name] = deviation_feature


X5 = X4.copy()
X5["House Style"] = df["House Style"]
add_deviation_feature(X5, "Year Built", "House Style")
add_deviation_feature(X5, "Overall Qual", "Neighborhood")
print(X5.head())