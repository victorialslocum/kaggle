import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'/home/victoriaslocum/kaggle/housing_prices/data/train.csv')
df_test = pd.read_csv(r'/home/victoriaslocum/kaggle/housing_prices/data/test.csv')

# list of features to remove: MiscFeature, Fence, PoolQC, FireplaceQu, Alley - too many null values
df.drop(columns=['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], inplace=True)

#turn obj columns into numbers
obj_col = df.select_dtypes('object').columns

for col in obj_col:
    map_list = {}
    i = 0
    for value in df[col].unique():
        map_list[value] = i
        i += 1
    df[col] = df[col].map(map_list)

#get rid of remaining null values with median
null_col = []
for col in df.columns:
    null_val = df[col].isnull().sum()
    if null_val != 0:
        null_col.append(col)

for col in null_col:
    df[col] = df[col].fillna(df[col].median())

x = df.drop(columns=['SalePrice'])
y = df['SalePrice']

#split data into train and validate
x_train, x_validate, y_train, y_validate=train_test_split(x, y, test_size=0.20, random_state=42,shuffle=True)

#linear regression
reg = LinearRegression().fit(x_train, y_train)
lr_score = reg.score(x_validate, y_validate)
print(lr_score)

#DTR
dtr_regressor = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
dtr_score = dtr_regressor.score(x_validate, y_validate)
print(dtr_score)

#RFR - does best
rfr_reg = RandomForestRegressor().fit(x_train, y_train)
rfr_score = rfr_reg.score(x_validate, y_validate)
print(rfr_score)

