import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, normalize

df = pd.read_csv(r'/home/victoriaslocum/kaggle/housing_prices/data/train.csv')
df_test = pd.read_csv(r'/home/victoriaslocum/kaggle/housing_prices/data/test.csv')

# list of features to remove: MiscFeature, Fence, PoolQC, FireplaceQu, Alley - too many null values
df.drop(columns=['Id','MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], inplace=True)

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

# FOR TEST DATA
# list of features to remove: MiscFeature, Fence, PoolQC, FireplaceQu, Alley - too many null values
x_test = df_test.drop(columns=['Id', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'])

#turn obj columns into numbers
t_obj_col = x_test.select_dtypes('object').columns

for col in t_obj_col:
    map_list = {}
    i = 0
    for value in x_test[col].unique():
        map_list[value] = i
        i += 1
    x_test[col] = x_test[col].map(map_list)

#get rid of remaining null values with median
t_null_col = []
for col in x_test.columns:
    null_val = x_test[col].isnull().sum()
    if null_val != 0:
        t_null_col.append(col)

for col in t_null_col:
    x_test[col] = x_test[col].fillna(x_test[col].median())

# get datasets for x, y 
x_columns = list(df.drop(columns=['SalePrice']))
y_columns = ['SalePrice']

means, maxs, mins = dict(), dict(), dict()
for col in df:
    means[col] = df[col].mean()
    maxs[col] = df[col].max()
    mins[col] = df[col].min()
df = (df - df.mean()) / (df.max() - df.min())

for col in x_test.columns:
    x_test[col] = (x_test[col] - means[col]) / (maxs[col] - mins[col])

x_df = pd.DataFrame(df, columns=x_columns)
y_df = pd.DataFrame(df, columns=y_columns)
x_train = torch.tensor(x_df.values, dtype=torch.float)
y_train = torch.tensor(y_df.values, dtype=torch.float)

x_test = torch.tensor(x_test.values, dtype=torch.float)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 500)
        self.layer_2 = nn.Linear(500, 1000)
        self.layer_3 = nn.Linear(1000, 200)
        self.layer_4 = nn.Linear(200, output_dim)
    def forward(self, x): 
        outputs = self.layer_1(x)
        outputs = self.layer_2(outputs)
        outputs = self.layer_3(outputs)
        outputs = self.layer_4(outputs)
        return outputs

epochs = 500
input_dim = 74
output_dim = 1
learning_rate = 0.0001

model = Model(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch%100==0 or epoch==0:
        # print(outputs)
        print(f"Train - Loss: {loss.item():.6f}")

with torch.no_grad():
    test_outputs = model(x_test)
    print(test_outputs)
    result = pd.DataFrame(test_outputs.data.numpy(), columns=['SalePrice'])
    result['SalePrice'] = result['SalePrice'] * (maxs['SalePrice'] - mins['SalePrice']) + means['SalePrice']

    id_column = df_test['Id']
    result['Id'] = id_column

    result = pd.DataFrame(result, columns=['Id', 'SalePrice'])
    print(result)

    result.to_csv('/home/victoriaslocum/kaggle/housing_prices/data/submission.csv', index=np.False_)