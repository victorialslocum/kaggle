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
x = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# normalize data
names = x.columns
# scaler = MinMaxScaler()
# d = scaler.fit_transform(x)
# x = pd.DataFrame(d, columns=names)

d = normalize(x)
x = pd.DataFrame(d, columns=names)

# scaler = MinMaxScaler()
# d_test = scaler.fit_transform(x_test)
# x_test = pd.DataFrame(d_test, columns=names)
d_test = normalize(x_test)
x_test = pd.DataFrame(d_test, columns=names)

#split data into train and validate and test
x_train, x_validate, y_train, y_validate=train_test_split(x, y, test_size=0.20, random_state=42,shuffle=True)
x_test = torch.Tensor(x_test.to_numpy())
x_train, x_validate = torch.Tensor(x_train.to_numpy()),torch.Tensor(x_validate.to_numpy())
y_train, y_validate = torch.Tensor(y_train.to_numpy()), torch.Tensor(y_validate.to_numpy())


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 20)
        self.layer_2 = nn.Linear(20, output_dim)
    def forward(self, x): 
        outputs = self.layer_1(x)
        outputs = self.layer_2(outputs)
        return outputs

epochs = 10000
input_dim = 74
output_dim = 1
learning_rate = 0.2

model = Model(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    correct = np.sum(outputs.squeeze().detach().numpy().round() == y_train)
    total = y_train.size(0)
    accuracy = 100 * (correct/total)

    losses.append(loss.item())

    if epoch%1000==0 or epoch==0:
        # print(outputs)
        print(f"Train - Loss: {loss.item():.2f}, Accuracy: {accuracy:.2f}")

val_losses = []
with torch.no_grad():
    val_outputs = model(x_validate)
    val_loss = criterion(val_outputs.squeeze(), y_validate)
    correct = np.sum(val_outputs.detach().numpy() == y_validate)
    total = y_validate.size(0)
    accuracy = 100 * (correct/total)

    val_losses.append(val_loss.item())

    print(f"Test - Loss: {val_loss.item():.2f}, Accuracy: {accuracy:.2f}")

with torch.no_grad():
    test_outputs = model(x_test)
    print(test_outputs)

id_column = df_test['Id']
final_df = pd.DataFrame(id_column)
final_df['SalePrice'] = test_outputs.squeeze().numpy().astype(np.int32)
final_df.to_csv('/home/victoriaslocum/kaggle/housing_prices/data/submission.csv', index=np.False_)