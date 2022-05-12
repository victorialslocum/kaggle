import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

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

names = x.columns
scaler = MinMaxScaler()
d = scaler.fit_transform(x)
x = pd.DataFrame(d, columns=names)

#split data into train and validate
x_train, x_validate, y_train, y_validate=train_test_split(x, y, test_size=0.20, random_state=42,shuffle=True)

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
input_dim = 75
output_dim = 1
learning_rate = 0.2

model = Model(input_dim, output_dim)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs.squeeze(1), y_train)
    loss.backward()
    optimizer.step()

    correct = np.sum(outputs.detach().numpy() == y_train)
    total = y_train.size(0)
    accuracy = 100 * (correct/total)

    losses.append(loss.item())

    if epoch%1000==0 or epoch==0:
        # print(outputs)
        print(f"Train - Loss: {loss.item():.2f}, Accuracy: {accuracy:.2f}")

val_losses = []
with torch.no_grad():
    test_outputs = model(x_validate)
    val_loss = criterion(test_outputs.squeeze(1), y_validate)
    correct = np.sum(test_outputs.detach().numpy() == y_validate)
    total = y_validate.size(0)
    accuracy = 100 * (correct/total)

    val_losses.append(val_loss.item())

    print(test_outputs)
    print(f"Test - Loss: {val_loss.item():.2f}, Accuracy: {accuracy:.2f}")



# id_column = df_test['PassengerId']
# final_df = pd.DataFrame(id_column)
# final_df['Survived'] = test_outputs.round().numpy().astype(np.int32)

# # filepath = Path('/home/victoriaslocum/kaggle/titanic_ml/data/final.csv')
# final_df.to_csv('/home/victoriaslocum/kaggle/titanic_ml/data/submission.csv', index=np.False_)