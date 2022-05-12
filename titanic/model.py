import torch
import torch.nn as nn
import pandas as pd
import numpy as np

df_train = pd.read_csv(r'/home/victoriaslocum/kaggle/titanic_ml/data/train.csv')
df_test = pd.read_csv(r'/home/victoriaslocum/kaggle/titanic_ml/data/test.csv')

pd_train_x = df_train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']].fillna(0).replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2])
pd_train_x['SibSp'].apply(lambda x:1 if x>1 else 0)
pd_train_x['Parch'].apply(lambda x:1 if x>1 else 0)

np_test_x = df_test[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']].fillna(0).replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2]).to_numpy()

np_train_x = pd_train_x.to_numpy()
np_train_y = df_train['Survived'].to_numpy()

x_train, x_test = torch.Tensor(np_train_x),torch.Tensor(np_test_x)
y_train = torch.Tensor(np_train_y)
y_train = y_train.view(y_train.shape[0],1)

class BiClassification(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BiClassification, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 20)
        self.layer_2 = nn.Linear(20, output_dim)
    def forward(self, x): 
        outputs = self.layer_1(x)
        outputs = torch.sigmoid(self.layer_2(outputs))
        return outputs

epochs = 5000
input_dim = 5
output_dim = 1
learning_rate = 0.01

model = BiClassification(input_dim, output_dim)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    correct = np.sum(outputs.detach().round().numpy() == y_train.detach().numpy())
    total = y_train.size(0)
    accuracy = 100 * (correct/total)

    losses.append(loss.item())

    if epoch%1000==0 or epoch==0:
        # print(outputs)
        print(f"Train - Loss: {loss.item():.2f}, Accuracy: {accuracy:.2f}")

with torch.no_grad():
    test_outputs = model(x_test)

id_column = df_test['PassengerId']
final_df = pd.DataFrame(id_column)
final_df['Survived'] = test_outputs.round().numpy().astype(np.int32)

# filepath = Path('/home/victoriaslocum/kaggle/titanic_ml/data/final.csv')
final_df.to_csv('/home/victoriaslocum/kaggle/titanic_ml/data/submission.csv', index=np.False_)