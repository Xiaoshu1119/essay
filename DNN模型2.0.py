# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:02:27 2025

@author: Lenovo
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('D:/abc/input data/PCA5维.csv')
#标准化数据
scaler_data = StandardScaler()
data = scaler_data.fit_transform(data)
# 提取输入特征 (X1 到 X22) 和输出
data = pd.DataFrame(data)
input_features = data.iloc[:, 1:]  # 提取第 2 列到第 26 列
output = data.iloc[:, 0]  # 提取第 1 列作为输出

# 将 Pandas DataFrame 转换为 PyTorch 张量
x = torch.tensor(input_features.values, dtype=torch.float32)
y1 = torch.tensor(output.values, dtype=torch.float32).reshape(-1, 1)  # 转换为 (n, 1) 的形状

# 拼接输入和输出数据
data_2 = torch.cat([x, y1], dim=1)

# 划分训练集和测试集
train_size = int(len(data_2) * 0.7)
test_size = len(data_2) - train_size
data_2 = data_2[torch.randperm(data_2.size(0)), :]  # 打乱数据
train_data = data_2[:train_size, :]  # 训练集
test_data = data_2[train_size:, :]  # 测试集

# 定义模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 卷积后展平，接若干全连接层与dropout
        self.fc_block = nn.Sequential(
            nn.Linear(40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # (batch, 25) -> (batch, 1, 25)
        x = x.unsqueeze(1)
        x = self.conv_block(x)  # (batch, 8, 25//2)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

model = DNN()

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
epochs = 10000
losses = []

X = train_data[:, :11]
Y = train_data[:, -1:]
print("train start\n")
for epoch in range(epochs):
    pred = model(X)
    loss = loss_fn(pred, Y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

# 绘制损失曲线
plt.plot(range(len(losses)), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 测试模型
X1 = test_data[:, :11]
Y2 = test_data[:, -1:]

with torch.no_grad():
    pred = model(X1)
    mse = loss_fn(pred, Y2)  # 计算均方误差
    print(f'测试集均方误差 (MSE): {mse.item()}')


# 将预测值和真实值合并为一个 DataFrame


pred_numpy = pred.numpy()
plt.figure(figsize=(20,10))
plt.plot(Y2.numpy(), label='True Value', marker='o')
plt.plot(pred_numpy, label='Predicted Value', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()
results = pd.DataFrame({
    'True_Value': Y2.numpy().flatten(),  # 真实值
    'Predicted_Value': pred_numpy.flatten()  # 预测值
})

# 保存到 CSV 文件
results.to_csv('results.csv', index=False)