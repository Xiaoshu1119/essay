# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:41:11 2025

@author: Lenovo
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error  # 导入 MSE 计算函数

# 加载数据
data = pd.read_csv('D:/abc/input data/PCA5维.csv')  # 替换为你的实际数据

# 分离输入和输出
X1 = data.iloc[:, 1:]  # 输入特征 (7900, 25)
y1 = data.iloc[:, 0]   # 输出值 (7900,)
X = X1.to_numpy()
y = y1.to_numpy()

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 调整形状以适应模型输入
# Transformer 模型需要输入形状为 (batch_size, seq_len, input_dim)
# 这里 seq_len=1，input_dim=25
X_train = X_train.unsqueeze(1)  # (6320, 1, 25)
X_val = X_val.unsqueeze(1)      # (1580, 1, 25)

# 定义 Transformer 模型
class TransformerForRegression(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.4):
        super(TransformerForRegression, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # 将输入特征映射到嵌入维度
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 1)  # 输出层，预测一个数值

    def forward(self, src):
        seq_len = src.shape[1]
        src = self.embedding(src) + self.positional_encoding[:, :seq_len, :]
        out = self.transformer_encoder(src)
        out = self.fc_out(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义模型参数
input_dim = 11  # 输入特征维度
embed_dim = 256
num_heads = 4
num_layers = 2
ff_dim = 512
max_seq_len = 1  # 序列长度为 1

# 初始化模型
model = TransformerForRegression(input_dim, embed_dim, num_heads, num_layers, ff_dim, max_seq_len)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
# 定义学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs)

# 训练循环

losses = []  # 存储训练损失
val_losses = []  # 存储验证损失
learning_rates = []  # 存储学习率

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)  # 前向传播
    loss = criterion(out, y_train.unsqueeze(1))  # 计算损失
    losses.append(loss.item())
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 验证
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val.unsqueeze(1))
        val_losses.append(val_loss.item())
    model.train()

    # 更新学习率
    scheduler.step()  # 根据验证集损失调整学习率
    learning_rates.append(optimizer.param_groups[0]['lr'])  # 记录当前学习率

    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Learning Rate: {optimizer.param_groups[0]['lr']}")

# 绘制损失曲线
plt.figure(figsize=(12, 6))

# 绘制训练损失和验证损失
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 绘制学习率变化
plt.subplot(1, 2, 2)
plt.plot(range(len(learning_rates)), learning_rates, label='Learning Rate', color='red')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.show()

# 反标准化预测结果和真实值
y_val_original = scaler_y.inverse_transform(y_val.unsqueeze(1).numpy()).flatten()
val_out_original = scaler_y.inverse_transform(val_out.numpy()).flatten()

# 计算反标准化后的 MSE
mse_original = mean_squared_error(y_val_original, val_out_original)
print(f"Mean Squared Error (Original Scale): {mse_original}")



#计算相对均方误差
# 计算相对均方误差
mean_y_squared = np.mean(y_val_original**2)  # 计算 y_true 的平方均值
rel_mse = mse_original / mean_y_squared  # 相对均方误差
print(f"Relative Mean Squared Error (RelMSE): {rel_mse:.4f}")
#计算相对误差
relative_errors = np.abs((y_val_original - val_out_original) / np.abs(y_val_original)) * 100  # 
#输出结果
print("relative errors (%)",relative_errors)
#计算平均相对误差
mean_relative_error = np.mean(relative_errors)
#输出结果
print(f'Mean Relative Error: {mean_relative_error:.2f}%')

# 可视化真实值和预测值
plt.figure(figsize=(20, 10))
plt.plot(y_val_original, label='True Value', marker='o')
plt.plot(val_out_original, label='Predicted Value', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title(f'True vs Predicted Values (Original Scale)\nRelMSE: {rel_mse:.4f}')
plt.legend()
plt.show()