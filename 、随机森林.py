# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:58:33 2025

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 导入数据集，25个输入特征，1个输出
data = pd.read_csv('D:/abc/input data/未处理的输入数据.csv')

# 提取特征和目标变量
X = data.iloc[:, 1:]  # 第2列到第26列作为输入特征（假设第1列是目标变量）
Y = data.iloc[:, 0]   # 第1列作为输出

# 标准化输入特征
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 标准化输出值
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten()  # 目标值需要 reshape 为 (n_samples, 1)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 使用随机森林回归模型进行训练
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 网格搜索调参
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500, 600, 700],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'Best parameters: {grid_search.best_params_}')

# 使用最佳参数重新训练模型
best_model = grid_search.best_estimator_
y_pred_scaled = best_model.predict(X_test)

# 反标准化预测结果
y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 评估模型
mse = mean_squared_error(y_test_original, y_pred)
print(f'Mean Squared Error: {mse}')

# 输出预测结果
print("Predictions:", y_pred)

# 特征重要性可视化
importances = best_model.feature_importances_
feature_names = data.columns[1:]  # 假设特征名在第2列到第26列

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot')
plt.show()

# 可视化真实值和预测值
plt.figure(figsize=(20, 10))
plt.plot(y_test_original, label='True Value', marker='o', linestyle='-', color='blue')
plt.plot(y_pred, label='Predicted Value', marker='x', linestyle='--', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()

# 误差分析
errors = y_test_original - y_pred

# 可视化误差分布
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()

# 分析误差较大的样本
error_indices = np.where(np.abs(errors) > np.mean(np.abs(errors)))[0]
print(f'Samples with large errors: {error_indices}')

# 计算相对误差
relative_errors = np.abs((y_test_original - y_pred) / np.abs(y_test_original)) * 100  # 以百分比形式表示

# 输出相对误差
print("Relative Errors (%):", relative_errors)

# 计算平均相对误差
mean_relative_error = np.mean(relative_errors)
print(f'Mean Relative Error: {mean_relative_error:.2f}%')

# 可视化相对误差
plt.figure(figsize=(10, 6))
plt.plot(relative_errors, marker='o', linestyle='-', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Relative Error (%)')
plt.title('Relative Error for Each Sample')
plt.axhline(y=mean_relative_error, color='red', linestyle='--', label=f'Mean Relative Error: {mean_relative_error:.2f}%')
plt.legend()
plt.show()

# 分析相对误差较大的样本
large_error_indices = np.where(relative_errors > np.mean(relative_errors))[0]
print(f'Samples with large relative errors: {large_error_indices}')
# 计算相对均方误差
mean_y_squared = np.mean(y_test_original**2)  # 计算 y_true 的平方均值
rel_mse = mse / mean_y_squared  # 相对均方误差
print(f"Relative Mean Squared Error (RelMSE): {rel_mse:.4f}")