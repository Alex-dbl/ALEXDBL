# -*- coding = utf-8 -*-
# @Time: 2024/7/31 16:23
# @Author: ALEX
# @File：model.py
# @Desc:Predicting ST Stocks Using XGBoost
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xgboost
import os
import graphviz
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from scipy.stats import uniform, randint
from xgboost import plot_tree
from graphviz import Graph

#引入可视化环境变量，需要自行配置环境
dot_path = r"C:\Program Files (x86)\Graphviz2.38\bin"
os.environ["PATH"] += os.pathsep + dot_path

# 加载预测数据
df = pd.read_excel(r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Data\all-2024-3月前.xlsx")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001, random_state=42, stratify=y)

# 定义XGBoost分类器
model = XGBClassifier(eval_metric='mlogloss')

# 定义参数的分布，从中随机选择参数进行搜索
param_dist = {
    'max_depth': randint(3, 8),  # 随机选择3到7之间的整数
    'min_child_weight': randint(1, 5),
    'gamma': uniform(0.5, 1.5),  # 从0.5到1.5的均匀分布
    'subsample': uniform(0.5, 1.0),
    'colsample_bytree': uniform(0.5, 1.0),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 500),  # 随机选择100到400之间的整数
}

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=1000,  # 要尝试的参数组合数量
    scoring='roc_auc',
    cv=5,  # 5-fold cross validation
    verbose=2,
    random_state=42,
    n_jobs=-1,  # 使用所有可用的处理器
    pre_dispatch='2*n_jobs',  # 用于控制作业分派的线程或进程数
    error_score=np.nan  # 如何处理出现的错误，这里使用np.nan
)

# 训练模型并进行参数搜索
random_search.fit(X_train, y_train)

# 输出最佳参数和最佳分数
print("Best parameters:", random_search.best_params_)
print("Best ROC AUC score:", random_search.best_score_)

# 使用最佳参数的模型进行预测
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 评估模型并绘制混淆矩阵
conf_matrix = confusion_matrix(y_val, (y_pred_proba > 0.965).astype(int))
print('Confusion Matrix:\n', conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])
tn, fp, fn, tp = conf_matrix.ravel()  # 获取混淆矩阵的值
plt.text(0, 0, f'TN: {tn}', ha="center", va="center", color="black")
plt.text(1, 0, f'FP: {fp}', ha="center", va="center", color="black")
plt.text(0, 1, f'FN: {fn}', ha="center", va="center", color="black")
plt.text(1, 1, f'TP: {tp}', ha="center", va="center", color="black")
plt.show()

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"Area Under the ROC Curve (AUC): {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 计算召回率（在阈值0.965时）
threshold = 0.965
y_pred = (y_pred_proba > threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除以零

# 打印召回率
print(f"Recall at threshold {threshold}: {recall:.4f}")

# 获取模型中树的总数
n_estimators = best_model.n_estimators

# 特征顺序列表
feature_names = [
    "volumn_120d_sum", "S_HOLDER_PCT", "ctsdowndays", "ctsdownchg", "close_20d_mean",
    "close_20d_min", "net_income_margin", "min_net_profit", "gross_margin", "S_QFA_EPS",
    "fs_eps_yoy", "S_FA_ROA", "S_FA_ROE_DEDUCTED", "S_FA_ARTURN", "S_FA_ASSETSTURN",
    "S_FA_FATURN", "S_FA_YOY_TR", "S_QFA_CGRSALES", "S_QFA_YOYPROFIT", "yieldm",
    "yield6m", "turnratem", "turnrate6m", "S_VAL_MV", "S_VAL_PE", "S_VAL_PB_NEW",
    "S_VAL_PS", "volatility52w", "beta52w"
]

# 假设 best_model 是已经拟合好的 XGBoost 模型
# 获取模型的特征重要性

# 特征贡献值
feature_importances = best_model.feature_importances_

# 计算总贡献值
total_importance = sum(feature_importances)

# 计算每个特征贡献的百分比
feature_importances_percentage = [importance / total_importance for importance in feature_importances]

# 绘制条形图
plt.figure(figsize=(18, 6))  # 可以根据需要调整图形大小

# 绘制特征重要性的条形图
plt.barh(range(len(feature_importances)), feature_importances, color='skyblue', label='Feature Importances')

# 添加百分比标签
yticks_positions = range(len(feature_importances))
plt.yticks(yticks_positions, feature_names)

# 在每个条形上添加百分比标签
for index in yticks_positions:
    plt.text(feature_importances[index], index, f'{feature_importances_percentage[index]:.2%}',
             va='center', ha='right')

# 设置图表标题和标签
plt.xlabel('Total Importance')
plt.ylabel('Feature')
plt.title('Feature Importances with Percentage')

# 显示图例
plt.legend()

# 显示图表
plt.show()
# 设置图表标题和标签
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances using XGBoost')

# 显示图表
plt.show()

# 定义模型文件的完整路径
model_filename = r"C:\Users\ALEXZANDER\Desktop\Xgboost_St_Predict\模型建立\模型检验\draw.pkl"

# 保存模型
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f'Model saved as {model_filename}')

# 按照可能为正例的可能性大小降序输出预测的正例样本
sorted_indices = np.argsort(-y_pred_proba)  # 降序排序
sorted_y_pred_proba = y_pred_proba[sorted_indices]

# 获取对应的原始标签
sorted_y_val = y_val[sorted_indices]

# 打印排序后的正例样本及其预测概率
print("按照可能为正例的可能性大小降序输出预测的正例样本：")
for index, proba in zip(sorted_indices, sorted_y_pred_proba):
    if sorted_y_val[index] == 1:  # 只输出正例样本
        print(f"Index: {index}, Predicted Probability: {proba}")