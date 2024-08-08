import pandas as pd
import pickle
import numpy as np

# 步骤1: 加载模型
model_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Result\xgb_model.pkl"
model = pickle.load(open(model_path, 'rb'))

# 步骤2: 读取数据
data_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Data\Filtered_2024_4_预测.xlsx"
df = pd.read_excel(data_path)

# 步骤3: 排除最后一列进行预测
X = df.iloc[:, :-1]  # 特征数据，不包括最后一列
y = df.iloc[:, -1]   # 目标数据，最后一列

# 使用模型进行预测，获取预测概率
predictions_proba = model.predict_proba(X)[:, 1]

# 根据预测概率判断预测结果（0或1）
predictions = (predictions_proba >= 0.978).astype(int)

# 按照预测为正例的可能性大小降序排列索引
sorted_indices = np.argsort(-predictions_proba)

# 创建一个新的DataFrame来存储排序后的结果
sorted_df = pd.DataFrame({
    'Index': sorted_indices,
    'Predicted Probability': predictions_proba[sorted_indices],
    'Prediction': predictions[sorted_indices],
    'S_INFO_WINDCODE': y.reindex(sorted_indices).values
})

# 保存到Excel文件，指定文件路径和sheet名称
output_excel_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Result\Sorted_Predictions4.xlsx"
sorted_df.to_excel(output_excel_path, index=False, sheet_name='Sorted Predictions')

# 计算0和1的个数
count_0 = (predictions == 0).sum()
count_1 = (predictions == 1).sum()

# 打印0和1的个数
print(f"0的个数: {count_0}")
print(f"1的个数: {count_1}")

print(f"排序结果已保存到 '{output_excel_path}'")