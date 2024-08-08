import pandas as pd

# 定义Excel文件路径
sorted_predictions_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Result\Sorted_Predictions4.xlsx"
data_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Data\4月真实st.xlsx"
output_path = r"C:\Users\ALEXZANDER\Desktop\XGB_ST\Result\predict4.xlsx"

# 读取排序后的预测结果
sorted_df = pd.read_excel(sorted_predictions_path)

# 读取原始数据
data_df = pd.read_excel(data_path)

# 进行自然连接，基于S_INFO_WINDCODE列
# 假设S_INFO_WINDCODE列在两个DataFrame中都存在
merged_df = pd.merge(sorted_df, data_df, on='S_INFO_WINDCODE', how='inner')

# 保存到Excel文件
merged_df.to_excel(output_path, index=False)

print(f"自然连接结果已保存到 '{output_path}'")
