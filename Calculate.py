import pandas as pd

# 定义文件路径
predictions_path = r"C:\Users\ALEXZANDER\Desktop\Xgboost_St_Predict\模型建立\实用性分析\Sorted_Predictions7.xlsx"
close_path = r"C:\Users\ALEXZANDER\Desktop\Xgboost_St_Predict\模型建立\实用性分析\close.csv"
output_path = r"C:\Users\ALEXZANDER\Desktop\Xgboost_St_Predict\模型建立\实用性分析\Updated_Predictions7.xlsx"

# 读取预测结果
predictions_df = pd.read_excel(predictions_path)

# 筛选Predicted Probability大于0.978的行
high_confidence_df = predictions_df[predictions_df['Predicted Probability'] > 0.978]

# high_confidence_df['rate']='N'
# 读取close.csv文件
close_df = pd.read_csv(close_path)
df = pd.read_csv(r"C:\Users\ALEXZANDER\Desktop\Xgboost_St_Predict\模型建立\实用性分析\close1.csv")

# 使用concat函数合并两个DataFrame
close_df = pd.concat([close_df, df], axis=0, ignore_index=True)

# 筛选TRADE_DT为20240401和20240430的行
close_20240401_df = close_df[close_df['TRADE_DT'] == 20240701]
close_20240430_df = close_df[close_df['TRADE_DT'] == 20240730]

# 根据S_INFO_WINDCODE合并两个DataFrame
merged_df = close_20240401_df.merge(
    close_20240430_df,
    on='S_INFO_WINDCODE',  # 根据S_INFO_WINDCODE合并
    suffixes=('_20240701', '_20240730')  # 为相同列名添加后缀以区分
)

merged_df['rate'] = merged_df['S_DQ_CLOSE_20240730'] / merged_df['S_DQ_CLOSE_20240701'] - 1

high_confidence_df = high_confidence_df.merge(
    merged_df[['S_INFO_WINDCODE', 'rate']],  # 只合并需要的列
    on='S_INFO_WINDCODE',  # 根据S_INFO_WINDCODE合并
    how='left'  # 使用left join以保留所有high_confidence_df中的行
)

# 将结果写入新的Excel文件
high_confidence_df.to_excel(output_path, index=False)

print(f"更新后的结果已保存到 '{output_path}'")