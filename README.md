XGB_ST：
项目描述：基于Xgboost对St股票进行预测

1.1 XGBOOST预测ST股票.pptx：本项目的ppt解释说明

1.2 Model.py：用于训练模型的程序

1.3 Predict.py：用于预测的程序

1.4 Connect.py：用于将预测后结果和当月真实st事件做连接的程序

1.5 Calculate.py：用于计算预测出的正例下个月走势的程序

1.6 Data：存储原数据的文件夹
1.6.1 2024_4_True_St.xlsx：2024年四月真实可预测st事件
1.6.2 2024_5_True_St.xlsx：2024年五月真实可预测st事件
1.6.3 All-2024-3mbefore.xlsx：用来训练的数据
1.6.4 close1-4.csv：用来存储A股股票收盘价的文件
1.6.5 2023_2-2024_7.xlsx：用来预测的各月数据

1.7 Result
1.7.1 Sorted_Predictions23_2-24_7.xlsx：按概率排序的预测结果
1.7.2 Predict24_4-5.xlsx：仅含有当月真实的st事件及其概率s
1.7.3 Updated_Predictions_23_2-24_7.xlsx:预测为正例的下个月总收益率
1.7.4 xgb_model.pkl：训练出的模型