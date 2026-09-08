# IQR四分位距法，箱型图底层的数学逻辑，不要求数据据服从正态分布
# 将 '目标列名' 替换为你实际需要检测的列
target_col = '目标列名'


# 1. 计算 Q1, Q3 和 IQR
Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1

# 2. 设定上下限警戒线
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 3. 揪出所有越界的异常行
outliers = df.loc[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
print(outliers)
