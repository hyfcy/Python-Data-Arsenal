"""
多层感知机 (MLP) 回归标准模板 (支持多输出)
--------------------------------------------------
适用场景：通用的多特征回归、多目标回归任务
主要特性：
1. 包含完整的数据预处理管道（严防测试集数据泄露）
2. 原生支持多输出回归（Multi-output Regression）
3. 内置标准回归评估指标 (MSE, R-squared)
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 第一部分：数据加载与特征工程
# ==========================================

# [TODO: 需要修改] 1. 定义你的特征 (X) 和目标 (y) 列名
FEATURE_COLS = ['feature_1', 'feature_2', 'feature_3', 'feature_4'] 
TARGET_COLS = ['target_1', 'target_2']

# 提取数据 (假设 df 已经在上文准备完毕)
X = df[FEATURE_COLS]
y = df[TARGET_COLS]

# [TODO: 可选修改] 2. 划分数据集
# test_size=0.2 代表 80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # 固定随机种子以保证结果可复现
)

# 3. 特征标准化 (Standardization)
# 将数据缩放到均值为 0，方差为 1，加速神经网络收敛
scaler = StandardScaler()

# 警告：严禁在测试集上使用 fit()，防止未来数据泄露到训练阶段
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       

# ==========================================
# 第二部分：模型构建与训练
# ==========================================

# [TODO: 可选修改] 4. 调整神经网络超参数
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64), # 隐藏层架构：当前为两层，节点数分别为 128 和 64
    activation='relu',            # 非线性激活函数，回归任务首选
    solver='adam',                # 权重优化算法
    max_iter=500,                 # 最大训练迭代次数，如提示未收敛可适当调大此值
    random_state=42               # 保证每次初始化的权重一致
)

# 5. 训练模型
print("模型训练中，请稍候...")
mlp_model.fit(X_train_scaled, y_train)

# ==========================================
# 第三部分：模型预测与效果评估
# ==========================================

# 6. 在未见过的测试集上进行预测
y_pred_mlp = mlp_model.predict(X_test_scaled)

# 7. 计算评估指标
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("-" * 30)
print(f"模型评估结果:")
print(f"均方误差 (MSE): {mse_mlp:.4f} (越小越好)")
print(f"决定系数 (R-squared): {r2_mlp:.4f} (越接近 1 越理想)")
print("-" * 30)
