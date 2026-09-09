"""
随机森林 (Random Forest) 回归标准模板 (支持多输出)
--------------------------------------------------
适用场景：通用的多特征非线性回归、多目标预测任务
主要特性：
1. 原生支持多输出回归（Multi-output Regression）
2. 树模型无需对输入特征进行标准化（Standardization/Normalization）缩放
3. 包含多核并行计算配置（n_jobs=-1）加速训练
4. 内置标准回归评估指标 (MSE, R-squared)
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 第一部分：数据加载与特征定义
# ==========================================

# [TODO: 需要修改] 1. 定义特征 (X) 和目标 (y) 列名
FEATURE_COLS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLS = ['target_1', 'target_2']

# 提取特征矩阵与目标矩阵 (假设 df 已经在上文加载完毕)
X = df[FEATURE_COLS]
y = df[TARGET_COLS]

# [TODO: 可选修改] 2. 划分数据集
# test_size=0.2 代表 80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # 固定随机种子以保证结果可复现
)

# ==========================================
# 第二部分：模型构建与训练
# ==========================================

# [TODO: 可选修改] 3. 实例化随机森林回归器
# 树模型基于分段阈值分裂，天然不受量纲影响，无需使用 StandardScaler
rf_model = RandomForestRegressor(
    n_estimators=100,  # 决策树的数量，数值越大越稳定但耗时更长
    max_depth=None,    # 树的最大深度，防止过拟合可设为整数（如 10, 20）
    n_jobs=-1,         # 开启所有 CPU 核心并行加速训练
    random_state=42    # 保证每次森林构建结果一致
)

# 4. 训练模型
print("随机森林模型训练中，请稍候...")
rf_model.fit(X_train, y_train)

# ==========================================
# 第三部分：模型预测与评估
# ==========================================

# 5. 在未见过的测试集上进行预测
y_pred_rf = rf_model.predict(X_test)

# 6. 计算回归评估指标
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("-" * 30)
print("模型评估结果:")
print(f"均方误差 (MSE): {mse_rf:.4f} (越小越好)")
print(f"决定系数 (R-squared): {r2_rf:.4f} (越接近 1 越好)")
print("-" * 30)
