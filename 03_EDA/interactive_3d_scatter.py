"""
Plotly 交互式 3D 散点图通用模板 (支持第四维连续/离散特征映射)
--------------------------------------------------
适用场景：
1. 探索三维空间/多维特征分布
2. 需要支持鼠标拖拽旋转、局部缩放、悬停查看数值的交互式数据分析
主要特性：
1. 参数与执行逻辑完全解耦，修改顶部配置即可复用
2. 包含安全抽样机制，避免数据量过大导致浏览器端渲染卡顿
3. 支持连续数值（色彩渐变）或离散分类（不同色块）的自动色彩映射
"""

import plotly.express as px
import pandas as pd

# ==========================================
# 第一部分：绘图参数配置 (TODO: 根据实际数据修改)
# ==========================================
# 假设 df 已经在上文通过 pd.read_csv() 等方式加载完毕

# 1. 字段映射配置
X_COL = 'feature_x'         # 映射到 X 轴的列名
Y_COL = 'feature_y'         # 映射到 Y 轴的列名
Z_COL = 'feature_z'         # 映射到 Z 轴的列名
COLOR_COL = 'target_value'  # 映射到颜色维度的列名 (可为数值型或分类型特征)

# 2. 文本与视觉配置
PLOT_TITLE = '交互式 3D 特征空间分布图'
COLOR_THEME = 'Viridis'     # 颜色主题，可选: 'Viridis', 'Plasma', 'Cividis', 'Turbo', 'Coolwarm'
POINT_SIZE = 3              # 散点像素大小 (3D 图中建议设置在 2~5 之间，避免相互遮挡)
POINT_OPACITY = 0.7         # 点的透明度 (0.0 ~ 1.0)

# 3. 性能与抽样配置
SAMPLE_SIZE = 3000          # 交互图建议控制在 2000~5000 之间保证帧率；若需全量渲染设为 None

# ==========================================
# 第二部分：数据准备与安全抽样
# ==========================================
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df_plot = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_plot = df.copy()

# ==========================================
# 第三部分：生成交互图表与渲染
# ==========================================
fig = px.scatter_3d(
    df_plot,
    x=X_COL,
    y=Y_COL,
    z=Z_COL,
    color=COLOR_COL,
    title=PLOT_TITLE,
    opacity=POINT_OPACITY,
    color_continuous_scale=COLOR_THEME
)

# 调整点尺寸和交互布局
fig.update_traces(marker=dict(size=POINT_SIZE))
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),  # 压缩边距，让 3D 主体更大
    scene=dict(
        xaxis_title=X_COL,
        yaxis_title=Y_COL,
        zaxis_title=Z_COL
    )
)

fig.show()
