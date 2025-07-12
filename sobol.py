import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt



def draw_sandian(features_t,S1,ST,label):
    
    # 生成条形图
    x = np.arange(len(features_t))  # x轴位置
    width = 0.35  # 条形宽度

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, S1, width, label='S1 (First-order)')
    bars2 = ax.bar(x + width/2, ST, width, label='ST (Total-order)')

    # 添加标签和标题
    ax.set_xlabel('Features')
    ax.set_ylabel('Sensitivity Index')
    ax.set_title(f'complex:Sobol Sensitivity Analysis Results({label})')
    ax.set_xticks(x)
    ax.set_xticklabels(features_t, rotation=45, ha="right")
    ax.legend()

    # 显示图形
    plt.tight_layout()
    #plt.savefig(f'eccel/complex/complex_sobol.svg', format='svg')  # 保存为SVG文件

    plt.show()

def sobol_mrr():
    # 读取数据
    # file_path = r'sampling_results.xlsx'
    file_path = './result/complex/sampling_results_60.xlsx'
    data = pd.read_excel(file_path)

    # 特征和目标变量
    features = data.columns[:-4]
    print(features)
    X = data[features].values
    Y_mrr = data['MRR'].values


    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合模型
    # model = RandomForestRegressor(n_estimators=100, random_state=0) #随机森林回归器
    # model = LinearRegression() #线性回归
    # 多项式回归模型
    degree = 3  #多项式阶数
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_scaled, Y_mrr)

    #定义 Sobol 问题
    problem = {
        'num_vars': X_scaled.shape[1],
        'names': features,
        'bounds': [[0, 1]] * X_scaled.shape[1]
    }


    # 生成采样点
    param_values = saltelli.sample(problem, 2048)

    # 预测采样点 MRR
    Y_mrr_pred = model.predict(param_values)

    # Sobol 敏感性分析
    sobol_indices = sobol.analyze(problem, Y_mrr_pred)

    # 输出结果
    print("Sobol Sensitivity Analysis Results:")
    print("S1:", sobol_indices['S1'])
    print("ST:", sobol_indices['ST'])
    print("S2:", sobol_indices['S2'])

    
    features_t = ['Degree distribution index','Relationship categories distribution index','Relationship types distribution index','Graph Density',
                 'Global_clustering_coefficient',   'Strongly_connected_components']
    S1,ST = sobol_indices['S1'], sobol_indices['ST']

    draw_sandian(features_t,S1,ST,label = 'MRR')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
import math

# 设置中文字体，确保标题和坐标轴标签能正确显示中文
rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei'

def draw_sobol_second_order_heatmap(s2_dict, X_vars, Y_vars):
  
    for idx, Y_var in enumerate(Y_vars):
        # 获取 Y_var 对应的二阶敏感度指数矩阵
        s2_matrix = np.array(s2_dict[Y_var]).reshape(len(X_vars), len(X_vars))
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(s2_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='sobol二阶指数')
        plt.xticks(np.arange(len(X_vars)), X_vars, rotation=45)
        plt.yticks(np.arange(len(X_vars)), X_vars)
        plt.title(f'RotatE:{Y_var}  结构特征交互作用热力图')
        plt.tight_layout()
        plt.savefig(f'./fig/rotate/rotate_sobol2_z.svg', format='svg')  # 保存为SVG文件
        plt.show()

def sobol_analysis_and_plot():
    # 读取数据
    file_path = './result/rotate/sampling_results_z.xlsx'
    data = pd.read_excel(file_path)

    # 特征和目标变量
    features = data.columns[:-4]
    X = data[features].values
    Y_mrr = data['MRR'].values


    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合模型（可以根据需要调整）
    degree = 3  # 多项式阶数
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_scaled, Y_mrr)

    # 定义 Sobol 问题
    problem = {
        'num_vars': X_scaled.shape[1],
        'names': features,
        'bounds': [[0, 1]] * X_scaled.shape[1]
    }

    # 生成采样点
    param_values = saltelli.sample(problem, 2048)

    # 预测采样点
    Y_mrr_pred = model.predict(param_values)

    # Sobol 敏感性分析
    sobol_indices = sobol.analyze(problem, Y_mrr_pred)

    # 输出 Sobol 结果
    print("Sobol Sensitivity Analysis Results for MRR:")
    print("S1:", sobol_indices['S1'])
    print("ST:", sobol_indices['ST'])
    print("S2:", sobol_indices['S2'])

    # 提取二阶敏感度指数（S2）
    s2_dict = {'MRR': sobol_indices['S2']}

    # 绘制二阶敏感度指数的热力图
    draw_sobol_second_order_heatmap(s2_dict, features, ['MRR'])

 

if __name__ == '__main__':
    np.random.seed(42)
    
    sobol_mrr()

    sobol_analysis_and_plot()