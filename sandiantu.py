
import pandas as pd
import matplotlib.pyplot as plt


# 读取 Excel 文件
data = pd.read_excel('./result/rotate/sampling_results.xlsx')

# 提取特征和标签
features = data.columns[:-4]  # 除了最后4列，其他都是特征
label = data.columns[-4]       # MRR

# 对于每个特征，绘制单独的图形
for feature in features:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[feature], data[label], s=10)  # 设置点的大小为10
    ax.set_title(f'rotate:{feature} vs {label}', fontsize=12)  # 标题
    ax.set_xlabel(feature, fontsize=10)  # x轴标签
    ax.set_ylabel(label, fontsize=10)    # y轴标签
    ax.tick_params(axis='both', which='major', labelsize=8)  # 坐标轴刻度字体
    plt.tight_layout(pad=1.0)
    
    # 若想保存图片，可以使用下面这行代码（取消注释即可）
    plt.savefig(f'eccel/rotate/rotate:{feature}_vs_{label}.svg', format='svg')
    
    plt.show()
