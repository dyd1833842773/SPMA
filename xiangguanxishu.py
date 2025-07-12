import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 读取 Excel 文件
data = pd.read_excel('./sampling_results_60_1.xlsx')

# 提取第七列特征和倒数第二列标签
feature_seventh = data.columns[2]  # 第七列特征
label = data.columns[-4]           # 倒数第二列标签

# 计算皮尔森相关系数和 p 值
pearson_corr, pearson_p_value = pearsonr(data[feature_seventh], data[label])

# 计算斯皮尔曼相关系数和 p 值
spearman_corr, spearman_p_value = spearmanr(data[feature_seventh], data[label])

# 输出结果
print("皮尔森相关系数:", pearson_corr)
print("皮尔森相关系数的 p 值:", pearson_p_value)
print("斯皮尔曼相关系数:", spearman_corr)
print("斯皮尔曼相关系数的 p 值:", spearman_p_value) 
