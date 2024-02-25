import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import pearsonr

# 生成模拟数据集
np.random.seed()
n_domains = 100
visits = np.random.randint(100, 1000, size=n_domains)
types = np.random.randint(0, 2, size=n_domains)
locations = np.random.randint(0, 50, size=n_domains)
df_domains = pd.DataFrame({'visits': visits, 'types': types, 'locations': locations})
df_normalized = (df_domains - df_domains.min()) / (df_domains.max() - df_domains.min())

# 改进的权重计算方法
def improved_weight(vector1, vector2):
    correlation = np.corrcoef(vector1, vector2)[0, 1]
    # 调整负值
    adjusted_weight = (correlation + 1) / 2
    return adjusted_weight

# 原始权重计算方法
def original_weight(vector1, vector2):
    # 计算欧氏距离
    distance = np.linalg.norm(vector1 - vector2)
    # 使用指数函数转换距离为权重，这里的σ是自定义的尺度参数，用于调整权重分布
    sigma = 0.1  # σ值根据数据的具体情况进行调整
    weight = np.exp(-distance / sigma)
    return weight

# 构建网络并计算模块度的函数
def build_network_and_calculate_modularity(df_normalized, weight_function):
    G = nx.Graph()
    for i in range(n_domains):
        for j in range(i+1, n_domains):
            weight = weight_function(df_normalized.iloc[i], df_normalized.iloc[j])
            if weight > 0:  # 添加正权重的边
                G.add_edge(i, j, weight=weight)
    
    partition = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(partition, G, weight='weight')
    return modularity

# 执行实验并收集结果
n_runs = 1
results = []

for run in range(1, n_runs + 1):
    mod_original = build_network_and_calculate_modularity(df_normalized, original_weight)
    mod_improved = build_network_and_calculate_modularity(df_normalized, improved_weight)
    results.append((run, mod_original, mod_improved))

# 将结果转换为DataFrame并打印
results_df = pd.DataFrame(results, columns=['Run', 'Modularity (Original)', 'Modularity (Improved)'])
print(results_df)
