import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from networkx.algorithms.centrality import degree_centrality
from sklearn.metrics import fowlkes_mallows_score

def evaluate_clustering_with_ground_truth(clusters, ground_truth):
    # 只有当有两个以上的群集时，才计算AMI和其他指标
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) > 1:
        ari = adjusted_rand_score(ground_truth, clusters)
        ami = adjusted_mutual_info_score(ground_truth, clusters)
        fmi=fowlkes_mallows_score(ground_truth, clusters)
        silhouette = silhouette_score(features_scaled, clusters)
        calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)
        davies_bouldin = davies_bouldin_score(features_scaled, clusters)
    else:
        # 如果只有一个群集，给指标赋予默认值
        ari, ami, silhouette, calinski_harabasz, davies_bouldin = (0, 0, 0, 0, 0)
    return ari, ami, fmi, silhouette, calinski_harabasz, davies_bouldin

# 载入数据
df = pd.read_csv('/Users/bytedance/Documents/GitHub/louvin-network/sampled_output_2.0k.csv')  # 请替换为您的数据集路径

# 真实的社区标签
ground_truth = df['family_no'].values

from sklearn.preprocessing import LabelEncoder, StandardScaler

# 假设 'encoded_fqdn' 和 'encoded_ip' 是需要进行编码的分类特征
# 使用 LabelEncoder 作为示例，根据实际情况选择编码方式
le_fqdn = LabelEncoder()
le_ip = LabelEncoder()

df['encoded_fqdn'] = le_fqdn.fit_transform(df['encoded_fqdn'])
df['encoded_ip'] = le_ip.fit_transform(df['encoded_ip'])

# 重新选择所有需要的特征
features1 = df[['requestCnt', 'flintType', 'encoded_fqdn']]
features2 = df[['requestCnt', 'flintType', 'encoded_fqdn','encoded_ip','encoded_fqdn']]

# 应用 StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features1)
features_scaled2 = scaler.fit_transform(features2)

# 现在 features_scaled 已经是数值类型，可以安全使用
df_normalized = pd.DataFrame(features_scaled, columns=['requestCnt', 'flintType', 'encoded_fqdn'])
df_normalized1 = pd.DataFrame(features_scaled2, columns=['requestCnt', 'flintType', 'encoded_fqdn','encoded_ip','encoded_fqdn'])


# 特征重要性顺序需要与 df_normalized 中的列顺序一致
feature_importances = np.array([ 
    0.04551161825656891, 0.5758007168769836, 0.20780739188194275])

# 更新 improved_weight 函数
def improved_weight(vector1, vector2, sigma = 0.02, feature_importances = feature_importances):
    weighted_diff = (vector1 - vector2) * feature_importances[:len(vector1)]  # 确保特征重要性数组与向量长度相匹配
    distance = np.linalg.norm(weighted_diff)
    weight = np.exp(-distance / sigma)
    return weight

# 构建网络并计算模块度的函数
def build_network_and_calculate_modularity(df_normalized, weight_function, sigma, feature_importances):
    G = nx.Graph()
    n_domains = len(df_normalized)
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            vector1 = df_normalized.iloc[i].values
            vector2 = df_normalized.iloc[j].values
            weight = weight_function(vector1, vector2, sigma, feature_importances)
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    return G

# 计算加权边的函数
def calculate_weighted_edges(G, df_normalized, weight_function, sigma, feature_importances, centrality_weight=0.5):
    centrality = degree_centrality(G)
    for i, j in G.edges():
        original_weight = G[i][j]['weight']
        adjusted_weight = original_weight * ((centrality[i] + centrality[j]) / 2 * centrality_weight + (1 - centrality_weight))
        G[i][j]['weight'] = adjusted_weight

# 带有优化的Louvain聚类函数
def louvain_clustering_with_optimizations(df_normalized, sigma=0.02, feature_importances=feature_importances, centrality_weight=0.5):
    G = build_network_and_calculate_modularity(df_normalized, improved_weight, sigma, feature_importances)
    calculate_weighted_edges(G, df_normalized, improved_weight, sigma, feature_importances, centrality_weight)
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    clusters = np.zeros(len(df_normalized))
    for node, cluster_id in partition.items():
        clusters[node] = cluster_id
    return clusters, partition

# 设置搜索区间和间隔
# sigma_values = np.arange(0.75, 1, 0.01)
# centrality_weight_values = np.arange(0, 1.05, 0.05)
# centrality_weight_values = 0.3





# 初始化最佳分数和参数
# best_ami = -1
# best_fmi = -1

# best_sigma = None
# best_centrality_weight = None
# best_sigma = 0.8
# best_centrality_weight = 0.3
# best_clusters = None

# 双重循环遍历所有sigma和centrality_weight的组合
# for sigma in sigma_values:
#     if sigma == 0:  # 防止除以零的情况
#         continue
#     # for centrality_weight in centrality_weight_values:
#     clusters, _ = louvain_clustering_with_optimizations(df_normalized, sigma, centrality_weight_values)
#     # 仅计算AMI，跳过无效的聚类结果
#     _, ami, fmi, _, _, _ = evaluate_clustering_with_ground_truth(clusters, ground_truth)
#     if ami != 0:
#         # 打印每一组参数的AMI
#         print(f"sigma={sigma:.2f}, centrality_weight={centrality_weight_values:.2f}, AMI={ami:.4f}, FMI={fmi:.4f}")
    
#         # 更新最佳分数和参数
#         if ami > best_ami:
#             best_ami = ami
#             best_sigma = sigma
#             best_centrality_weight = centrality_weight_values
#         if fmi > best_fmi:
#             best_fmi = fmi
#             best_sigma = sigma
#             best_centrality_weight = centrality_weight_values

# # 根据原始格式打印最佳AMI
# print(f"Best Louvain Clustering: AMI={best_ami}, FMI={best_fmi}")


# 迭代 Louvain 算法，尝试不同的 sigma 值
sigmas = [0.01, 0.02, 0.05]
best_ami = -1
best_sigma = 0.9
best_centrality_weight = 0.6
best_clusters = None
# for sigma in sigmas:
#     clusters, _ = louvain_clustering_with_optimizations(df_normalized, sigma,best_centrality_weight)
#     _, ami, _, _, _ = evaluate_clustering_with_ground_truth(clusters, ground_truth,best_centrality_weight)
#     if ami > best_ami:
#         best_ami = ami
#         best_sigma = sigma
#         best_clusters = clusters

best_clusters, _ = louvain_clustering_with_optimizations(df_normalized, sigma=best_sigma, feature_importances=feature_importances, centrality_weight=best_centrality_weight)

evaluate_louvain_best = evaluate_clustering_with_ground_truth(best_clusters, ground_truth)

def dbscan_clustering(features_scaled):
    model = DBSCAN(eps=0.5, min_samples=5)
    clusters = model.fit_predict(features_scaled)
    return clusters

def agglomerative_clustering(features_scaled):
    model = AgglomerativeClustering(n_clusters=5)
    clusters = model.fit_predict(features_scaled)
    return clusters

def evaluate_clustering(clusters, features_scaled):
    silhouette = silhouette_score(features_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)
    davies_bouldin = davies_bouldin_score(features_scaled, clusters)
    return silhouette, calinski_harabasz, davies_bouldin

def visualize_clustering(df, clusters, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)
    df['Cluster'] = clusters
    df['TSNE-1'] = tsne_results[:,0]
    df['TSNE-2'] = tsne_results[:,1]
    sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='Cluster', data=df, palette='viridis')
    plt.title(title)
    plt.show()
    

# 应用聚类算法并评估
# 应用聚类算法并评估
evaluate_louvain_best = evaluate_clustering_with_ground_truth(best_clusters, ground_truth)

dbscan_clusters = dbscan_clustering(features_scaled2)
evaluate_dbscan = evaluate_clustering_with_ground_truth(dbscan_clusters, ground_truth)

agglomerative_clusters = agglomerative_clustering(features_scaled2)
evaluate_agglomerative = evaluate_clustering_with_ground_truth(agglomerative_clusters, ground_truth)

# 可视化聚类结果
# visualize_clustering(df, louvain_clusters, "Louvain Clustering")
# visualize_clustering(df, dbscan_clusters, "DBSCAN Clustering")
# visualize_clustering(df, agglomerative_clusters, "Agglomerative Clustering")

# 打印聚类评估结果，包括ARI和AMI
print("Clustering Evaluation Metrics:")
print(f"Louvain Clustering: ARI={evaluate_louvain_best[0]}, AMI={evaluate_louvain_best[1]},FMI={evaluate_louvain_best[2]}, Silhouette={evaluate_louvain_best[3]}, Calinski-Harabasz={evaluate_louvain_best[4]}, Davies-Bouldin={evaluate_louvain_best[5]}")
print(f"DBSCAN Clustering: ARI={evaluate_dbscan[0]}, AMI={evaluate_dbscan[1]}, FMI={evaluate_dbscan[2]},Silhouette={evaluate_dbscan[3]}, Calinski-Harabasz={evaluate_dbscan[4]}, Davies-Bouldin={evaluate_dbscan[5]}")
print(f"Agglomerative Clustering: ARI={evaluate_agglomerative[0]}, AMI={evaluate_agglomerative[1]}, FMI={evaluate_agglomerative[2]},Silhouette={evaluate_agglomerative[3]}, Calinski-Harabasz={evaluate_agglomerative[4]}, Davies-Bouldin={evaluate_agglomerative[5]}")