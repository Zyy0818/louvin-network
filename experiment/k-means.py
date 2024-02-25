import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表风格为亮色背景，带有网格线
sns.set(style="whitegrid")

# 加载数据
df = pd.read_csv('/Users/bytedance/Desktop/paper/louvin-network/sampled_output.csv')

# 数据标准化，不包括family_no
features = df[['count', 'requestCnt', 'flintType']]  # 使用选择的特征集
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

def apply_clustering(model, features_scaled, df, title_suffix):
    # 应用聚类
    clusters = model.fit_predict(features_scaled)
    df['cluster_label'] = clusters

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)

    # 将降维结果添加到数据框中
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # 可视化设置
    plt.figure(figsize=(16,10))

    # 为每个聚类和家族计算中心
    cluster_centers = df.groupby('cluster_label')[['tsne-2d-one', 'tsne-2d-two']].mean()

    # 绘制样本点
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+']
    for cluster in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == cluster]
        plt.scatter(cluster_data['tsne-2d-one'], cluster_data['tsne-2d-two'], 
                    label=f"Cluster {cluster}", alpha=0.7, 
                    c=[sns.color_palette("hsv", len(df['family_no'].unique()))[x] for x in cluster_data['family_no']], 
                    marker=markers[cluster % len(markers)])

    plt.title(f't-SNE Visualization with {title_suffix}', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(title='Cluster Label')
    plt.show()

# 应用不同的聚类算法
models = [
    (DBSCAN(eps=0.5, min_samples=5), "DBSCAN Clustering"),
    # (KMeans(n_clusters=5, random_state=42), "K-Means Clustering"),
    (AgglomerativeClustering(n_clusters=5), "Agglomerative Clustering")
]

for model, title in models:
    apply_clustering(model, features_scaled, df.copy(), title)
