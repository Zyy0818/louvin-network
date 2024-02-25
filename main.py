import pandas as pd
from pycaret.clustering import *


# 载入数据集
file_path = '2023-12-30 16:51:11.csv'
data = pd.read_csv(file_path)

# 抽样，例如抽取10%的数据
sampled_data = data.sample(frac=0.005, random_state=123)

# 设置聚类环境
clu_setup = setup(sampled_data, session_id=123, log_experiment=True, ignore_features=['family_no'])

# 创建并比较聚类模型
# kmeans = create_model('kmeans')
# hclust = create_model('hclust')
# dbscan = create_model('dbscan')
# birch = create_model('birch')
# mean_shift = create_model('meanshift')
affinity_propagation = create_model('ap')

# 生成聚类图
# plot_model(kmeans, plot='cluster')
# plot_model(hclust, plot='cluster')
# plot_model(dbscan, plot='cluster')
# plot_model(birch, plot='cluster')
# plot_model(mean_shift, plot='cluster')
plot_model(affinity_propagation, plot='cluster')