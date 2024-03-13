import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 加载数据
data_path = '/Users/bytedance/Documents/GitHub/louvin-network/real_environment.csv'  # 请替换为您的文件路径
data = pd.read_csv(data_path)

# 数据预处理
# 删除不需要的列
data = data.drop(columns=['Unnamed: 0'])

# 对分类特征进行标签编码
categorical_features = ['encoded_ip', 'encoded_value', 'encoded_fqdn']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 分割数据为特征集和目标变量
X = data.drop(columns=['family_no'])
y = data['family_no']

# 对目标变量进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 分割为训练集和测试集
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 使用XGBoost训练模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train_encoded)

# 获取特征重要性
feature_importance = model.feature_importances_
feature_importance_dict = dict(zip(X.columns, feature_importance))

# 按重要性排序并展示
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
for feature, importance in sorted_feature_importance:
    print(f'{feature}: {importance}')

# 根据需要，您可以保存模型或进一步分析特征重要性
