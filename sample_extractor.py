import pandas as pd

def sample_csv(input_file: str, output_file: str, sample_fraction: float = 0.005):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 抽取指定比例的样本
    sample_df = df.sample(frac=sample_fraction)
    
    # 保存样本到新的CSV文件
    sample_df.to_csv(output_file, index=False)

float = 0.005
# 调用函数
# 这里的'input.csv'是您的原始CSV文件路径，'sampled_output.csv'是您想要保存抽样后数据的新CSV文件路径
sample_csv('2023-12-30 16:51:11.csv', 'sampled_output_'+str(float*1000)+'k.csv',float)
