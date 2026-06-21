import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# ---------------------- 1. 数据加载 ----------------------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载所有数据
memory_data = load_data("实验数据/neicun.json")
up_data = load_data("实验数据/carts_up_data.json")
cpu_data = load_data("实验数据/cpu.json")
restart_data = load_data("实验数据/pod重启次数.json")

# ---------------------- 2. 数据预处理 ----------------------
# 提取Pod的时序指标
def extract_metrics(data, metric_name, pod_name):
    metrics = []
    timestamps = []
    for item in data['data']['result']:
        if item['metric']['pod'] == pod_name:
            for ts, val in item['values']:
                timestamps.append(int(ts))
                metrics.append(float(val))
    return pd.DataFrame({'timestamp': timestamps, metric_name: metrics}).sort_values('timestamp')

# 标准化函数
scaler = MinMaxScaler()
def normalize_series(series):
    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

# 提取核心Pod指标
pod_list = ["carts-b6c5c87f9-tht6g", "carts-b6c5c87f9-4bwfj", "carts-db-6bd47f898c-t7h4l"]
metrics_dict = {}
for pod in pod_list:
    # 内存指标
    mem_df = extract_metrics(memory_data, 'memory', pod)
    # CPU增量（15s间隔）
    cpu_df = extract_metrics(cpu_data, 'cpu', pod)
    cpu_df['cpu_increment'] = cpu_df['cpu'].diff().fillna(0)
    # up指标（部分Pod无）
    try:
        up_df = extract_metrics(up_data, 'up', pod)
        merged_df = pd.merge(mem_df, cpu_df, on='timestamp', how='outer').merge(up_df, on='timestamp', how='outer')
    except:
        merged_df = pd.merge(mem_df, cpu_df, on='timestamp', how='outer')
        merged_df['up'] = np.nan
    # 标准化
    merged_df['memory_norm'] = normalize_series(merged_df['memory'])
    merged_df['cpu_inc_norm'] = normalize_series(merged_df['cpu_increment'])
    metrics_dict[pod] = merged_df

# ---------------------- 3. MARINA异常评分 ----------------------
def marina_anomaly_score(df):
    # 单指标异常分数（偏离均值3σ判定为异常）
    def single_score(series):
        mean = series.mean()
        std = series.std()
        return np.where(np.abs(series - mean) > 3*std, 1.0, series / series.max())
    
    # 计算各指标分数
    df['mem_score'] = single_score(df['memory_norm'])
    df['cpu_score'] = single_score(df['cpu_inc_norm'])
    df['up_score'] = df['up'].apply(lambda x: 0.0 if x == 1 else 1.0 if x == 0 else 0.0)
    # 综合分数（加权平均）
    df['total_score'] = 0.4*df['mem_score'] + 0.4*df['cpu_score'] + 0.2*df['up_score']
    return df['total_score'].mean()

# 输出各Pod异常分数
for pod, df in metrics_dict.items():
    score = marina_anomaly_score(df)
    print(f"Pod {pod} 综合异常分数：{score:.2f}")

# ---------------------- 4. MARINA根因分析 ----------------------
def marina_root_cause(pod_df, pod_name):
    # 计算指标关联度
    if 'up' in pod_df.columns and not pd.isna(pod_df['up']).all():
        corr_mem_up = pearsonr(pod_df['memory_norm'].fillna(0), pod_df['up'].fillna(0))[0]
        corr_cpu_up = pearsonr(pod_df['cpu_inc_norm'].fillna(0), pod_df['up'].fillna(0))[0]
        print(f"\nPod {pod_name} 根因分析：")
        print(f"内存与up指标关联度：{corr_mem_up:.2f}")
        print(f"CPU增量与up指标关联度：{corr_cpu_up:.2f}")
        # 根因判定
        # 修改 marina.py 中的根因判定逻辑
        # 原代码：if corr_cpu_up < -0.8:
        # 修改为：
        if corr_cpu_up < -0.6:  # 将阈值从-0.8放宽到-0.6
            print("根因：CPU资源占用过高，存在服务不可用风险")
        elif corr_mem_up > 0.6:
            print("根因：内存使用率与服务可用性强相关，存在内存瓶颈风险")
        else:
            print("根因：无显著资源类故障")

# 执行根因分析
for pod, df in metrics_dict.items():
    marina_root_cause(df, pod)