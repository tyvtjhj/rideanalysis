import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from workout_parser import load_fit_file

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据提取 ====================

def extract_records(fit_path: str) -> pd.DataFrame:
    fit_file = load_fit_file(fit_path)
    records = []
    for message in fit_file.get_messages('record'):
        record_dict = {}
        for field in message.fields:
            record_dict[field.name] = field.value
        records.append(record_dict)
    return pd.DataFrame(records)


def extract_laps(fit_path: str) -> pd.DataFrame:
    fit_file = load_fit_file(fit_path)
    laps = []
    for message in fit_file.get_messages('lap'):
        lap_dict = {}
        for field in message.fields:
            lap_dict[field.name] = field.value
        laps.append(lap_dict)
    return pd.DataFrame(laps)


def extract_session(fit_path: str) -> dict:
    fit_file = load_fit_file(fit_path)
    session_data = {}
    for message in fit_file.get_messages('session'):
        for field in message.fields:
            session_data[field.name] = field.value
        break
    return session_data


# ==================== 功率分区 ====================

def classify_power_zone(power: float) -> str:
    if power < 100:
        return '\u6062\u590d\u533a'
    elif power < 200:
        return '\u8010\u529b\u533a'
    elif power < 300:
        return '\u8282\u594f\u533a'
    elif power < 400:
        return '\u9608\u503c\u533a'
    else:
        return '\u51b2\u523a\u533a'


def add_power_zone_fixed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['power_zone'] = df['power'].apply(classify_power_zone)
    return df


def add_power_bin_percentile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    percentiles = df['power'].quantile([0.25, 0.5, 0.75, 0.9])
    df['power_bin'] = pd.cut(
        df['power'],
        bins=[0, percentiles[0.25], percentiles[0.5],
              percentiles[0.75], percentiles[0.9], float('inf')],
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )
    return df


def add_rolling_power_zone(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    df = df.copy()
    df['rolling_avg_power'] = df['power'].rolling(window=window_size).mean()
    df['window_zone'] = pd.cut(
        df['rolling_avg_power'],
        bins=[0, 150, 250, 350, float('inf')],
        labels=['\u4f4e\u529f\u7387', '\u4e2d\u529f\u7387', '\u9ad8\u529f\u7387', '\u6781\u9ad8\u529f\u7387']
    )
    return df


# ==================== 功率变化检测 ====================

def detect_power_changes(df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
    df = df.copy()
    df['power_diff'] = df['power'].diff().abs()
    df['is_activity_change'] = df['power_diff'] > threshold
    df['activity_segment'] = df['is_activity_change'].cumsum()
    return df


# ==================== 自动分段 ====================

def segment_by_power_change(
    df: pd.DataFrame,
    threshold: float = 50,
    min_segment_length: int = 30
) -> pd.DataFrame:
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    df_sorted['power_diff'] = df_sorted['power'].diff().abs()
    df_sorted['is_segment_start'] = df_sorted['power_diff'] > threshold
    df_sorted.loc[0, 'is_segment_start'] = True
    df_sorted['segment_id'] = df_sorted['is_segment_start'].cumsum()

    segment_lengths = df_sorted.groupby('segment_id').size()
    short_segments = segment_lengths[segment_lengths < min_segment_length].index

    for seg_id in short_segments:
        if seg_id > 1:
            df_sorted.loc[df_sorted['segment_id'] == seg_id, 'segment_id'] = seg_id - 1

    df_sorted['segment_id'] = df_sorted['segment_id'].rank(method='dense').astype(int)
    return df_sorted


def segment_by_window_variance(
    df: pd.DataFrame,
    window_size: int = 60,
    threshold_ratio: float = 2
) -> pd.DataFrame:
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    df_sorted['power_var'] = df_sorted['power'].rolling(window=window_size).var()
    df_sorted['var_diff'] = df_sorted['power_var'].pct_change(fill_method=None).abs()
    df_sorted['is_segment_start'] = df_sorted['var_diff'] > threshold_ratio
    df_sorted.loc[0, 'is_segment_start'] = True
    df_sorted['segment_id'] = df_sorted['is_segment_start'].cumsum()
    return df_sorted


# ==================== 可视化 ====================

def plot_power_curve(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df['power'], color='red', linewidth=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power vs Time')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_heart_rate(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df['heart_rate'], color='blue', linewidth=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Heart Rate (bpm)')
    ax.set_title('Heart Rate vs Time')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_dual_axis(df: pd.DataFrame, save_path: str = None):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (W)', color='red')
    ax1.plot(df['timestamp'], df['power'], color='red', linewidth=0.8)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Heart Rate (bpm)', color='blue')
    ax2.plot(df['timestamp'], df['heart_rate'], color='blue', linewidth=0.8)
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.autofmt_xdate(rotation=45)
    plt.title('Power and Heart Rate vs Time')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_segmented_power(df_segmented: pd.DataFrame, max_segments: int = 10, save_path: str = None):
    fig, ax = plt.subplots(figsize=(15, 6))

    for seg_id in df_segmented['segment_id'].unique()[:max_segments]:
        seg_data = df_segmented[df_segmented['segment_id'] == seg_id]
        ax.plot(seg_data['timestamp'], seg_data['power'],
                label=f'Segment {seg_id}', linewidth=1.5)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Power Segmentation by Change Detection', fontsize=14, pad=20)
    fig.autofmt_xdate(rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    fig.tight_layout(pad=3.0)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_all_segments(df_segmented: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(15, 8))

    for seg_id in df_segmented['segment_id'].unique():
        seg_data = df_segmented[df_segmented['segment_id'] == seg_id]
        ax.plot(seg_data['timestamp'], seg_data['power'],
                label=f'Segment {seg_id}', linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Segmentation by Change Detection (All Segments)')
    fig.autofmt_xdate(rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ==================== 打印函数 ====================

def print_record_summary(df: pd.DataFrame):
    print("\n=== Record 数据摘要 ===")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    if 'power' in df.columns:
        print(f"平均功率: {df['power'].mean():.1f} W")
    if 'heart_rate' in df.columns:
        print(f"平均心率: {df['heart_rate'].mean():.1f} bpm")
    if 'cadence' in df.columns:
        print(f"平均踏频: {df['cadence'].mean():.1f} rpm")


def print_lap_summary(lap_df: pd.DataFrame):
    print("\n=== Lap 数据摘要 ===")
    print(f"Lap 数量: {len(lap_df)}")
    display_cols = [c for c in [
        'message_index', 'start_time', 'total_timer_time',
        'total_distance', 'avg_power', 'max_power',
        'avg_heart_rate', 'max_heart_rate'
    ] if c in lap_df.columns]
    if display_cols:
        print(lap_df[display_cols].head(10))


def print_session_summary(session_data: dict):
    print("\n=== Session 数据摘要 ===")
    if 'sport' in session_data:
        print(f"运动类型: {session_data['sport']}")
    if 'total_timer_time' in session_data:
        print(f"总时长: {session_data['total_timer_time'] / 60:.1f} 分钟")
    if 'total_distance' in session_data:
        print(f"总距离: {session_data['total_distance'] / 1000:.2f} km")
    if 'avg_power' in session_data:
        print(f"平均功率: {session_data['avg_power']} W")
    if 'max_heart_rate' in session_data:
        print(f"最大心率: {session_data['max_heart_rate']} bpm")


def print_segment_stats(df_segmented: pd.DataFrame):
    print("\n=== 各分段统计 ===")
    agg_dict = {
        'timestamp': ['min', 'max'],
        'power': ['mean', 'min', 'max', 'std']
    }
    if 'heart_rate' in df_segmented.columns:
        agg_dict['heart_rate'] = ['mean', 'min', 'max']
    segment_stats = df_segmented.groupby('segment_id').agg(agg_dict)
    print(segment_stats)
