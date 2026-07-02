from pathlib import Path
import fitparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INTENSITY_COLORS = {
    'active': '#e74c3c',
    'rest': '#3498db',
    'cooldown': '#2ecc71',
}
INTENSITY_LABELS = {
    'active': '\u6d3b\u52a8',
    'rest': '\u4f11\u606f',
    'cooldown': '\u51b7\u8eab',
}


def load_fit_file(fit_path: str) -> fitparse.FitFile:
    path = Path(fit_path)
    if not path.exists():
        raise FileNotFoundError(f"FIT 文件不存在: {fit_path}")
    return fitparse.FitFile(str(path))


def get_fit_message_types(fit_path: str) -> set:
    fit_file = load_fit_file(fit_path)
    message_types = set()
    for message in fit_file.get_messages():
        message_types.add(message.name)
    return message_types


def extract_workout_steps(fit_path: str) -> pd.DataFrame:
    fit_file = load_fit_file(fit_path)
    workout_steps = []
    for message in fit_file.get_messages('workout_step'):
        step = {}
        for field in message.fields:
            if field.value is not None and not field.name.startswith('unknown'):
                step[field.name] = field.value
        workout_steps.append(step)
    return pd.DataFrame(workout_steps)


def print_workout_steps(df: pd.DataFrame):
    print("\n=== 训练步骤 (Workout Steps) ===")
    print(f"步骤数量: {len(df)}")
    display_cols = [c for c in [
        'message_index', 'wkt_step_name', 'duration_time',
        'target_type', 'target_power_zone',
        'custom_target_power_low', 'custom_target_power_high',
        'intensity', 'duration_type'
    ] if c in df.columns]
    print(df[display_cols])


def plot_training_power_targets(df: pd.DataFrame, save_path: str = None):
    valid = df[df['duration_time'].notna()].copy()
    valid['step_label'] = valid.apply(
        lambda r: r['wkt_step_name'] if pd.notna(r['wkt_step_name']) else f"WarmUp {int(r['message_index']) + 1}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    y_pos = range(len(valid))

    for i, (_, row) in enumerate(valid.iterrows()):
        low = row['custom_target_power_low']
        high = row['custom_target_power_high']
        intensity = row['intensity'] if pd.notna(row['intensity']) else 'active'
        color = INTENSITY_COLORS.get(intensity, '#95a5a6')
        ax.barh(i, high - low, left=low, height=0.6, color=color, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['step_label'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Target Power (W)', fontsize=12)
    ax.set_title('Training Plan - Target Power Range per Step', fontsize=14, pad=15)

    legend_patches = [
        mpatches.Patch(color=color, label=INTENSITY_LABELS.get(key, key))
        for key, color in INTENSITY_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_training_duration(df: pd.DataFrame, save_path: str = None):
    valid = df[df['duration_time'].notna()].copy()
    valid['step_label'] = valid.apply(
        lambda r: r['wkt_step_name'] if pd.notna(r['wkt_step_name']) else f"WarmUp {int(r['message_index']) + 1}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    y_pos = range(len(valid))

    for i, (_, row) in enumerate(valid.iterrows()):
        intensity = row['intensity'] if pd.notna(row['intensity']) else 'active'
        color = INTENSITY_COLORS.get(intensity, '#95a5a6')
        ax.barh(i, row['duration_time'] / 60, height=0.6, color=color, edgecolor='white', linewidth=0.5)

    total_duration = valid['duration_time'].sum() / 60
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['step_label'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Duration (min)', fontsize=12)
    ax.set_title(f'Training Plan - Duration per Step (Total: {total_duration:.0f} min)', fontsize=14, pad=15)

    legend_patches = [
        mpatches.Patch(color=color, label=INTENSITY_LABELS.get(key, key))
        for key, color in INTENSITY_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_training_summary(df: pd.DataFrame, save_path: str = None):
    valid = df[df['duration_time'].notna()].copy()
    intensity_duration = valid.groupby('intensity')['duration_time'].sum() / 60

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors_pie = [INTENSITY_COLORS.get(k, '#95a5a6') for k in intensity_duration.index]
    labels_pie = [INTENSITY_LABELS.get(k, k) for k in intensity_duration.index]
    wedges, texts, autotexts = ax1.pie(
        intensity_duration.values, labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, explode=[0.02] * len(intensity_duration)
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax1.set_title('Time Distribution by Intensity', fontsize=13)

    step_types = valid['wkt_step_name'].fillna('Warm Up').value_counts()
    colors_bar = [
        INTENSITY_COLORS.get(
            valid[valid['wkt_step_name'].fillna('Warm Up') == name]['intensity'].iloc[0]
            if pd.notna(valid[valid['wkt_step_name'].fillna('Warm Up') == name]['intensity'].iloc[0])
            else 'active',
            '#95a5a6'
        )
        for name in step_types.index
    ]
    ax2.bar(range(len(step_types)), step_types.values, color=colors_bar, edgecolor='white')
    ax2.set_xticks(range(len(step_types)))
    ax2.set_xticklabels(step_types.index, fontsize=9)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Step Type Distribution', fontsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    total_min = valid['duration_time'].sum() / 60
    fig.suptitle(f'Training Plan Summary (Total: {total_min:.0f} min)', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_training_plan(df: pd.DataFrame, save_path: str = None):
    valid = df[df['duration_time'].notna()].copy()
    valid['step_label'] = valid.apply(
        lambda r: r['wkt_step_name'] if pd.notna(r['wkt_step_name']) else f"WarmUp {int(r['message_index']) + 1}",
        axis=1
    )
    valid['mid_power'] = (valid['custom_target_power_low'] + valid['custom_target_power_high']) / 2
    valid['cum_time'] = valid['duration_time'].cumsum() / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1]})

    for i, (_, row) in enumerate(valid.iterrows()):
        low = row['custom_target_power_low']
        high = row['custom_target_power_high']
        intensity = row['intensity'] if pd.notna(row['intensity']) else 'active'
        color = INTENSITY_COLORS.get(intensity, '#95a5a6')
        ax1.barh(i, high - low, left=low, height=0.6, color=color, edgecolor='white', linewidth=0.5)

    ax1.set_yticks(range(len(valid)))
    ax1.set_yticklabels(valid['step_label'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Target Power (W)', fontsize=12)
    ax1.set_title('Target Power Range per Step', fontsize=13)

    legend_patches = [
        mpatches.Patch(color=color, label=INTENSITY_LABELS.get(key, key))
        for key, color in INTENSITY_COLORS.items()
    ]
    ax1.legend(handles=legend_patches, loc='lower right', fontsize=9)

    prev = 0
    for i, (_, row) in enumerate(valid.iterrows()):
        dur = row['duration_time'] / 60
        intensity = row['intensity'] if pd.notna(row['intensity']) else 'active'
        color = INTENSITY_COLORS.get(intensity, '#95a5a6')
        ax2.barh(0, dur, left=prev, height=0.8, color=color, edgecolor='white', linewidth=0.5)
        if dur >= 1:
            ax2.text(prev + dur / 2, 0, row['step_label'], ha='center', va='center', fontsize=8, color='white',
                     fontweight='bold')
        prev += dur

    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_yticks([])
    ax2.set_title('Training Timeline (Continuous)', fontsize=13)
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=9)

    total_min = valid['duration_time'].sum() / 60
    fig.suptitle(f'Training Plan: VO2max 30/30s Intervals (Total: {total_min:.0f} min)', fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    BASE_DIR = Path(__file__).parent / 'test_data'
    WORKOUT_FIT = BASE_DIR / '2026-06-09_\u6444\u6c27\u91cf3030s41.fit'

    if not WORKOUT_FIT.exists():
        print(f"FIT \u6587\u4ef6\u4e0d\u5b58\u5728: {WORKOUT_FIT}")
        print("\n>>> test_data \u76ee\u5f55\u4e2d\u7684\u6587\u4ef6:")
        for f in BASE_DIR.iterdir():
            print(f"  - {f.name}")
        sys.exit(1)

    print(">>> FIT \u6587\u4ef6\u6d88\u606f\u7c7b\u578b:")
    msg_types = get_fit_message_types(str(WORKOUT_FIT))
    for t in sorted(msg_types):
        print(f"  - {t}")

    steps_df = extract_workout_steps(str(WORKOUT_FIT))
    print_workout_steps(steps_df)

    print("\n>>> \u7ed8\u5236\u8bad\u7ec3\u8ba1\u5212\u53ef\u89c6\u5316\u56fe\u8868...")
    plot_training_power_targets(steps_df)
    plot_training_duration(steps_df)
    plot_training_summary(steps_df)
    plot_training_plan(steps_df)
    print("\u53ef\u89c6\u5316\u5b8c\u6210\uff01")
