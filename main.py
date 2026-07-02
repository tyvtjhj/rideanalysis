import sys
from pathlib import Path

from workout_parser import get_fit_message_types, extract_workout_steps, print_workout_steps
from activity_analyzer import (
    extract_records, extract_laps, extract_session,
    add_power_zone_fixed, add_power_bin_percentile, add_rolling_power_zone,
    detect_power_changes,
    segment_by_power_change, segment_by_window_variance,
    plot_power_curve, plot_heart_rate, plot_dual_axis,
    plot_segmented_power, plot_all_segments,
    print_record_summary, print_lap_summary,
    print_session_summary, print_segment_stats,
)


def main():
    BASE_DIR = Path(__file__).parent / 'test_data'

    WORKOUT_FIT = BASE_DIR / '2026-06-09_\u6444\u6c27\u91cf3030s41.fit'
    ACTIVITY_FIT = BASE_DIR / '23182812295_ACTIVITY.fit'

    # ----- 第一阶段：训练计划 -----
    print("=" * 60)
    print("\u7b2c\u4e00\u9636\u6bb5\uff1a\u8bad\u7ec3\u8ba1\u5212")
    print("=" * 60)

    print("\n>>> FIT \u6587\u4ef6\u6d88\u606f\u7c7b\u578b:")
    msg_types = get_fit_message_types(str(WORKOUT_FIT))
    for t in sorted(msg_types):
        print(f"  - {t}")

    steps_df = extract_workout_steps(str(WORKOUT_FIT))
    print_workout_steps(steps_df)

    # ----- 第二阶段：活动数据分析与可视化 -----
    print("\n" + "=" * 60)
    print("\u7b2c\u4e8c\u9636\u6bb5\uff1a\u6d3b\u52a8\u6570\u636e\u5206\u6790\u4e0e\u53ef\u89c6\u5316")
    print("=" * 60)

    if not ACTIVITY_FIT.exists():
        print(f"\n!!! \u6d3b\u52a8 FIT \u6587\u4ef6\u4e0d\u5b58\u5728: {ACTIVITY_FIT}")
        print("!!! \u8df3\u8fc7\u6d3b\u52a8\u6570\u636e\u5206\u6790\u3002\u8bf7\u5c06\u6d3b\u52a8 FIT \u6587\u4ef6\u653e\u5165 test_data \u76ee\u5f55\u540e\u91cd\u8bd5\u3002")
        print("\n>>> \u5f53\u524d test_data \u76ee\u5f55\u4e2d\u7684\u6587\u4ef6:")
        for f in BASE_DIR.iterdir():
            print(f"  - {f.name}")
        return 0

    df = extract_records(str(ACTIVITY_FIT))
    print_record_summary(df)

    lap_df = extract_laps(str(ACTIVITY_FIT))
    print_lap_summary(lap_df)

    session_data = extract_session(str(ACTIVITY_FIT))
    print_session_summary(session_data)

    print("\n>>> \u7ed8\u5236\u529f\u7387\u66f2\u7ebf...")
    plot_power_curve(df)

    print(">>> \u7ed8\u5236\u5fc3\u7387\u66f2\u7ebf + \u53ccY\u8f74\u5bf9\u6bd4\u56fe...")
    plot_heart_rate(df)
    plot_dual_axis(df)

    print("\n>>> \u56fa\u5b9a\u9608\u503c\u529f\u7387\u5206\u533a:")
    df_zones = add_power_zone_fixed(df)
    print(df_zones['power_zone'].value_counts())

    print("\n>>> \u767e\u5206\u4f4d\u6570\u529f\u7387\u5206\u7bb1:")
    df_bins = add_power_bin_percentile(df)
    print(df_bins['power_bin'].value_counts())

    print("\n>>> \u6ed1\u52a8\u7a97\u53e3\u5e73\u5747\u529f\u7387\u5206\u533a (window=60):")
    df_rolling = add_rolling_power_zone(df, window_size=60)
    print(df_rolling['window_zone'].value_counts())

    print("\n>>> \u529f\u7387\u53d8\u5316\u68c0\u6d4b (threshold=50W):")
    df_changes = detect_power_changes(df, threshold=50)
    change_count = df_changes['is_activity_change'].sum()
    print(f"  \u68c0\u6d4b\u5230 {change_count} \u4e2a\u529f\u7387\u53d8\u5316\u70b9 (\u5171 {len(df_changes)} \u4e2a\u91c7\u6837\u70b9)")

    print("\n>>> \u81ea\u52a8\u5206\u6bb5 (\u65b9\u6cd51: \u529f\u7387\u53d8\u5316\u68c0\u6d4b, threshold=50, min_segment=30):")
    df_segmented = segment_by_power_change(df, threshold=50, min_segment_length=30)
    print(f"  \u5206\u6bb5\u6570\u91cf: {df_segmented['segment_id'].nunique()}")
    print_segment_stats(df_segmented)

    print("\n>>> \u81ea\u52a8\u5206\u6bb5 (\u65b9\u6cd52: \u7a97\u53e3\u65b9\u5dee\u68c0\u6d4b, window=60, threshold_ratio=2):")
    df_segmented2 = segment_by_window_variance(df, window_size=60, threshold_ratio=2)
    print(f"  \u5206\u6bb5\u6570\u91cf: {df_segmented2['segment_id'].nunique()}")

    print("\n>>> \u7ed8\u5236\u5206\u6bb5\u529f\u7387\u66f2\u7ebf (Top 10)...")
    plot_segmented_power(df_segmented, max_segments=10)

    print("\n>>> \u7ed8\u5236\u5168\u90e8\u5206\u6bb5\u529f\u7387\u66f2\u7ebf...")
    plot_all_segments(df_segmented)

    print("\n" + "=" * 60)
    print("\u5206\u6790\u5b8c\u6210\uff01")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
