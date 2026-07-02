from pathlib import Path
import fitparse
import pandas as pd


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
