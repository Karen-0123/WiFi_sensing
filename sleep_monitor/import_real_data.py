import os
import json
import math
import warnings
warnings.filterwarnings("ignore")

import pymysql
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from crud import create_session

TARGET_EMAIL = "test_user1@gmail.com"
CSV_FILE_PATH = r"C:\Users\Admin\OneDrive\Documents\subject015_features_aligned.csv"

def get_db_secure():
    return pymysql.connect(
        host="mysql-46cb3ab-ntou-project.h.aivencloud.com",
        port=21225,
        user="avnadmin",
        password="AVNS_kegvXqQywhPKN1Xr4Yp",
        database="defaultdb",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl_mode": "REQUIRED"}
    )

def compute_clinical_score(total_min, rem_min, core_min, awake_min, motion_count):
    hours = total_min / 60.0
    if 7.0 <= hours <= 9.0:
        dur_score = 40.0
    elif hours < 7.0:
        dur_score = max(0.0, 40.0 * (hours / 7.0))
    else:
        dur_score = max(20.0, 40.0 - (hours - 9.0) * 5.0)

    valid_sleep = max(1.0, total_min - awake_min)
    rem_ratio = rem_min / valid_sleep
    rem_score = 25.0 if (0.18 <= rem_ratio <= 0.28) else max(10.0, 25.0 * (1.0 - abs(rem_ratio - 0.22) * 2))

    core_ratio = core_min / valid_sleep
    core_score = 25.0 if (0.65 <= core_ratio <= 0.82) else max(10.0, 25.0 * (1.0 - abs(core_ratio - 0.75)))

    awake_ratio = awake_min / max(1.0, total_min)
    rest_score = max(0.0, 10.0 - (awake_ratio * 30.0 + motion_count * 0.2))

    score_100 = round(min(100.0, dur_score + rem_score + core_score + rest_score), 1)
    score_10 = round(score_100 / 10.0, 1)
    return score_100, score_10

def build_single_night_features(df_raw, feature_cols):
    df = df_raw.copy()
    dev_col = [c for c in ["Breathing_Rate_Deviation", "bpm_deviation_final", "RespDeviation"] if c in df.columns][0]
    var_col = [c for c in ["Breathing_Rate_Variability", "var_history_final", "RespVar"] if c in df.columns][0]
    has_events = "Num_Events" in df.columns

    for w in [5, 10, 15]:
        for col in [dev_col, var_col]:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
            df[f"{col}_roll_range_{w}"] = df[col].rolling(w, min_periods=1).max() - df[col].rolling(w, min_periods=1).min()
        if has_events:
            df[f"Num_Events_roll_sum_{w}"] = df["Num_Events"].rolling(w, min_periods=1).sum()

    for col in [dev_col, var_col]:
        for step in [1, 2, -1, -2]:
            tag = f"lag_{step}" if step > 0 else f"lead_{abs(step)}"
            df[f"{col}_{tag}"] = df[col].shift(step).bfill().ffill().fillna(0.0)

    for col in [dev_col, var_col]:
        df[f"{col}_diff1"] = df[col].diff().fillna(0.0)
    df["Dev_x_Var"] = df[dev_col] * df[var_col]

    df["Sleep_Progress"] = np.linspace(0, 1, len(df))
    df["Progress_x_Var"] = df["Sleep_Progress"] * df[var_col]
    df["Progress_sin"] = np.sin(2 * np.pi * df["Sleep_Progress"] * 5)
    df["Progress_cos"] = np.cos(2 * np.pi * df["Sleep_Progress"] * 5)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

def sanitize_float(val, default=0.0):
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return float(val)

def run_import():
    print("==========================================================")
    print("  WiFi CSI 睡眠監測系統 — 模型推論與資料庫同步")
    print("==========================================================")

    if not os.path.exists(CSV_FILE_PATH):
        print(f"錯誤：找不到特徵 CSV 檔案: {CSV_FILE_PATH}")
        return

    try:
        model = joblib.load("sleep_model.pkl")
        scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
        with open("model_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        feature_cols = config["feature_cols"]
        print(f"成功載入 ML 冠軍模型: {config.get('model_type', 'LightGBM')}")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    try:
        connection = get_db_secure()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (TARGET_EMAIL,))
            existing = cursor.fetchone()
            user_id = existing["id"] if existing else None
            if not user_id:
                cursor.execute(
                    "INSERT INTO users (username, display_name, password, wake_preference) VALUES (%s, '測試用戶', 'mock_pass', '{\"window_min\": 30}')",
                    (TARGET_EMAIL,)
                )
                connection.commit()
                user_id = cursor.lastrowid
        connection.close()
    except Exception as e:
        print(f"資料庫用戶驗證失敗: {e}")
        return

    raw_df = pd.read_csv(CSV_FILE_PATH)
    raw_df.columns = raw_df.columns.str.strip()

    try:
        t0 = pd.to_datetime(raw_df["Start_Time"].iloc[0])
        t1 = pd.to_datetime(raw_df["Start_Time"].iloc[1])
        step_seconds = int((t1 - t0).total_seconds())
        if step_seconds <= 0: step_seconds = 180
    except Exception:
        step_seconds = 180

    started_at = pd.to_datetime(raw_df["Start_Time"].iloc[0]) if "Start_Time" in raw_df.columns else (datetime.now() - timedelta(hours=5))
    step_minutes = step_seconds / 60.0

    session_id = create_session(user_id=user_id, started_at=started_at, date=started_at.strftime("%Y-%m-%d"))

    dev_col = [c for c in ["Breathing_Rate_Deviation", "bpm_deviation_final", "RespDeviation"] if c in raw_df.columns][0]
    var_col = [c for c in ["Breathing_Rate_Variability", "var_history_final", "RespVar"] if c in raw_df.columns][0]
    raw_df[dev_col] = pd.to_numeric(raw_df[dev_col], errors="coerce").fillna(0.0)
    raw_df[var_col] = pd.to_numeric(raw_df[var_col], errors="coerce").fillna(0.0)
    if "Num_Events" not in raw_df.columns: raw_df["Num_Events"] = 0
    if "Wake_Sleep" not in raw_df.columns: raw_df["Wake_Sleep"] = 1

    X_feats = build_single_night_features(raw_df, feature_cols)
    X_scaled = scaler.transform(X_feats.values) if scaler is not None else X_feats.values

    # 機率推論
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        classes = list(getattr(model, "classes_", [1, 2]))
        pos_idx = classes.index(1) if 1 in classes else 0
        rem_probs = probs[:, pos_idx]
    else:
        rem_probs = np.zeros(len(raw_df))

    base_thresh = float(config.get("best_threshold", 0.50))
    effective_thresh = max(0.12, float(np.percentile(rem_probs, 75))) if np.max(rem_probs) < base_thresh else base_thresh

    print("\n--- 機器學習推論機率診斷 ---")
    print(f"REM 預測機率最高值: {np.max(rem_probs):.4f}")
    print(f"REM 預測機率平均值: {np.mean(rem_probs):.4f}")
    print(f"REM 預測機率 75百分位數: {np.percentile(rem_probs, 75):.4f}")
    print(f"REM 預測機率 90百分位數: {np.percentile(rem_probs, 90):.4f}")
    print(f"使用自適應門檻: {effective_thresh:.2f}")

    # 階段判定：awake, rem, core
    final_stages = []
    for i in range(len(raw_df)):
        if raw_df["Wake_Sleep"].iloc[i] == 0:
            final_stages.append("awake")
        elif rem_probs[i] >= effective_thresh:
            final_stages.append("rem")
        else:
            final_stages.append("core")

    watch_rem_epochs = (raw_df["Sleep_Stage"] == 1).sum() if "Sleep_Stage" in raw_df.columns else 0
    watch_core_epochs = (raw_df["Sleep_Stage"] == 2).sum() if "Sleep_Stage" in raw_df.columns else 0
    watch_awake_epochs = (raw_df["Sleep_Stage"] == 0).sum() if "Sleep_Stage" in raw_df.columns else 0

    bulk_data = []
    stage_counts = {"core": 0, "rem": 0, "awake": 0}
    total_rr = 0.0

    for i in range(len(raw_df)):
        stage_name = final_stages[i]
        dev_val = float(raw_df[dev_col].iloc[i])
        rr = round(14.5 + (dev_val * 0.4), 1)
        if rr <= 8.0 or rr >= 28.0: rr = 16.0

        timestamp = started_at + timedelta(seconds=step_seconds * i)
        motion = int(raw_df["Num_Events"].iloc[i]) if "Num_Events" in raw_df.columns else (1 if stage_name == "awake" else 0)

        bulk_data.append((session_id, timestamp, rr, 1.0, stage_name, motion))
        stage_counts[stage_name] += 1
        total_rr += rr

    try:
        conn = get_db_secure()
        with conn.cursor() as cursor:
            insert_sql = """
                INSERT INTO respiration_logs (session_id, timestamp, respiration_rate, signal_quality, inferred_stage, motion_detected)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.executemany(insert_sql, bulk_data)

            awake_min = int(stage_counts['awake'] * step_minutes)
            rem_min = int(stage_counts['rem'] * step_minutes)
            core_min = int(stage_counts['core'] * step_minutes)
            total_min = awake_min + rem_min + core_min

            avg_rr = round(total_rr / len(bulk_data), 2)
            total_motion = int((raw_df["Wake_Sleep"] == 0).sum())

            score_100, real_score = compute_clinical_score(total_min, rem_min, core_min, awake_min, total_motion)
            real_score = sanitize_float(real_score, 8.5)
            avg_rr = sanitize_float(avg_rr, 15.8)

            update_sql = """
                UPDATE sleep_summaries 
                SET sleep_score = %s, avg_respiration_rate = %s, deep_sleep_minutes = 0, 
                    light_sleep_minutes = %s, rem_sleep_minutes = %s, awake_minutes = %s,
                    ended_at = %s, status = 'done' 
                    WHERE id = %s
            """
            cursor.execute(update_sql, (real_score, avg_rr, core_min, rem_min, awake_min, datetime.now(), session_id))

        conn.commit()
        conn.close()

        print("\n" + "=" * 65)
        print(f" 檔案來源         : {os.path.basename(CSV_FILE_PATH)}")
        print(f" 監測時長         : {total_min} 分鐘 ({total_min/60:.1f} 小時 )")
        print(f" 平均呼吸率       : {avg_rr} BPM")
        print("-" * 65)
        print(f" 手錶真實標籤  : REM {int(watch_rem_epochs*step_minutes)}m | Core(NREM) {int(watch_core_epochs*step_minutes)}m | 清醒 {int(watch_awake_epochs*step_minutes)}m")
        print(f"  ML 模型預測   : REM {rem_min}m | Core(NREM) {core_min}m | 清醒 {awake_min}m")
        print(f" 睡眠評分 (10分制): {real_score} 分 (百分制: {score_100} 分)")
        print("=" * 65)
        print(" 推論與同步完成！請至前端網頁按 F5 查看更新結果！")
        print("=" * 65)

    except Exception as e:
        print(f"寫入或結算失敗: {e}")

if __name__ == "__main__":
    run_import()