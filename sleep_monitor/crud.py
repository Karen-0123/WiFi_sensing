# crud.py
import sqlite3
from datetime import datetime
from typing import Optional
from database import get_db
from models import SessionChartData, ChartDataPoint, SleepStageBreakdown


# ══════════════════════════════════════════
# 一、建立新睡眠 Session
# ══════════════════════════════════════════

def create_session(user_id: int, started_at: datetime, date: str) -> int:
    """
    建立一筆新的睡眠監測紀錄。
    回傳新建立的 session id（後續寫入呼吸數據時需要用到）。

    參數：
        user_id    : 使用者 id（必須已存在於 users 表）
        started_at : 開始監測的時間
        date       : 入睡當天日期，格式 "YYYY-MM-DD"

    回傳：
        新 session 的 id (int)
    """
    sql = """
        INSERT INTO sleep_summaries (user_id, date, started_at, status)
        VALUES (?, ?, ?, 'recording')
    """
    with get_db() as conn:
        cursor = conn.execute(sql, (user_id, date, started_at))
        session_id = cursor.lastrowid  # 取得自動產生的 id

    print(f"✅ Session 建立成功，session_id = {session_id}")
    return session_id


# ══════════════════════════════════════════
# 二、新增一筆呼吸數據
# ══════════════════════════════════════════

def append_respiration_log(
    session_id: int,
    timestamp: datetime,
    respiration_rate: float,
    signal_quality: float,
    inferred_stage: str,        # 由 MATLAB 傳入：deep / light / rem / awake
    motion_detected: int = 0,   # 0=無移動, 1=有移動
) -> int:
    """
    新增一筆呼吸數據到 respiration_logs 表。
    睡眠階段（inferred_stage）由 MATLAB 計算後直接傳入，Python 不再推算。
    自動判斷是否為異常值（outlier）。

    回傳：
        新增紀錄的 id (int)
    """
    # 異常值判斷：呼吸率超出正常範圍 或 訊號品質過低
    is_outlier = _detect_outlier(respiration_rate, signal_quality)

    sql = """
        INSERT INTO respiration_logs
            (session_id, timestamp, respiration_rate,
             signal_quality, inferred_stage, is_outlier, motion_detected)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_db() as conn:
        cursor = conn.execute(sql, (
            session_id,
            timestamp,
            respiration_rate,
            signal_quality,
            inferred_stage,
            is_outlier,
            motion_detected,
        ))
        log_id = cursor.lastrowid

    return log_id


def _detect_outlier(respiration_rate: float, signal_quality: float) -> int:
    """
    內部函數：判斷這筆數據是否為異常值。
    條件（任一成立即標記為異常）：
      - 呼吸率低於 6 或高於 30（明顯超出生理範圍）
      - 訊號品質低於 0.3（CSI 訊號太差，數據不可信）

    回傳 1（是異常）或 0（正常）
    """
    if respiration_rate < 6 or respiration_rate > 30:
        return 1
    if signal_quality < 0.3:
        return 1
    return 0


# ══════════════════════════════════════════
# 三、結算睡眠 Session
# ══════════════════════════════════════════

def close_session(session_id: int) -> dict:
    """
    結算一個睡眠 Session：
      1. 計算各睡眠階段時間（分鐘）
      2. 計算平均呼吸率（排除異常值）
      3. 計算睡眠分數（0–100）
      4. 計算獎勵積分
      5. 將以上結果更新回 sleep_summaries 表，狀態改為 'done'

    回傳：
        結算結果的 dict
    """
    with get_db() as conn:

        # ── 步驟 1：取得所有非異常值的呼吸紀錄 ──
        logs = conn.execute("""
            SELECT respiration_rate, inferred_stage
            FROM respiration_logs
            WHERE session_id = ?
              AND is_outlier = 0      -- 排除異常值
        """, (session_id,)).fetchall()

        if not logs:
            raise ValueError(f"Session {session_id} 沒有有效的呼吸數據，無法結算")

        # ── 步驟 2：計算各睡眠階段時間 ──
        # 每筆數據代表 3 分鐘的監測區間
        MINUTES_PER_LOG = 3

        stage_counts = {"deep": 0, "light": 0, "rem": 0, "awake": 0}
        total_rr = 0.0

        for log in logs:
            stage = log["inferred_stage"]
            if stage in stage_counts:
                stage_counts[stage] += 1
            total_rr += log["respiration_rate"]

        deep_minutes  = stage_counts["deep"]  * MINUTES_PER_LOG
        light_minutes = stage_counts["light"] * MINUTES_PER_LOG
        rem_minutes   = stage_counts["rem"]   * MINUTES_PER_LOG

        # ── 步驟 3：計算平均呼吸率 ──
        avg_rr = round(total_rr / len(logs), 2)

        # ── 步驟 4：計算睡眠分數 ──
        sleep_score = _calculate_sleep_score(
            deep_minutes, light_minutes, rem_minutes
        )

        # ── 步驟 5：計算獎勵積分（每 10 分睡眠分數 = 1 點）──
        reward_points = int(sleep_score // 10)

        # ── 步驟 6：取得結束時間（最後一筆數據的 timestamp）──
        last_log = conn.execute("""
            SELECT timestamp FROM respiration_logs
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id,)).fetchone()
        ended_at = last_log["timestamp"]

        # ── 步驟 7：更新 sleep_summaries ──
        conn.execute("""
            UPDATE sleep_summaries
            SET ended_at             = ?,
                sleep_score          = ?,
                avg_respiration_rate = ?,
                deep_sleep_minutes   = ?,
                light_sleep_minutes  = ?,
                rem_sleep_minutes    = ?,
                reward_points        = ?,
                status               = 'done'
            WHERE id = ?
        """, (
            ended_at,
            sleep_score,
            avg_rr,
            deep_minutes,
            light_minutes,
            rem_minutes,
            reward_points,
            session_id,
        ))

    result = {
        "session_id":            session_id,
        "sleep_score":           sleep_score,
        "avg_respiration_rate":  avg_rr,
        "deep_sleep_minutes":    deep_minutes,
        "light_sleep_minutes":   light_minutes,
        "rem_sleep_minutes":     rem_minutes,
        "reward_points":         reward_points,
        "status":                "done",
    }
    print(f"✅ Session {session_id} 結算完成：分數 {sleep_score}，積分 {reward_points}")
    return result


def _calculate_sleep_score(
    deep_minutes: int,
    light_minutes: int,
    rem_minutes: int,
) -> float:
    """
    內部函數：根據睡眠階段時間計算睡眠分數（0–100）。

    計分邏輯（可依需求調整）：
      - 深睡（deep） 權重最高：每分鐘 +0.5 分，上限 40 分
      - REM           權重次之：每分鐘 +0.3 分，上限 30 分
      - 淺睡（light） 權重最低：每分鐘 +0.1 分，上限 30 分
      - 總分上限 100 分
    """
    deep_score  = min(deep_minutes  * 0.5, 40.0)
    rem_score   = min(rem_minutes   * 0.3, 30.0)
    light_score = min(light_minutes * 0.1, 30.0)

    total = round(deep_score + rem_score + light_score, 1)
    return min(total, 100.0)  # 確保不超過 100


# ══════════════════════════════════════════
# 四、取得前端圖表資料
# ══════════════════════════════════════════

def get_session_chart_data(session_id: int) -> SessionChartData:
    """
    取得一個 Session 的完整圖表資料。
    回傳 SessionChartData 物件，前端可直接用來渲染所有圖表。
    """
    with get_db() as conn:

        # ── 取得 Session 摘要 ──
        summary = conn.execute("""
            SELECT * FROM sleep_summaries
            WHERE id = ?
        """, (session_id,)).fetchone()

        if not summary:
            raise ValueError(f"找不到 session_id = {session_id} 的紀錄")

        # ── 取得所有呼吸數據（依時間排序）──
        logs = conn.execute("""
            SELECT timestamp, respiration_rate, signal_quality,
                   inferred_stage, is_outlier, motion_detected
            FROM respiration_logs
            WHERE session_id = ?
            ORDER BY timestamp ASC   -- 使用複合索引，查詢效率高
        """, (session_id,)).fetchall()

        # ── 組合時序圖表資料點 ──
        chart_data = [
            ChartDataPoint(
                timestamp        = log["timestamp"],
                respiration_rate = log["respiration_rate"],
                inferred_stage   = log["inferred_stage"],
                is_outlier       = log["is_outlier"],
                signal_quality   = log["signal_quality"],
                motion_detected  = log["motion_detected"],
            )
            for log in logs
        ]

        # ── 計算睡眠階段分佈（含 awake）──
        MINUTES_PER_LOG = 3
        stage_minutes = {"deep": 0, "light": 0, "rem": 0, "awake": 0}

        for log in logs:
            stage = log["inferred_stage"]
            if stage in stage_minutes:
                stage_minutes[stage] += MINUTES_PER_LOG

        stage_breakdown = SleepStageBreakdown(
            deep_sleep_minutes  = stage_minutes["deep"],
            light_sleep_minutes = stage_minutes["light"],
            rem_sleep_minutes   = stage_minutes["rem"],
            awake_minutes       = stage_minutes["awake"],
        )

        # ── 統計異常值數量 ──
        outlier_count = sum(1 for log in logs if log["is_outlier"] == 1)

    return SessionChartData(
        session_id           = session_id,
        user_id              = summary["user_id"],
        date                 = summary["date"],
        started_at           = summary["started_at"],
        ended_at             = summary["ended_at"],
        status               = summary["status"],
        sleep_score          = summary["sleep_score"],
        avg_respiration_rate = summary["avg_respiration_rate"],
        reward_points        = summary["reward_points"],
        chart_data           = chart_data,
        stage_breakdown      = stage_breakdown,
        total_logs           = len(logs),
        outlier_count        = outlier_count,
    )