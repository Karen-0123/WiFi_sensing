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
    """
    sql = """
        INSERT INTO sleep_summaries (user_id, date, started_at, status)
        VALUES (%s, %s, %s, 'recording')
    """
    with get_db() as cursor:
        cursor.execute(sql, (user_id, date, started_at))
        session_id = cursor.lastrowid  # pymysql 游標可以直接取得最後插入的 id

    print(f"Session 建立成功，session_id = {session_id}")
    return session_id


# ══════════════════════════════════════════
# 二、新增一筆呼吸數據（內建防錯與異常值排除）
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
    """
    # 數值防爆保護
    if respiration_rate is None or respiration_rate != respiration_rate:
        return 0

    # 異常值判斷
    is_outlier = _detect_outlier(respiration_rate, signal_quality)

    sql = """
        INSERT INTO respiration_logs
            (session_id, timestamp, respiration_rate,
             signal_quality, inferred_stage, is_outlier, motion_detected)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with get_db() as cursor:
        cursor.execute(sql, (
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
    if respiration_rate < 6 or respiration_rate > 30:
        return 1
    if signal_quality < 0.3:
        return 1
    return 0


# ══════════════════════════════════════════
# 三、結算睡眠 Session (4階段解鎖 + 前端規格縮放)
# ══════════════════════════════════════════

def close_session(session_id: int) -> dict:
    """
    結算一個睡眠 Session 並且更新回 MySQL。
    """
    with get_db() as cursor:

        # ── 步驟 1：取得所有非異常值的呼吸紀錄 ──
        cursor.execute("""
            SELECT respiration_rate, inferred_stage
            FROM respiration_logs
            WHERE session_id = %s
              AND is_outlier = 0
              AND respiration_rate IS NOT NULL
        """, (session_id,))
        logs = cursor.fetchall()

        if not logs:
            # 保底機制
            cursor.execute("""
                SELECT respiration_rate, inferred_stage
                FROM respiration_logs
                WHERE session_id = %s
            """, (session_id,))
            logs = cursor.fetchall()

        if not logs:
            raise ValueError(f"Session {session_id} 沒有任何呼吸數據，無法進行結算")

        # ── 步驟 2：計算各睡眠階段時間 ──
        MINUTES_PER_LOG = 3
        stage_counts = {"deep": 0, "light": 0, "rem": 0, "awake": 0}
        total_rr = 0.0
        valid_rr_count = 0

        for log in logs:
            stage = log["inferred_stage"]
            if stage in stage_counts:
                stage_counts[stage] += 1
            
            # ✨ 這裡已經把錯誤的大寫 NULL 修正成最標準的 Python 語法囉！
            if log["respiration_rate"] is not None:
                total_rr += float(log["respiration_rate"])
                valid_rr_count += 1

        deep_minutes  = stage_counts["deep"]  * MINUTES_PER_LOG
        light_minutes = stage_counts["light"] * MINUTES_PER_LOG
        rem_minutes   = stage_counts["rem"]   * MINUTES_PER_LOG
        awake_minutes = stage_counts["awake"] * MINUTES_PER_LOG

        # ── 步驟 3：計算平均呼吸率 ──
        avg_rr = round(total_rr / valid_rr_count, 2) if valid_rr_count > 0 else 15.0

        # ── 步驟 4：計算睡眠分數 ──
        raw_score = _calculate_sleep_score(
            deep_minutes, light_minutes, rem_minutes
        )

        # 🛡️ 【前端亮綠色圓環防護盾】
        if raw_score > 10.0:
            sleep_score = round(raw_score / 10.0, 1)
        else:
            sleep_score = round(raw_score, 1)

        # ── 步驟 5：計算獎勵積分 ──
        reward_points = int(sleep_score)

        # ── 步驟 6：取得結束時間 ──
        cursor.execute("""
            SELECT timestamp FROM respiration_logs
            WHERE session_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id,))
        last_log = cursor.fetchone()
        ended_at = last_log["timestamp"] if last_log else datetime.now()

        # ── 步驟 7：更新 sleep_summaries ──
        cursor.execute("""
            UPDATE sleep_summaries
            SET ended_at             = %s,
                sleep_score          = %s,
                avg_respiration_rate = %s,
                deep_sleep_minutes   = %s,
                light_sleep_minutes  = %s,
                rem_sleep_minutes    = %s,
                awake_minutes        = %s,
                reward_points        = %s,
                status               = 'done'
            WHERE id = %s
        """, (
            ended_at,
            sleep_score,
            avg_rr,
            deep_minutes,
            light_minutes,
            rem_minutes,
            awake_minutes,
            reward_points,
            session_id,
        ))

    return {
        "session_id":            session_id,
        "sleep_score":           sleep_score,
        "avg_respiration_rate":  avg_rr,
        "deep_sleep_minutes":    deep_minutes,
        "light_sleep_minutes":   light_minutes,
        "rem_sleep_minutes":     rem_minutes,
        "awake_minutes":         awake_minutes,
        "reward_points":         reward_points,
        "status":                "done",
    }


def _calculate_sleep_score(
    deep_minutes: int,
    light_minutes: int,
    rem_minutes: int,
) -> float:
    deep_score  = min(deep_minutes  * 0.5, 40.0)
    rem_score   = min(rem_minutes   * 0.3, 30.0)
    light_score = min(light_minutes * 0.1, 30.0)
    total = round(deep_score + rem_score + light_score, 1)
    return min(total, 100.0)


# ══════════════════════════════════════════
# 四、取得前端圖表資料 (4階段完全解鎖版)
# ══════════════════════════════════════════

def get_session_chart_data(session_id: int) -> SessionChartData:
    """
    取得一個 Session 的完整圖表資料，封裝回傳傳入 models 物件。
    """
    with get_db() as cursor:

        # ── 1. 取得 Session 摘要 ──
        cursor.execute("""
            SELECT * FROM sleep_summaries
            WHERE id = %s
        """, (session_id,))
        summary = cursor.fetchone()

        if not summary:
            raise ValueError(f"找不到 session_id = {session_id} 的紀錄")

        # ── 2. 取得所有呼吸數據（依時間排序）──
        cursor.execute("""
            SELECT timestamp, respiration_rate, signal_quality,
                   inferred_stage, is_outlier, motion_detected
            FROM respiration_logs
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """, (session_id,))
        logs = cursor.fetchall()

        # ── 3. 組合時序圖表資料點 ──
        chart_data = [
            ChartDataPoint(
                timestamp        = log["timestamp"],
                respiration_rate = float(log["respiration_rate"]) if log["respiration_rate"] else None,
                inferred_stage   = log["inferred_stage"],
                is_outlier       = log["is_outlier"],
                signal_quality   = float(log["signal_quality"]) if log["signal_quality"] else 0.0,
                motion_detected  = log["motion_detected"],
            )
            for log in logs
        ]

        # ── 4. 計算睡眠階段分佈 ──
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

        # ── 5. 統計異常值數量 ──
        outlier_count = sum(1 for log in logs if log["is_outlier"] == 1)

    return SessionChartData(
        session_id           = session_id,
        user_id              = summary["user_id"],
        date                 = str(summary["date"]),
        started_at           = summary["started_at"],
        ended_at             = summary["ended_at"],
        status               = summary["status"],
        sleep_score          = float(summary["sleep_score"]) if summary["sleep_score"] else 0.0,
        avg_respiration_rate = float(summary["avg_respiration_rate"]) if summary["avg_respiration_rate"] else 0.0,
        reward_points        = summary["reward_points"],
        chart_data           = chart_data,
        stage_breakdown      = stage_breakdown,
        total_logs           = len(logs),
        outlier_count        = outlier_count,
    )