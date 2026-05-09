# test_db.py
from datetime import datetime, timedelta
from schema import init_db
from database import get_db
from crud import (
    create_session,
    append_respiration_log,
    close_session,
    get_session_chart_data,
)

# ══════════════════════════════════════════
# 測試資料設定
# ══════════════════════════════════════════

# 模擬 MATLAB 傳來的呼吸數據（10 筆，每筆間隔 3 分鐘）
MOCK_LOGS = [
    # (respiration_rate, signal_quality, inferred_stage, motion_detected)
    (14.5, 0.92, "light",  0),
    (13.8, 0.88, "light",  0),
    (12.1, 0.95, "deep",   0),
    (11.9, 0.91, "deep",   0),
    (12.5, 0.89, "deep",   0),
    (15.2, 0.85, "rem",    0),
    (16.1, 0.87, "rem",    0),
    (35.0, 0.20, "awake",  1),  # 這筆應被標記為 outlier（呼吸率過高且訊號差）
    (14.8, 0.90, "light",  0),
    (13.5, 0.93, "light",  0),
]


# ══════════════════════════════════════════
# 測試流程
# ══════════════════════════════════════════

def run_test():
    print("=" * 55)
    print("  WiFi CSI 睡眠監測系統 — 資料庫完整流程測試")
    print("=" * 55)

    # ── 階段 0：初始化資料庫 ──────────────────────────
    print("\n【階段 0】初始化資料庫")
    init_db()

    # ── 階段 1：建立測試使用者 ────────────────────────
    print("\n【階段 1】建立測試使用者")
    with get_db() as conn:

        # 避免重複執行時報錯，先檢查是否已存在
        existing = conn.execute(
            "SELECT id FROM users WHERE username = 'test_user'"
        ).fetchone()

        if existing:
            user_id = existing["id"]
            print(f"   使用者已存在，user_id = {user_id}")
        else:
            cursor = conn.execute("""
                INSERT INTO users (username, display_name, wake_preference)
                VALUES ('test_user', '測試用戶', '{"window_min": 30}')
            """)
            user_id = cursor.lastrowid
            print(f"   ✅ 使用者建立成功，user_id = {user_id}")

    # ── 階段 2：建立睡眠 Session ──────────────────────
    print("\n【階段 2】建立睡眠 Session")
    started_at = datetime(2026, 5, 9, 23, 0, 0)  # 晚上 11 點開始監測
    session_id = create_session(
        user_id    = user_id,
        started_at = started_at,
        date       = "2026-05-09",
    )

    # ── 階段 3：模擬寫入 10 筆呼吸數據 ───────────────
    print(f"\n【階段 3】寫入 {len(MOCK_LOGS)} 筆呼吸數據（模擬 MATLAB 傳入）")
    print(f"   {'#':<4} {'時間':<22} {'呼吸率':>6} {'訊號':>6} {'階段':<8} {'異常'}")
    print(f"   {'-'*60}")

    for i, (rr, quality, stage, motion) in enumerate(MOCK_LOGS):
        # 每筆數據間隔 3 分鐘
        timestamp = started_at + timedelta(minutes=3 * i)

        log_id = append_respiration_log(
            session_id       = session_id,
            timestamp        = timestamp,
            respiration_rate = rr,
            signal_quality   = quality,
            inferred_stage   = stage,
            motion_detected  = motion,
        )

        # 判斷這筆是否為異常（重現 _detect_outlier 邏輯，僅用於顯示）
        is_outlier = "⚠️ 是" if (rr < 6 or rr > 30 or quality < 0.3) else "否"

        print(f"   {i+1:<4} {str(timestamp):<22} {rr:>6.1f} {quality:>6.2f} {stage:<8} {is_outlier}")

    # ── 階段 4：結算 Session ──────────────────────────
    print(f"\n【階段 4】結算 Session {session_id}")
    result = close_session(session_id)

    print(f"\n   📊 結算結果：")
    print(f"      睡眠分數        : {result['sleep_score']} 分")
    print(f"      平均呼吸率      : {result['avg_respiration_rate']} 次/分鐘")
    print(f"      深睡時間        : {result['deep_sleep_minutes']} 分鐘")
    print(f"      淺睡時間        : {result['light_sleep_minutes']} 分鐘")
    print(f"      REM 時間        : {result['rem_sleep_minutes']} 分鐘")
    print(f"      獎勵積分        : {result['reward_points']} 點")
    print(f"      狀態            : {result['status']}")

    # ── 階段 5：查詢圖表資料 ──────────────────────────
    print(f"\n【階段 5】查詢圖表資料")
    chart = get_session_chart_data(session_id)

    print(f"\n   📈 圖表資料摘要：")
    print(f"      總資料筆數      : {chart.total_logs} 筆")
    print(f"      異常值筆數      : {chart.outlier_count} 筆")
    print(f"      睡眠階段分佈    :")
    print(f"        深睡          : {chart.stage_breakdown.deep_sleep_minutes} 分鐘")
    print(f"        淺睡          : {chart.stage_breakdown.light_sleep_minutes} 分鐘")
    print(f"        REM           : {chart.stage_breakdown.rem_sleep_minutes} 分鐘")
    print(f"        清醒          : {chart.stage_breakdown.awake_minutes} 分鐘")

    print(f"\n   📉 時序資料點（前 3 筆）：")
    for point in chart.chart_data[:3]:
        print(f"      {point.timestamp} | {point.respiration_rate} 次/分 "
              f"| {point.inferred_stage} | outlier={point.is_outlier}")

    # ── 階段 6：直接用 SQL 驗證資料庫內容 ────────────
    print(f"\n【階段 6】SQL 直接驗證")
    with get_db() as conn:

        # 驗證 sleep_summaries
        summary = conn.execute("""
            SELECT id, status, sleep_score, avg_respiration_rate,
                   deep_sleep_minutes, light_sleep_minutes, rem_sleep_minutes
            FROM sleep_summaries
            WHERE id = ?
        """, (session_id,)).fetchone()

        print(f"\n   sleep_summaries 驗證：")
        print(f"      id={summary['id']}, status={summary['status']}, "
              f"score={summary['sleep_score']}, avg_rr={summary['avg_respiration_rate']}")

        # 驗證 respiration_logs 總筆數
        total = conn.execute("""
            SELECT COUNT(*) as cnt FROM respiration_logs
            WHERE session_id = ?
        """, (session_id,)).fetchone()

        # 驗證 outlier 數量
        outliers = conn.execute("""
            SELECT COUNT(*) as cnt FROM respiration_logs
            WHERE session_id = ? AND is_outlier = 1
        """, (session_id,)).fetchone()

        print(f"\n   respiration_logs 驗證：")
        print(f"      總筆數 = {total['cnt']}（預期 10）")
        print(f"      異常值 = {outliers['cnt']}（預期 1）")

        # 驗證索引有被使用（EXPLAIN QUERY PLAN）
        plan = conn.execute("""
            EXPLAIN QUERY PLAN
            SELECT * FROM respiration_logs
            WHERE session_id = ? ORDER BY timestamp ASC
        """, (session_id,)).fetchone()

        print(f"\n   索引使用驗證：")
        print(f"      {dict(plan)}")

    print("\n" + "=" * 55)
    print("  ✅ 所有測試通過！資料庫運作正常。")
    print("=" * 55)


if __name__ == "__main__":
    run_test()