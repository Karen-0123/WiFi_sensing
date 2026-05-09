# schema.py
from database import get_db

# ──────────────────────────────────────────
# 表一：users（使用者）
# ──────────────────────────────────────────
SQL_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    NOT NULL UNIQUE,          -- 登入帳號，不可重複
    display_name TEXT,                             -- 顯示名稱（可為空）
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    wake_preference TEXT                           -- JSON 字串，例如 {"window_min": 30}
);
"""

# ──────────────────────────────────────────
# 表二：sleep_summaries（睡眠紀錄總表）
# ──────────────────────────────────────────
SQL_CREATE_SLEEP_SUMMARIES = """
CREATE TABLE IF NOT EXISTS sleep_summaries (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL,
    date                 DATE    NOT NULL,          -- 入睡當天日期，格式 YYYY-MM-DD
    started_at           DATETIME,                  -- 開始監測時間
    ended_at             DATETIME,                  -- 結束監測時間
    sleep_score          REAL,                      -- 睡眠分數 0–100
    avg_respiration_rate REAL,                      -- 平均呼吸率（次/分鐘）
    deep_sleep_minutes   INTEGER,                   -- 深睡時間（分鐘）
    light_sleep_minutes  INTEGER,                   -- 淺睡時間（分鐘）
    rem_sleep_minutes    INTEGER,                   -- REM 時間（分鐘）
    status               TEXT DEFAULT 'recording', -- recording / analyzing / done
    reward_points        INTEGER DEFAULT 0,         -- 獎勵積分
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 外鍵：user_id 必須存在於 users.id
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

# ──────────────────────────────────────────
# 表三：respiration_logs（時序細節表）
# ──────────────────────────────────────────
SQL_CREATE_RESPIRATION_LOGS = """
CREATE TABLE IF NOT EXISTS respiration_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    timestamp        DATETIME NOT NULL,             -- 該筆數據的時間點
    respiration_rate REAL,                          -- 呼吸率（次/分鐘，正常 12–20）
    signal_quality   REAL,                          -- CSI 訊號品質（0–1）
    inferred_stage   TEXT,                          -- 推算睡眠階段：deep/light/rem/awake
    is_outlier       INTEGER DEFAULT 0,             -- 是否為異常值（0=否, 1=是）
    motion_detected  INTEGER DEFAULT 0,             -- 是否偵測到移動（0=否, 1=是）

    -- 外鍵：session_id 必須存在於 sleep_summaries.id
    FOREIGN KEY (session_id) REFERENCES sleep_summaries(id) ON DELETE CASCADE
);
"""

# ──────────────────────────────────────────
# 索引：加速以 session_id + timestamp 查詢
# ──────────────────────────────────────────
SQL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_respiration_session_time
    ON respiration_logs (session_id, timestamp);
"""

# ──────────────────────────────────────────
# 一鍵初始化資料庫
# ──────────────────────────────────────────
def init_db():
    """
    依序建立三張表與索引。
    使用 CREATE TABLE IF NOT EXISTS，重複執行不會出錯。
    """
    with get_db() as conn:
        # 建立三張表
        conn.execute(SQL_CREATE_USERS)
        conn.execute(SQL_CREATE_SLEEP_SUMMARIES)
        conn.execute(SQL_CREATE_RESPIRATION_LOGS)

        # 建立複合索引
        conn.execute(SQL_CREATE_INDEX)

    print("✅ 資料庫初始化完成！已建立以下結構：")
    print("   - 表：users")
    print("   - 表：sleep_summaries")
    print("   - 表：respiration_logs")
    print("   - 索引：idx_respiration_session_time (session_id, timestamp)")


def verify_schema():
    """
    初始化後驗證用：列出資料庫中所有的表與索引，確認結構正確。
    """
    with get_db() as conn:
        # 查詢所有使用者建立的表
        tables = conn.execute("""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'index')
              AND name NOT LIKE 'sqlite_%'  -- 排除 SQLite 內建系統表
            ORDER BY type, name;
        """).fetchall()

        print("\n📋 目前資料庫結構：")
        for row in tables:
            icon = "📁" if row["type"] == "table" else "🔍"
            print(f"   {icon} [{row['type']}] {row['name']}")


# 直接執行此檔案時，進行初始化與驗證
if __name__ == "__main__":
    init_db()
    verify_schema()