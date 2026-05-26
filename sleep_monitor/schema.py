#schema.py
from database import get_db

# ──────────────────────────────────────────
# 表一：users（使用者）
# ──────────────────────────────────────────
SQL_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(255) NOT NULL UNIQUE,      -- 登入帳號，不可重複
    display_name    VARCHAR(255),                      -- 顯示名稱（可為空）
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    wake_preference TEXT                               -- JSON 字串，例如 {"window_min": 30}
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ──────────────────────────────────────────
# 表二：sleep_summaries（睡眠紀錄總表）
# ──────────────────────────────────────────
SQL_CREATE_SLEEP_SUMMARIES = """
CREATE TABLE IF NOT EXISTS sleep_summaries (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    user_id              INT NOT NULL,
    date                 DATE NOT NULL,          -- 入睡當天日期，格式 YYYY-MM-DD
    started_at           DATETIME,                  -- 開始監測時間
    ended_at             DATETIME,                  -- 結束監測時間
    sleep_score          DECIMAL(5,2),              -- 睡眠分數 0–100
    avg_respiration_rate DECIMAL(5,2),              -- 平均呼吸率（次/分鐘）
    deep_sleep_minutes   INT,                       -- 深睡時間（分鐘）
    light_sleep_minutes  INT,                       -- 淺睡時間（分鐘）
    rem_sleep_minutes    INT,                       -- REM 時間（分鐘）
    status               VARCHAR(50) DEFAULT 'recording', -- recording / analyzing / done
    reward_points        INT DEFAULT 0,          -- 獎勵積分
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 外鍵：user_id 必須存在於 users.id
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ──────────────────────────────────────────
# 表三：respiration_logs（時序細節表）
# ──────────────────────────────────────────
SQL_CREATE_RESPIRATION_LOGS = """
CREATE TABLE IF NOT EXISTS respiration_logs (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    session_id       INT NOT NULL,
    timestamp        DATETIME NOT NULL,             -- 該筆數據的時間點
    respiration_rate DECIMAL(5,2),                  -- 呼吸率（次/分鐘，正常 12–20）
    signal_quality   DECIMAL(4,2),                  -- CSI 訊號品質（0–1）
    inferred_stage   VARCHAR(50) ,                  -- 推算睡眠階段：deep/light/rem/awake
    is_outlier       INT DEFAULT 0,                 -- 是否為異常值（0=否, 1=是）
    motion_detected  INT DEFAULT 0,                 -- 是否偵測到移動（0=否, 1=是）

    -- 外鍵：session_id 必須存在於 sleep_summaries.id
    FOREIGN KEY (session_id) REFERENCES sleep_summaries(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ──────────────────────────────────────────
# 索引：加速以 session_id + timestamp 查詢
# ──────────────────────────────────────────
SQL_CREATE_INDEX = """
CREATE INDEX idx_respiration_session_time
    ON respiration_logs (session_id, timestamp);
"""


def init_db():
    """
    依序建立三張表與索引。
    """
    with get_db() as cursor:
        # 建立三張表
        cursor.execute(SQL_CREATE_USERS)
        cursor.execute(SQL_CREATE_SLEEP_SUMMARIES)
        cursor.execute(SQL_CREATE_RESPIRATION_LOGS)

        # 檢查並建立索引（MySQL 如果重複建立索引會報錯，所以先用 try 包起來）
        try:
            cursor.execute(SQL_CREATE_INDEX)
        except Exception:
            pass  # 如果索引已經存在，就跳過

    print("MySQL 資料庫初始化完成！已建立以下結構：")
    print("   - 表：users")
    print("   - 表：sleep_summaries")
    print("   - 表：respiration_logs")
    print("   - 索引：idx_respiration_session_time (session_id, timestamp)")


def verify_schema():
    """
    初始化後驗證用：列出資料庫中所有的表，確認結構正確。
    """
    with get_db() as cursor:
        # 查詢 MySQL 裡面的所有資料表
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        print("\n目前 MySQL 資料庫內的表格：")
        for row in tables:
            # pymysql 回傳的是字典，其鍵值會是 'Tables_in_資料庫名'
            table_name = list(row.values())[0]
            print(f" table] {table_name}")


if __name__ == "__main__":
    init_db()
    verify_schema()