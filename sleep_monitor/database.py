# database.py
import sqlite3
import os
from contextlib import contextmanager

# 資料庫檔案路徑
DB_PATH = os.path.join(os.path.dirname(__file__), "sleep.db")


def get_connection() -> sqlite3.Connection:
    """
    建立並回傳一個 SQLite 連線物件。
    - detect_types：讓 sqlite3 自動將欄位轉換成 Python 原生型別（如 datetime）
    - row_factory：讓查詢結果可以用欄位名稱存取，而不是只能用 index
    """
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )
    conn.row_factory = sqlite3.Row  # 結果可用 row["欄位名"] 存取
    conn.execute("PRAGMA foreign_keys = ON")  # 啟用外鍵約束（SQLite 預設關閉）
    return conn


@contextmanager
def get_db():
    """
    Context manager 版本的連線管理器。
    用法：
        with get_db() as conn:
            conn.execute(...)

    - 正常結束時自動 commit
    - 發生例外時自動 rollback，並重新拋出錯誤
    - 無論如何都會關閉連線，避免資源洩漏
    """
    conn = get_connection()
    try:
        yield conn          # 把連線交給 with 區塊使用
        conn.commit()       # 區塊正常結束 → 提交變更
    except Exception as e:
        conn.rollback()     # 發生錯誤 → 回滾所有變更
        raise e             # 重新拋出例外，讓上層知道出錯了
    finally:
        conn.close()        # 無論成功或失敗，都關閉連線


def check_connection():
    """
    確認資料庫連線是否正常。
    印出 SQLite 版本與 DB 檔案路徑。
    """
    try:
        with get_db() as conn:
            version = conn.execute("SELECT sqlite_version()").fetchone()[0]
            print(f"✅ 連線成功！")
            print(f"   SQLite 版本：{version}")
            print(f"   資料庫路徑：{DB_PATH}")
    except Exception as e:
        print(f"❌ 連線失敗：{e}")


if __name__ == "__main__":
    check_connection()