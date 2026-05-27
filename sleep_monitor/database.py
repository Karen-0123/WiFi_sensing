#database.py
import pymysql
from contextlib import contextmanager

# MySQL 連線設定
DB_CONFIG = {
    "host": "mysql-46cb3ab-ntou-project.h.aivencloud.com",
    "port": 21225,
    "user": "avnadmin",
    "password": "AVNS_kegvXqQywhPKN1Xr4Yp",
    "database": "defaultdb",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "ssl": {"ssl_mode": "REQUIRED"}  
}


def get_connection():
    """
    建立並回傳一個 MySQL 連線物件。
    """
    conn = pymysql.connect(**DB_CONFIG)
    return conn


@contextmanager
def get_db():
    """
    Context manager 版本的連線管理器。
    用法跟原本完全一樣：
        with get_db() as cursor:
            cursor.execute(...)
    """
    conn = get_connection()
    cursor = conn.cursor()  # 建立游標來執行 SQL
    try:
        yield cursor        # 把游標交給 with 區塊使用
        conn.commit()       # 區塊正常結束 → 提交變更
    except Exception as e:
        conn.rollback()     # 發生錯誤 → 回滾所有變更
        raise e
    finally:
        cursor.close()      # 關閉游標
        conn.close()        # 關閉連線


def check_connection():
    """
    確認 MySQL 資料庫連線是否正常。
    """
    try:
        with get_db() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()["VERSION()"]
            print(f"   MySQL 連線成功！")
            print(f"   MySQL 版本：{version}")
            print(f"   目前使用的資料庫：{DB_CONFIG['database']}")
    except Exception as e:
        print(f" MySQL 連線失敗，請檢查 XAMPP 是否有亮綠燈：{e}")


if __name__ == "__main__":
    check_connection()