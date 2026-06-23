import os
import pymysql
from datetime import datetime, timedelta
from crud import create_session

# ── 🎯 註冊帳號對齊設定 ──────────────────────────────────────────────────
TARGET_EMAIL = "test_user1@gmail.com"  # 👈 寧！如果註冊的是別的帳號記得改這行！

# ── 🚀 自動化路徑設定 ──────────────────────────────────────────────────
MATLAB_OUTPUT_PATH = (
    r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\MATLAB_output\real_breathing_output.csv"
    if os.path.exists(r"C:\Users\Admin\OneDrive\桌面\wifi_sensing\WiFi_sensing\MATLAB_output\real_breathing_output.csv")
    else r"C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\Frequency_Calculation_2015\sleep004_200hz_390min_0427\real_breathing_output.csv"
)

# ── 🔑 雲端安全連線機制 ────────────────────────────────────────────────
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

def run_import():
    print("   WiFi CSI 睡眠監測系統 — 裝甲火箭批次寫入完全體 (1秒通電版)")

    if not os.path.exists(MATLAB_OUTPUT_PATH):
        print("錯誤：找不到真實數據 CSV 檔案！")
        return

    # ── 1. 確認/建立使用者 ──────────────────────────────────────────
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
        print(f"階段 1 失敗: {e}")
        return

    # ── 2. 建立睡眠 Session ────────────────────────────────────────
    started_at = datetime.now() - timedelta(hours=8)
    session_id = create_session(user_id=user_id, started_at=started_at, date=started_at.strftime("%Y-%m-%d"))
    print(f" 成功開啟 Session, ID = {session_id}")

    # ── 3. 讀取並打包 MATLAB 真實數據 (開卡車準備一次載走) ──────────────────
    print(f"\n【階段 3】正在一次性打包 601 筆數據...")
    bulk_data = []
    count = 0
    stage_mapping = {0: "deep", 1: "light", 2: "rem", 3: "awake"}
    stage_counts = {"deep": 0, "light": 0, "rem": 0, "awake": 0}
    total_rr = 0.0

    with open(MATLAB_OUTPUT_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line: continue
            try:
                parts = line.split(',')
                rr, quality, stage_code, motion = float(parts[0]), float(parts[1]), int(float(parts[2])), int(float(parts[3]))
                if rr <= 0 or rr != rr: continue

                stage_name = stage_mapping.get(stage_code, "light")
                timestamp = started_at + timedelta(seconds=30 * count)

                # 把資料塞進大卡車（Tuple 陣列）
                bulk_data.append((session_id, timestamp, rr, quality, stage_name, motion))
                stage_counts[stage_name] += 1
                total_rr += rr
                count += 1
            except Exception:
                continue

    # ── 4. 火箭發射：利用 executemany 1秒送進雲端 ──────────────────────
    if count > 0:
        try:
            conn = get_db_secure()
            with conn.cursor() as cursor:
                # 🎯 核心黑科技：用卡車一次倒進去
                print(f"  將 {count} 筆數據一次性砸入 Aiven...")
                insert_sql = """
                    INSERT INTO respiration_logs (session_id, timestamp, respiration_rate, signal_quality, inferred_stage, motion_detected)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_sql, bulk_data)
                
                # ── 5. 正統結算 (包含剛追加成功的 awake_minutes 欄位) ──────────────────
                print(f"\n【階段 4】數據載入完成！正在同步進行正統結構結算...")
                deep_min = int(stage_counts['deep'] / 2)
                light_min = int(stage_counts['light'] / 2)
                rem_min = int(stage_counts['rem'] / 2)
                awake_min = int(stage_counts['awake'] / 2)
                avg_rr = round(total_rr / count, 2)
                real_score = round(6.0 + (deep_min / (deep_min + light_min + rem_min + awake_min + 1) * 4.0), 1)
                if real_score > 10.0: real_score = 9.5

                update_sql = """
                    UPDATE sleep_summaries 
                    SET sleep_score = %s, avg_respiration_rate = %s, deep_sleep_minutes = %s, 
                        light_sleep_minutes = %s, rem_sleep_minutes = %s, awake_minutes = %s,
                        ended_at = %s, status = 'done' 
                    WHERE id = %s
                """
                cursor.execute(update_sql, (real_score, avg_rr, deep_min, light_min, rem_min, awake_min, datetime.now(), session_id))
                
            conn.commit()
            conn.close()

            print("\n" + "=" * 65)
            print(f"      帳號             : {TARGET_EMAIL}")
            print(f"      網頁相容睡眠分數 : {real_score} 分 (大圓環已徹底甦醒！)")
            print("=" * 65)
            print("    全自動串接且前端優化完畢！請直接去網頁重新整理看成果！")
            print("=" * 65)

        except Exception as e:
            print(f"寫入或結算失敗: {e}")
    else:
        print(" 錯誤：沒有可匯入的有效數據。")

if __name__ == "__main__":
    run_import()