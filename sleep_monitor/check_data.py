import pymysql

try:
    print("正在嘗試跨海連線至 Aiven 雲端資料庫...")
    
    #  連線資訊（已換上妳們最新通電的 MySQL native password 新密碼）
    db = pymysql.connect(
        host='mysql-46cb3ab-ntou-project.h.aivencloud.com',
        port=21225,
        user='avnadmin',
        password='AVNS_kegvXqQywhPKN1Xr4Yp',  # 最新修改的那個密碼
        database='defaultdb',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    cursor = db.cursor()
    
    # 同時抓取最新使用者與時序呼吸率數據
    print("連線成功！正在撈取 Aiven 數據庫內容...\n")
    
    # 1. 查最新的使用者
    cursor.execute("SELECT id, username, display_name FROM users ORDER BY id DESC LIMIT 3;")
    users = cursor.fetchall()
    print("=== [users 表] 最新註冊用戶 ===")
    for u in users:
        print(f"ID: {u['id']} | 帳號: {u['username']} | 暱稱: {u['display_name']}")
        
    # 2. 查最新的時序數據
    cursor.execute("SELECT id, session_id, timestamp, respiration_rate, inferred_stage FROM respiration_logs ORDER BY id DESC LIMIT 5;")
    logs = cursor.fetchall()
    print("\n=== [respiration_logs 表] 最新 5 筆時序數據 ===")
    for row in logs:
        print(f"ID: {row['id']} | Session: {row['session_id']} | 時間: {row['timestamp']} | 呼吸率: {row['respiration_rate']} BPM | 階段: {row['inferred_stage']}")
        
    db.close()
    print(" 資料驗證完畢！雲端數據確實存在且完美運作中！")
   

except Exception as e:
    print(f"讀取失敗，原因: {e}")