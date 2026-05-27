import pymysql

try:
    print("正在嘗試跨海連線至 Aiven 雲端資料庫...")
    
    # 🔑 1. 注入截圖中 100% 正確的 Aiven 連線資訊
    db = pymysql.connect(
        host='mysql-46cb3ab-ntou-project.h.aivencloud.com',
        port=21225,
        user='avnadmin',
        password='AVNS_kegvXqQywhPKN1Xr4Yp', # 密碼完全一致
        database='defaultdb',
        charset='utf8mb4'
    )
    
    cursor = db.cursor()
    
    print("連線成功！正在強制注入 `password` 欄位...")
    
    # 🎯 2. 執行正統 SQL：在 username 欄位後面追加長度 255 的 password 欄位
    sql_command = "ALTER TABLE `users` ADD `password` VARCHAR(255) NOT NULL AFTER `username`;"
    cursor.execute(sql_command)
    
    # 💾 3. 提交變更並關閉連線
    db.commit()
    db.close()
    
    print("====================================================")
    print("✅ 恭喜！`password` 欄位已成功堂堂正正加入 Aiven 雲端資料表！")
    print("====================================================")

except Exception as e:
    print(f"❌ 注入失敗，錯誤原因: {e}")