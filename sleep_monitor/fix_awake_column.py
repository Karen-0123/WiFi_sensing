import pymysql

try:
    print("正在連線至 Aiven 雲端資料庫修正資料表結構...")
    
    # 🔑 使用最新 SSL 與 Native 密碼連線
    db = pymysql.connect(
        host="mysql-46cb3ab-ntou-project.h.aivencloud.com",
        port=21225,
        user="avnadmin",
        password="AVNS_kegvXqQywhPKN1Xr4Yp",  # 🎯 妳最新生成的原生密碼
        database="defaultdb",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl_mode": "REQUIRED"}
    )
    
    cursor = db.cursor()
    
    print("連線成功！正在 `sleep_summaries` 表中追加 `awake_minutes` 欄位...")
    
    #  正統 SQL 指令：在 rem_sleep_minutes 後面，追加 awake_minutes 整數欄位
    sql_command = "ALTER TABLE `sleep_summaries` ADD `awake_minutes` INT NULL AFTER `rem_sleep_minutes`;"
    cursor.execute(sql_command)
    
    db.commit()
    db.close()
    
    print("====================================================")
    print(" 成功！`awake_minutes` 欄位已堂堂正正加入 Aiven 雲端總表！")
    print("====================================================")

except Exception as e:
    print(f" 欄位追加失敗，原因可能為已存在或連線有誤: {e}")