# WiFi_sensing
前後端資料模型與核心 API 規格：
前端動作                  傳入模型               回傳模型
─────────────────────────────────────────────────────────
新增使用者           →   UserCreate         →   UserResponse
建立睡眠 Session     →   SessionCreate      →   SessionResponse
新增呼吸數據         →   RespirationLogCreate→  RespirationLogResponse
查詢圖表資料         →   (只需 session_id)  →   SessionChartData
操作成功通知         →                      →   SuccessResponse
操作失敗通知         →                      →   ErrorResponse

函數                        用途                      回傳
─────────────────────────────────────────────────────────────────
create_session()         建立新睡眠紀錄              session_id (int)
append_respiration_log() 新增一筆 MATLAB 呼吸數據    log_id (int)
close_session()          結算分數、更新 summary      結算結果 (dict)
get_session_chart_data() 取得前端圖表完整資料        SessionChartData

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
開發環境與技術棧 ：
硬體設備：Intel 5300 
後端演算法 / 自動化：Python 3.x / MATLAB Engine API for Python / OS Windows Samba
資料庫管理：SQLite 3
網頁前端與伺服器：PHP 8.x / HTML5 / CSS3 / JavaScript (Canvas / Chart.js 生理圖表繪製) / XAMPP Apache Server
公網穿透工具：ngrok CLI

部署與本地運行指南 (Installation & Running)
1. 環境準備
請確保本地已安裝 XAMPP（內含 Apache）以及安裝 ngrok 並綁定專屬 Authtoken。
2. 啟動網頁伺服器
-打開 XAMPP Control Panel，點擊 Apache -> Start（確認亮綠燈且 Port 80 正常運作）
-將本網頁專案資料夾置於 XAMPP 的 htdocs 目錄下。
3. 一鍵開啟 ngrok 安全外網隧道
在 VS Code 終端機或 CMD 執行以下指令，開闢對外傳送門：
Bash ngrok.exe http 80
啟動後，複製畫面上 Forwarding 欄位所生成的 https://xxxx.ngrok-free.dev 安全網址。
4. 跨裝置訪問與展示
在任何行動裝置（手機、iPad、外網筆電）的瀏覽器輸入拼接後的絕對路徑，即可進行專題展示：
！首次進入請點擊藍色「Visit Site」按鈕繞過安全提示頁面
https://armband-exact-multiply.ngrok-free.dev/wifi_sensing/WiFi_sensing/sleep_monitor/WebApp/index.html

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
統架構與資料流 (System Architecture)
```text
[ 數據採集端 ]  Intel 5300 網卡 (錄製睡眠 CSI 原始數據 .dat)
                     ↓  (跨作業系統傳輸)
[ 數據傳輸橋 ]  Samba 共享資料夾 (Samba Share Protocol)
                     ↓ 
[ 訊號處理核心] MATLAB Engine (訊號清洗、帶通濾波、呼吸特徵提取)
                     ↓
[ 生理分期預算] Python Preprocessing (數據清洗、睡眠四階段模型判定)
                     ↓
[ 資料儲存端 ]  SQLite 嵌入式資料庫 (本地持久化儲存 sleep.db)
                     ↓
[ 展示端 ]  Apache / PHP 網頁網誌 (運行於 XAMPP 環境)
                     ↓  (安全公網外網傳送門)
[ 雲端 Dashboard] ngrok 外網安全隧道 (全球可視化行動端 Dashboard)


