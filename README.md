# WiFi_sensing

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
