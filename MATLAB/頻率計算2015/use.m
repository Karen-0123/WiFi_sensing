% ===== CSI 處理主程式 (main.m) =====
clear; clc; close all;

% 1. 設定你的 .dat 檔案路徑 (請替換為你實際的檔案名稱)
filename = 'D:\大學資料\WiFi_sensing\data\test\nothing_right_left_up_down_002.dat'; 
%filename = 'C:\Users\fupei\Desktop\csi\data\test\nothing_right_left_up_down_002.dat'; 

% 2. 讀取與過濾原始資料
% 使用我們寫好的 get_clean_csi_5300 函式
fprintf('步驟 1: 讀取資料...\n');
[csi_matrix, timestamp, rssi] = read_intel5300_dat(filename);

% 3. 訊號處理：共軛相乘與 SG 濾波
% 將剛剛得到的 csi_matrix 餵給 process_csi_signal
fprintf('步驟 2: 處理訊號 (消除相位偏移與濾波)...\n');
[amp_filtered, phase_filtered] = process_csi_signal(csi_matrix);

% 4. 頻譜分析與最佳特徵選擇
fprintf('步驟 4: 執行帶通濾波、頻譜分析與特徵選擇...\n');
[best_name, best_sig, breathing_rate_hz] = select_respiration_stream(amp_filtered, phase_filtered);

% 5. 精細峰值偵測與視覺化 (新增的步驟)
fprintf('步驟 5: 執行精細峰值偵測 (Window-based Validation)...\n');
[peak_indices, peak_values] = detect_respiration_peaks(best_sig);

% 7. 動態 BPM 計算與繪圖 (最終步驟)
fprintf('步驟 6: 計算動態呼吸率 (BPM)...\n');
total_samples = length(best_sig);
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(peak_indices, total_samples);

% 8. 執行滑動視窗睡眠分期 (新增)
[stage_history, stage_time] = classify_sleep_timeline(bpm_timeline, time_axis_bpm);

fprintf('======================================\n');
fprintf('所有流程執行完畢！\n');