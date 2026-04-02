
clear; clc; close all;

folder = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\data\breathe\down\csi_breathe_30down001\';

file_list = {
    'real_breathe_20260328_152130_seg1.dat',
    'real_breathe_20260328_152230_seg2.dat',
    'real_breathe_20260328_152330_seg3.dat'
};

% 初始化總矩陣
csi_matrix = []; 

% --- 步驟 1: 循環讀取與資料接龍 ---
fprintf('步驟 1: 正在整合 3 段分段資料...\n');
for i = 1:length(file_list)
    full_path = [folder, file_list{i}];
    
    if exist(full_path, 'file')
        fprintf('  >> 正在讀取第 %d 段: %s\n', i, file_list{i});
        [temp_matrix, ~, ~] = read_intel5300_dat(full_path);
        csi_matrix = [csi_matrix; temp_matrix]; 
    else
        warning('找不到檔案: %s，跳過此段。', full_path);
    end
end

if isempty(csi_matrix)
    error('所有檔案讀取失敗，請檢查路徑與工具包！');
end

fprintf('整合完成！總封包數量: %d (約 %.1f 秒)\n', size(csi_matrix, 1), size(csi_matrix, 1)/200);

% 3. 訊號處理：使用整合好的 csi_matrix
fprintf('步驟 2: 處理訊號 (消除相位偏移與濾波)...\n');
[amp_filtered, phase_filtered] = process_csi_signal(csi_matrix);

% 4. 頻譜分析與最佳特徵選擇
fprintf('步驟 4: 執行帶通濾波、頻譜分析與特徵選擇...\n');
[best_name, best_sig, ~] = select_respiration_stream(amp_filtered, phase_filtered);

% 5. 精細峰值偵測與視覺化
fprintf('步驟 5: 執行精細峰值偵測 (Window-based Validation)...\n');
[peak_indices, peak_values] = detect_respiration_peaks(best_sig);

% 7. 動態 BPM 計算與繪圖 
fprintf('步驟 6: 計算動態呼吸率 (BPM)...\n');
total_samples = length(best_sig);
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(peak_indices, total_samples);

fprintf('所有流程執行完畢！\n');