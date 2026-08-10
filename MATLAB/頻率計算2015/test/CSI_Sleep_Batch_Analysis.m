% ===== 針對所有 seg 區段檔案的連續睡眠監測 =====
clear; clc; close all;

% 1. 環境設定與參數初始化
% 設定資料路徑
data_folder = 'D:\大學資料\sleep_dataset\sleep014_200hz_360min_0804';
Fs_orig = 200;
Fs_target = 40; 
my_filename = 'subject014_features.csv'; % ML input

% 檢索資料夾內所有包含 'seg' 字眼的 .dat 檔案
file_pattern = fullfile(data_folder, '*seg*.dat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0, error('找不到指定的資料檔案！'); end

% 檔案排序：確保區段檔案按編號 (seg1, seg2...) 順序處理
seg_numbers = zeros(1, num_files);
for k = 1:num_files
    tokens = regexp(file_list(k).name, 'seg(\d+)', 'tokens');
    if ~isempty(tokens), seg_numbers(k) = str2double(tokens{1}{1}); end
end
[~, sort_idx] = sort(seg_numbers);
file_list = file_list(sort_idx);

% 自動從檔名解析真實時間
seg_start_times = cell(num_files, 1);
seg_end_times   = cell(num_files, 1);

for k = 1:num_files
    % 正則匹配 yyyyMMdd_HHmmss，例如 20260705_030629
    tokens = regexp(file_list(k).name, '(\d{8})_(\d{6})', 'tokens');
    if ~isempty(tokens)
        date_str = tokens{1}{1}; % '20260705'
        time_str = tokens{1}{2}; % '030629'
        
        % 解析年月日時分秒
        yyyy = str2double(date_str(1:4));
        MM   = str2double(date_str(5:6));
        dd   = str2double(date_str(7:8));
        hh   = str2double(time_str(1:2));
        mm   = str2double(time_str(3:4));
        ss   = str2double(time_str(5:6));
        
        % 建立精準 datetime 物件
        t_start = datetime(yyyy, MM, dd, hh, mm, ss);
        t_end   = t_start + minutes(3); % 預設每段長度為 3 分鐘
        
        seg_start_times{k} = t_start;
        seg_end_times{k}   = t_end;
    else
        error('檔案名稱 [%s] 無法解析時間格式！', file_list(k).name);
    end
end

% 初始化全局變數
all_bpm_cell        = cell(1, num_files);
all_time_cell       = cell(1, num_files);
all_motion_flags = []; all_motion_time = [];
current_offset = 0; % 時間偏移量（秒）
all_90th_percentile = NaN(1, num_files); % 每個區段的 90th 百分位
processed_success   = false(1, num_files);

set(0, 'DefaultFigureVisible', 'off'); % 迴圈中不顯示圖像以加速處理
fprintf('開始處理 %d 個檔案區段 (加入體動偵測與訊號處理)...\n', num_files);

%% 2. 核心訊號處理迴圈
for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);
    
    % 空檔檢查
    file_info = dir(filename);
    if isempty(file_info) || file_info.bytes == 0
        fprintf('[預防跳過] 第 %d 個檔案 (%s) 為 0KB 空檔，自動標記為 NaN。\n', i, file_list(i).name);
        current_offset = current_offset + (total_samples / Fs_target);
        continue; % 第 i 位置預設就是 NaN，直接進入下一個檔案
    end
    
    try
        % 讀取 Intel 5300 CSI 原始數據
        [csi_matrix, timestamp_sec, ~] = read_intel5300_dat(filename);
        
        % 抗混疊低通濾波與均勻重採樣 (獨立模組) (csi_resampled : [N_uniform,30,2,3])
        [csi_matrix, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);
        
        % 訊號預處理：計算幅度 (Amplitude) 與相位 (Phase)
        [amp_f, phase_f] = process_csi_signal(csi_matrix);
        
        % 串流選擇：挑選呼吸特徵最明顯的子載波 (Subcarrier)
        [~, best_sig, ~] = select_respiration_stream(amp_f, phase_f, Fs_target);
        
        % 呼吸峰值檢測
        [peak_idx, ~] = detect_respiration_peaks(best_sig, gap_mask, Fs_target);
        
        % 計算動態呼吸率 (BPM)
        total_samples = length(best_sig);
        [seg_90th, bpm_seg, time_seg] = calculate_dynamic_bpm(peak_idx, total_samples, gap_mask, Fs_target);
        
        % 合併數據：將當前區段結果加入全局陣列
        all_bpm_cell{i}        = bpm_seg;
        all_time_cell{i}       = time_seg + current_offset;
        all_90th_percentile(i) = seg_90th;
        
        % 更新下一區段的起始時間偏移
        current_offset = current_offset + (total_samples / Fs_target);
        processed_success(i) = true;
        clear csi_matrix amp_f phase_f best_sig;
    catch ME
        fprintf('警告：處理檔案 %s 時發生錯誤，跳過該區段。\n', file_list(i).name);
        fprintf('錯誤原因: %s\n', ME.message);
        fprintf('錯誤發生在第 %d 行\n', ME.stack(1).line);
        
        current_offset = current_offset + seg_duration;
    end
end
set(0, 'DefaultFigureVisible', 'on');

% 迴圈結束後，一次性拉平展平
all_bpm  = [all_bpm_cell{:}];
all_time = [all_time_cell{:}];

%% 3. 特徵提取與統計分析

% 3.1 計算呼吸變異度 
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 180, 180, num_files); % 每 180 秒為一個 Epoch，步長也是 180 秒 (無重疊)

% 3.2 計算呼吸頻率偏差 (BPM Deviation)
baseline_bpm = calculate_nrem_baseline(all_bpm); % 1. 計算基線
bpm_deviation = abs(all_90th_percentile - baseline_bpm); % 2. 計算偏差

% 3.3 建立 Table 與 CSV 匯出
record_start = datetime(2026, 7, 27, 23, 0, 0); 
featureTable = export_sleep_features(bpm_deviation, var_history, my_filename, seg_start_times, seg_end_times);

% 建議在 MATLAB 算完特徵後，直接平鋪（Flatten）導出為標準 .csv 檔案（或是以 Pandas 載入的 DataFrame 格式）。
% 欄位結構設計：
% [subject_id, record_id, epoch_id, resp_deviation, resp_var, stage1_label, psg_groundtruth]

% % 專題新增：將真實運作數據自動匯出為CSV 檔
% 
% fprintf('\n【資料庫連接階段】正在產生資料庫匯入檔...\n');
% 
% % 1. 取得演算法預測結果的基準長度（通常 sleep_stages 最短，用它當作 N）
% N = length(sleep_stages); 
% 
% % 2. 安全切齊所有長度，並利用 (:) 強制轉為直向行向量 (Column Vector)
% col_bpm     = all_bpm(1:N);
% col_bpm     = col_bpm(:); 
% 
% col_stages  = sleep_stages(1:N);
% col_stages  = col_stages(:);
% 
% col_motion  = all_motion_flags(1:N);
% col_motion  = col_motion(:);
% 
% % 訊號品質預設填 1.0 (之後有需要可以串接子載波的信噪比或訊號強度)
% col_quality = ones(N, 1); 
% 
% % 3. 將四個直向向量結合成一個 N x 4 的大型數據矩陣
% output_matrix = [col_bpm, col_quality, col_stages, col_motion];
% 
% % 4. 將資料夾路徑與檔名結合，確保 CSV 輸出在跟資料檔案同一個位置，方便 Python 讀取
% output_csv_path = fullfile(data_folder, 'real_breathing_output.csv');
% 
% % 5. 寫入 CSV 檔案
% csvwrite(output_csv_path, output_matrix);
% 
% fprintf('真實監測數據已成功轉置！\n');
% fprintf('檔案位置: %s\n', output_csv_path);
% fprintf('提示：現Python 腳本讀取這個 CSV 檔案匯入 sleep.db 囉！\n');
