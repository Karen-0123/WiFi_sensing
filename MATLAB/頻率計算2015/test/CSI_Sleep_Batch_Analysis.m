% ===== 針對所有 seg 區段檔案的連續睡眠監測 =====
clear; clc; close all;

% 1. 環境設定與參數初始化
% 設定資料路徑
data_folder = 'C:\Users\fupei\Desktop\csi\data\sleep\sleep003_200hz_390min_0426';
Fs_orig = 200;
Fs_target = 40; 
my_filename = 'subject003_features.csv'; % ML input

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
all_state           = cell(num_files, 1); % 每個 Epoch 的 Sleep/Awake 結果
rollover_events     = cell(num_files, 1); % 每個 Epoch 的翻身事件數
processed_success   = false(1, num_files);
is_valid            = false(1, num_files); % 標記有效記錄 (非空檔且 num_events 不為空)

set(0, 'DefaultFigureVisible', 'off'); % 迴圈中不顯示圖像以加速處理
fprintf('開始處理 %d 個檔案區段 (加入體動偵測與訊號處理)...\n', num_files);

%% 2. 核心訊號處理迴圈
for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);
    
    % [需求 1] 1. 空檔檢查 (檔案大小為0則跳過)
    file_info = dir(filename);
    if isempty(file_info) || file_info.bytes == 0
        fprintf('[跳過記錄] 第 %d 個檔案 (%s) 為 0KB 空檔。\n', i, file_list(i).name);
        current_offset = current_offset + 180; % 增加預設區段時間 (180秒)
        continue;
    end
    
    try
        % 讀取 Intel 5300 CSI 原始數據
        [csi_matrix, timestamp_sec, ~] = read_intel5300_dat(filename);
        
        % 抗混疊低通濾波與均勻重採樣 (獨立模組)
        [csi_matrix, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);
        
        % 訊號預處理：計算幅度 (Amplitude) 與相位 (Phase)
        [amp_f, phase_f] = process_csi_signal(csi_matrix);
        
        % ===== Sleep / Awake 判斷 =====
        % 使用 PC1 振幅進行翻身偵測
        [num_events, events, ~] = detect_rollover(amp_f, Fs_target);
        
        % [需求 1] 2. num_events 為空檢查 (若為空則跳過)
        if isempty(num_events)
            fprintf('[跳過記錄] 第 %d 個檔案 (%s) 的 num_events 為空。\n', i, file_list(i).name);
            current_offset = current_offset + 180;
            continue;
        end
        
        [state, ~] = detect_sleep_state(events, 180);
        rollover_events{i} = num_events;
        all_state{i} = state;
        
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
        is_valid(i)          = true; % 標記為成功寫入的有效記錄
        
        clear csi_matrix amp_f phase_f best_sig;
    catch ME
        fprintf('警告：處理檔案 %s 時發生錯誤，跳過該區段。\n', file_list(i).name);
        fprintf('錯誤原因: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('錯誤發生在第 %d 行\n', ME.stack(1).line);
        end
        current_offset = current_offset + 180;
        continue;
    end
end
set(0, 'DefaultFigureVisible', 'on');

% 迴圈結束後，一次性拉平展平
all_bpm  = [all_bpm_cell{:}];
all_time = [all_time_cell{:}];

%% 3. 特徵提取與統計分析

% 3.1 計算呼吸變異度 
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 180, 180, num_files);

% 3.2 計算呼吸頻率偏差 (BPM Deviation)
baseline_bpm = calculate_nrem_baseline(all_bpm); 
bpm_deviation = abs(all_90th_percentile - baseline_bpm);

% 3.3 [修正需求 2] 使用 FIFO 佇列進行被跳過記錄與有效記錄的 var_history 值交換
var_history_adjusted = var_history;
var_queue = []; % FIFO 佇列

for i = 1:num_files
    if ~is_valid(i)
        % 若該筆記錄被跳過，將其 var_history 存入佇列尾端
        if ~isnan(var_history(i))
            var_queue(end+1) = var_history(i);
        end
    else
        % 若該筆記錄有效，且佇列中有先前被跳過的值
        if ~isempty(var_queue)
            % 1. 取出佇列中最舊的值 (FIFO 首端)
            oldest_var = var_queue(1);
            var_queue(1) = []; % 出列 (Pop)
            
            % 2. 進行值交換：把當前有效記錄的值放入佇列尾端，並將最舊的值賦給當前記錄
            var_queue(end+1) = var_history_adjusted(i);
            var_history_adjusted(i) = oldest_var;
        end
    end
end

% 3.4 [需求 1] 根據 is_valid 篩選出有效資料列，準備匯出至 CSV
bpm_deviation_final   = bpm_deviation(is_valid);
var_history_final     = var_history_adjusted(is_valid);
rollover_events_final = rollover_events(is_valid);
seg_start_times_final = seg_start_times(is_valid);
seg_end_times_final   = seg_end_times(is_valid);
wake_sleep_final      = all_state(is_valid);

% 3.5 建立 Table 與 CSV 匯出
featureTable = export_sleep_features(bpm_deviation_final, ...
                                     var_history_final, ...
                                     rollover_events_final, ...
                                     my_filename, ...
                                     seg_start_times_final, ...
                                     seg_end_times_final, ...
                                     wake_sleep_final);
                                 
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
