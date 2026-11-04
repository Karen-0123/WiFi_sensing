% ===== 針對所有 seg 區段檔案的連續睡眠監測 (基於《WiFi-Sleep》完整訊號處理與 13 維特徵工程) =====
clear; clc; close all;

% 1. 環境設定與參數初始化
% 設定資料路徑
data_folder = "C:\Users\fupei\Desktop\csi\data\sleep\sleep015_200hz_360min_0805";
Fs_orig = 200;
Fs_target = 40;
my_filename = 'subject015_features.csv'; % ML input

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
        date_str = tokens{1}{1};
        time_str = tokens{1}{2};

        yyyy = str2double(date_str(1:4));
        MM   = str2double(date_str(5:6));
        dd   = str2double(date_str(7:8));
        hh   = str2double(time_str(1:2));
        mm   = str2double(time_str(3:4));
        ss   = str2double(time_str(5:6));

        t_start = datetime(yyyy, MM, dd, hh, mm, ss);
        t_end   = t_start + minutes(3); % 預設每段長度為 3 分鐘

        seg_start_times{k} = t_start;
        seg_end_times{k}   = t_end;
    else
        error('檔案名稱 [%s] 無法解析時間格式！', file_list(k).name);
    end
end

% 初始化全局變數 (既有特徵 + 論文新增之強化呼吸與體動特徵)
all_bpm_cell        = cell(1, num_files);
all_time_cell       = cell(1, num_files);
current_offset      = 0; % 時間偏移量（秒）
all_90th_percentile = NaN(num_files, 1); % 每個區段的 90th 百分位
all_state           = cell(num_files, 1); % 每個 Epoch 的 Sleep/Awake 結果
rollover_events     = cell(num_files, 1); % 每個 Epoch 的翻身事件數 (MBM)
processed_success   = false(num_files, 1);

% 論文新增之跨區段強化特徵儲存陣列
resp_depth_var_arr  = NaN(num_files, 1);
fit_mean_arr        = NaN(num_files, 1);
fit_var_arr         = NaN(num_files, 1);
ie_ratio_mean_arr   = NaN(num_files, 1);
ie_ratio_var_arr    = NaN(num_files, 1);
mbm_freq_arr        = zeros(num_files, 1);
plm_freq_arr        = zeros(num_files, 1);

set(0, 'DefaultFigureVisible', 'off'); % 迴圈中不顯示圖像以加速處理
fprintf('開始處理 %d 個檔案區段 (基於 WiFi-Sleep CSI Ratio + MRC-PCA + Doppler-MUSIC + ACF)...\n', num_files);

%% 2. 核心訊號處理迴圈
for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);

    try
        % Step 1: 讀取 Intel 5300 CSI 原始數據
        [csi_matrix, timestamp_sec, ~] = read_intel5300_dat(filename);

        % Step 2: 抗混疊低通濾波與均勻重採樣
        [csi_matrix, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);

        % Step 3: 訊號預處理：CSI Ratio + MRC-PCA 呼吸增強與最佳 Tx 選擇
        [amp_f, phase_f, csi_ratio_complex, mrc_pca_details] = process_csi_signal(csi_matrix, Fs_target);

        % Step 4: Doppler-MUSIC 速度譜體動偵測 (區分 MBM 翻身與 PLM 微動)
        [num_events, events, ~, motion_info] = detect_rollover(csi_ratio_complex, Fs_target, 'GapMask', gap_mask);
        [state, ~] = detect_sleep_state(events, 180);
        rollover_events{i} = num_events;
        all_state{i}       = state;
        mbm_freq_arr(i)    = motion_info.mbm_frequency;
        plm_freq_arr(i)    = motion_info.plm_frequency;

        % Step 5: 串流選擇 (結合振幅與相位互補，並避開體動干擾區段評估 PSD)
        [~, best_sig, ~, best_sg_sig] = select_respiration_stream( ...
            amp_f, phase_f, Fs_target, motion_info.motion_mask, mrc_pca_details);

        % Step 6: 呼吸波峰與波谷檢測 + 提取強化呼吸波形特徵 (深度變異、FIT、I/E Ratio)
        [peak_idx, ~, ~, ~, waveform_features] = detect_respiration_peaks( ...
            best_sig, gap_mask, Fs_target, motion_info.motion_mask, best_sg_sig);

        resp_depth_var_arr(i) = waveform_features.resp_depth_var;
        fit_mean_arr(i)       = waveform_features.fit_mean;
        fit_var_arr(i)        = waveform_features.fit_var;
        ie_ratio_mean_arr(i)  = waveform_features.ie_ratio_mean;
        ie_ratio_var_arr(i)   = waveform_features.ie_ratio_var;

        % Step 7: 計算動態呼吸率 (ACF 自相關函數 + 體動區間切除與線性插值修補)
        total_samples = length(best_sig);
        [seg_90th, bpm_seg, time_seg] = calculate_dynamic_bpm( ...
            peak_idx, total_samples, gap_mask, Fs_target, best_sig, motion_info.motion_mask);

        % 合併數據：將當前區段結果加入全局陣列
        all_bpm_cell{i}        = bpm_seg;
        all_time_cell{i}       = time_seg + current_offset;
        all_90th_percentile(i) = seg_90th;

        % 更新下一區段的起始時間偏移
        current_offset = current_offset + 180;
        processed_success(i) = true;
    catch ME
        fprintf('警告：處理檔案 %s 時發生錯誤，跳過該區段。\n', file_list(i).name);
        fprintf('錯誤原因: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('錯誤發生在 %s 第 %d 行\n', ME.stack(1).name, ME.stack(1).line);
        end

        current_offset = current_offset + 180;
        continue;
    end
end
set(0, 'DefaultFigureVisible', 'on');

% 迴圈結束後，一次性拉平展平所有有效 BPM
all_bpm = [all_bpm_cell{:}];

%% 3. 特徵提取與整晚統計分析

% 3.1 整晚高階多項式去趨勢、每個 seg 的呼吸變異度與呼吸率統計特徵
[var_history, var_time, varience, rr_features] = calculate_breathing_variability(all_bpm_cell, all_time_cell, num_files);

% 3.2 整晚 NREM 呼吸基線與頻率偏差
baseline_bpm  = calculate_nrem_baseline(all_bpm);
bpm_deviation = abs(all_90th_percentile - baseline_bpm);

% 強制所有 feature 都變成 column vector
bpm_deviation       = bpm_deviation(:);
var_history         = var_history(:);
all_90th_percentile = all_90th_percentile(:);
varience            = varience(:);

% 封裝《WiFi-Sleep》論文完整延伸特徵
extra_features = struct();
extra_features.Resp_Rate_Mean       = rr_features.mean;
extra_features.Resp_Rate_Smoothed   = rr_features.smoothed;
extra_features.Resp_Rate_Derivative = rr_features.derivative;
extra_features.Resp_Rate_IQR        = rr_features.iqr;
extra_features.Resp_Depth_Variance  = resp_depth_var_arr;
extra_features.FIT_Mean             = fit_mean_arr;
extra_features.FIT_Variance         = fit_var_arr;
extra_features.IE_Ratio_Mean        = ie_ratio_mean_arr;
extra_features.IE_Ratio_Variance    = ie_ratio_var_arr;
extra_features.MBM_Frequency        = mbm_freq_arr;
extra_features.PLM_Frequency        = plm_freq_arr;

% 3.3 匯出包含既有 9 欄 + 論文新增 11 欄 (共 20 欄) 之完整特徵 CSV
featureTable = export_sleep_features(bpm_deviation, ...
                                     var_history, ...
                                     rollover_events, ...
                                     all_90th_percentile, ...
                                     varience, ...
                                     my_filename, ...
                                     seg_start_times, ...
                                     seg_end_times, ...
                                     all_state, ...
                                     extra_features);
