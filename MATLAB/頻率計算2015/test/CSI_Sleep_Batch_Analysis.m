clear; clc; close all;

% 1. 環境設定與參數初始化
data_folder = 'D:\大學資料\sleep_dataset\sleep011_200hz_306min_0705';
Fs_orig   = 200;
Fs_target = 40; 

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

% 初始化全局變數
all_bpm = []; all_time = []; 
all_motion_flags = []; all_motion_time = [];
all_best_sig = []; all_gap_mask = []; all_true_peaks = [];
current_offset = 0; 
all_90th_percentile = []; 
total_rollovers = 0; % 紀錄總翻身次數

set(0, 'DefaultFigureVisible', 'off'); % 迴圈中不顯示圖像以加速處理
fprintf('====== 系統啟動：開始批次處理 %d 個檔案區段 ======\n', num_files);

%% 2. 核心訊號處理迴圈
for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);
    try
        % 讀取 Intel 5300 CSI 原始數據
        [csi_matrix, timestamp_sec, ~] = read_intel5300_dat(filename);
        
        % 抗混疊低通濾波與均勻重採樣
        [csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);
        
        % 訊號預處理：幅度與相位
        [amp_f, phase_f] = process_csi_signal(csi_resampled, Fs_target);
        
        % 串流選擇
        [best_name, best_sig, ~] = select_respiration_stream(amp_f, phase_f, Fs_target);
        best_sig_col = best_sig(:);
        if any(isnan(best_sig_col)), best_sig_col(isnan(best_sig_col)) = 0; end
        
        % 翻身與體動動作偵測
        [rollover_events, ~] = detect_rollover(amp_f, Fs_target, 'WinSec', 3, 'ThreshStd', 0.6);
        if ~isempty(rollover_events)
            total_rollovers = total_rollovers + length(rollover_events);
        end
        
        % 呼吸峰值檢測
        [peak_idx, ~] = detect_respiration_peaks(best_sig_col, gap_mask, Fs_target);
        
        % 計算動態呼吸率 (BPM)
        total_samples = length(best_sig_col);
        [seg_90th, bpm_seg, time_seg] = calculate_dynamic_bpm(peak_idx, total_samples, gap_mask, Fs_target);
        
        % 產生區段體動標籤
        motion_seg = zeros(size(bpm_seg));
        if ~isempty(rollover_events)
            for ev = 1:length(rollover_events)
                m_start = rollover_events(ev).start_time;
                m_end   = rollover_events(ev).end_time;
                motion_seg(time_seg >= m_start & time_seg <= m_end) = 1;
            end
        end
        
        % 合併數據至全局陣列
        all_bpm = [all_bpm, bpm_seg];
        all_time = [all_time, time_seg + current_offset];
        all_motion_flags = [all_motion_flags, motion_seg];
        all_motion_time  = [all_motion_time, time_seg + current_offset];
        all_90th_percentile = [all_90th_percentile, seg_90th];
        
        % 拼接全場景波形與頂點（專供對齊 singletest 繪圖）
        all_best_sig = [all_best_sig; best_sig_col];
        all_gap_mask = [all_gap_mask; gap_mask(:)];
        if ~isempty(peak_idx)
            global_peaks = peak_idx + round(current_offset * Fs_target);
            all_true_peaks = [all_true_peaks; global_peaks(:)];
        end
        
        % 更新下一區段的起始時間偏移
        current_offset = current_offset + (total_samples / Fs_target);
        clear csi_matrix csi_resampled amp_f phase_f best_sig;
    catch ME
        fprintf('警告：處理檔案 %s 時發生錯誤，跳過該區段。\n', file_list(i).name);
        fprintf('錯誤原因: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('錯誤發生在第 %d 行\n', ME.stack(1).line);
        end
    end
end
set(0, 'DefaultFigureVisible', 'on');

%% 3. 特徵提取與統計分析

% 計算呼吸變異度 (Breathing Variability) - 5分鐘滑動視窗
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 300, 30);

% 計算呼吸頻率偏差 (BPM Deviation)
baseline_bpm = calculate_nrem_baseline(all_bpm); 
bpm_deviation = abs(all_bpm - baseline_bpm);

%% 4. 結果視覺化 (對齊 singletest 圖表風格)

figure('Name', 'CSI 全局生理動態監測報告', 'Position', [100, 100, 1050, 650]);

t_global_uniform = (0:length(all_best_sig)-1) / Fs_target;
total_gap_ratio = (sum(all_gap_mask) / length(all_gap_mask)) * 100;

% 子圖 1: 全局時域訊號與丟包標示
subplot(2,1,1);
yl_sig = [-3, 3]; 
if ~isempty(all_best_sig)
    yl_sig = [min(all_best_sig) - 0.5, max(all_best_sig) + 0.5]; 
end

% 塗上丟包區間 (淺粉紅)
gap_diff = diff([0; all_gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends   = find(gap_diff == -1) - 1;
for g = 1:length(gap_starts)
    patch([t_global_uniform(gap_starts(g)) t_global_uniform(gap_ends(g)) t_global_uniform(gap_ends(g)) t_global_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
end

% 畫主波形與波峰紅點
plot(t_global_uniform, all_best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.2); hold on;
if ~isempty(all_true_peaks)
    valid_peaks = all_true_peaks(all_true_peaks <= length(all_best_sig));
    plot(valid_peaks/Fs_target, all_best_sig(valid_peaks), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
end

title(['時域波形: 全區段連續訊號 (紅點: 波峰 | 粉紅陰影: 丟包區間)'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('標準化幅值');
grid on; axis tight; ylim(yl_sig);

% 子圖 2: BPM Timeline 走勢圖
subplot(2,1,2);
plot(all_time, all_bpm, 'm-s', 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', 'm'); hold on;

mean_bpm = mean(all_bpm, 'omitnan');
ax = gca; x_lim = get(ax, 'XLim');
line(x_lim, [mean_bpm mean_bpm], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

mean_90th = mean(all_90th_percentile, 'omitnan');
text(x_lim(2), mean_bpm, [' 平均呼吸率: ', num2str(mean_bpm, '%.1f'), ' bpm'], ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

ylim([5 45]);
title(['動態生理監測: 實時呼吸率 | 全局丟包率: ', num2str(total_gap_ratio, '%.1f'), '% | 平均 90th 極限值: ', num2str(mean_90th, '%.1f'), ' bpm'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('呼吸率 (BPM)');
grid on;

%% 5. 呼吸變異性 (RRV) 生理指標報告
rrv_data = calculate_rrv_metrics(all_true_peaks, Fs_target);

if ~isnan(rrv_data.SDNN)
    fprintf('\n================== 呼吸變異性 (RRV) 生理指標 ==================\n');
    fprintf(' 藍色波形總平均 BPM   : %.2f BPM\n', mean_bpm);
    fprintf(' 偵測到的總呼吸次數   : %d 次\n', rrv_data.Total_Breaths);
    fprintf(' 偵測到的總翻身次數   : %d 次\n', total_rollovers);
    fprintf(' 平均單次呼吸時間     : %.2f 秒 (Mean BB)\n', rrv_data.Mean_BB);
    fprintf(' 總呼吸變異性指標     : %.4f 秒 (SDNN)\n', rrv_data.SDNN);
    fprintf(' 短期快速呼吸變異性   : %.4f 秒 (RMSSD)\n', rrv_data.RMSSD);
    fprintf('=============================================================\n');
end

%% 6. CSV 資料庫匯出
fprintf('\n【資料庫連接階段】正在產生資料庫匯入檔...\n');

N = min([length(all_bpm), length(var_history), length(bpm_deviation)]);

col_time        = all_time(1:N)';
col_bpm         = all_bpm(1:N)';
col_variability = var_history(1:N)'; 
col_deviation   = bpm_deviation(1:N)';
col_motion      = all_motion_flags(1:N)';
col_quality     = ones(N, 1); 

export_table = table(...
    col_time, col_bpm, col_variability, col_deviation, col_motion, col_quality, ...
    'VariableNames', {'Time_Sec', 'BPM', 'Variability', 'Deviation', 'Motion_Flag', 'Signal_Quality'}...
);

output_csv_path = fullfile(data_folder, 'real_breathing_output.csv');
writetable(export_table, output_csv_path);

fprintf('監測數據與呼吸變異數已成功匯出！\n');
fprintf('檔案位置: %s\n', output_csv_path);

%% =========================================================================
% 輔助子函數：計算動態呼吸變異數 (Breathing Variability)
% =========================================================================
function [var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, window_sec, step_sec)
    if nargin < 3, window_sec = 300; end
    if nargin < 4, step_sec = 30; end
    
    num_pts = length(all_bpm);
    var_history = zeros(size(all_bpm));
    var_time = all_time;
    
    for k = 1:num_pts
        t_curr = all_time(k);
        in_win = (all_time >= (t_curr - window_sec/2)) & (all_time <= (t_curr + window_sec/2));
        valid_bpms = all_bpm(in_win & ~isnan(all_bpm) & all_bpm >= 5 & all_bpm <= 40);
        
        if length(valid_bpms) >= 3
            var_history(k) = var(valid_bpms); 
        else
            var_history(k) = 0; 
        end
    end
end