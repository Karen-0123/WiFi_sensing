clear; clc; close all;


% 1. 環境設定與參數初始化
data_folder = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\linux-80211n-csitool-supplementary-master\sleep011_200hz_306min_0705';
Fs_orig   = 200;
Fs_target = 40; 

% 檢索資料夾內所有 .dat 檔案 (支援 seg1.dat 或 001.dat 等各種命名格式)
file_pattern = fullfile(data_folder, '*.dat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0, error('Can not find target .dat files in directory!'); end

% 檔案排序：依據檔名中的數字進行精確排序 (例如 seg1, seg2 ... seg100)
seg_numbers = zeros(1, num_files);
for k = 1:num_files
    tokens = regexp(file_list(k).name, '(\d+)', 'tokens');
    if ~isempty(tokens)
        seg_numbers(k) = str2double(tokens{end}{1}); 
    else
        seg_numbers(k) = k;
    end
end
[~, sort_idx] = sort(seg_numbers);
file_list = file_list(sort_idx);

% 初始化全局變數
all_bpm = []; all_time = []; 
all_motion_flags = []; all_motion_time = [];
all_drop_rates = []; 
all_best_sig = []; all_gap_mask = []; all_true_peaks = [];
current_offset = 0; 
all_90th_percentile = []; 
total_rollovers = 0;  % 紀錄總翻身次數

set(0, 'DefaultFigureVisible', 'off'); % 批次處理時隱藏中間圖形以提升運算速度
fprintf('====== System Start: Batch Processing %d File Segments ======\n', num_files);


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
        
        % 計算當前 Segment 的 Drop Rate (丟包率)
        seg_drop_rate = (sum(gap_mask) / total_samples) * 100;
        drop_rate_seg = ones(size(bpm_seg)) * seg_drop_rate;
        
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
        all_drop_rates   = [all_drop_rates, drop_rate_seg];
        all_90th_percentile = [all_90th_percentile, seg_90th];
        
        % 拼接全場景波形與頂點
        all_best_sig = [all_best_sig; best_sig_col];
        all_gap_mask = [all_gap_mask; gap_mask(:)];
        if ~isempty(peak_idx)
            global_peaks = peak_idx + round(current_offset * Fs_target);
            all_true_peaks = [all_true_peaks; global_peaks(:)];
        end
        
        % 更新下一區段的時間偏移
        current_offset = current_offset + (total_samples / Fs_target);
        clear csi_matrix csi_resampled amp_f phase_f best_sig;
        
        % 進度提示 (每 10 個檔案印出一次)
        if mod(i, 10) == 0 || i == num_files
            fprintf(' Processing Progress: %d / %d files completed.\n', i, num_files);
        end
    catch ME
        fprintf('Warning: Error processing %s, skip segment.\n', file_list(i).name);
    end
end
set(0, 'DefaultFigureVisible', 'on');


%% 3. 核心特徵計算：BRV (變異數) 與 BR_Deviation (偏差)
% =========================================================================

% 呼叫下方 10 階 Butter 濾波去趨勢 + 300s 歸一化 BRV 函數
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 300, 30);

% 呼叫 NREM 基線函數計算呼吸偏差
baseline_bpm = calculate_nrem_baseline(all_bpm); 
bpm_deviation = abs(all_bpm - baseline_bpm);

%% =========================================================================
% 4. 結果視覺化 (長時段監測報告圖表)
% =========================================================================

figure('Name', 'CSI Long-term Sleep Respiration Monitoring Report', 'Position', [100, 100, 1050, 650]);

t_global_uniform = (0:length(all_best_sig)-1) / Fs_target;
total_gap_ratio = (sum(all_gap_mask) / length(all_gap_mask)) * 100;

% 子圖 1: 全局時域波形與丟包區間
subplot(2,1,1);
yl_sig = [-3, 3]; 
if ~isempty(all_best_sig)
    yl_sig = [min(all_best_sig) - 0.5, max(all_best_sig) + 0.5]; 
end

gap_diff = diff([0; all_gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends   = find(gap_diff == -1) - 1;
for g = 1:length(gap_starts)
    patch([t_global_uniform(gap_starts(g)) t_global_uniform(gap_ends(g)) t_global_uniform(gap_ends(g)) t_global_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
end

plot(t_global_uniform, all_best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.0); hold on;
if ~isempty(all_true_peaks)
    valid_peaks = all_true_peaks(all_true_peaks <= length(all_best_sig));
    plot(valid_peaks/Fs_target, all_best_sig(valid_peaks), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
end

title('Time Domain Waveform (Red: Peaks | Pink Patch: Packet Drop)', 'FontSize', 12);
xlabel('Time (s)'); ylabel('Normalized Amp');
grid on; axis tight; ylim(yl_sig);

% 子圖 2: Dynamic BPM Timeline
subplot(2,1,2);
plot(all_time, all_bpm, 'm-s', 'LineWidth', 1.5, 'MarkerSize', 3, 'MarkerFaceColor', 'm'); hold on;

mean_bpm = mean(all_bpm, 'omitnan');
ax = gca; x_lim = get(ax, 'XLim');
line(x_lim, [mean_bpm mean_bpm], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

text(x_lim(2), mean_bpm, [' Mean BPM: ', num2str(mean_bpm, '%.1f'), ' bpm'], ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

ylim([5 45]);
title(['Real-time Respiration Rate Timeline | Total Drop Rate: ', num2str(total_gap_ratio, '%.1f'), '%'], 'FontSize', 12);
xlabel('Time (s)'); ylabel('Breathing Rate (BPM)');
grid on;

%% =========================================================================
% 5. 打包 7 大特徵欄位並匯出 CSV (專供 Google Colab SVM 使用)
% =========================================================================
fprintf('\n[Export] Generating Full Dataset CSV for Colab SVM Model...\n');

N = min([length(all_bpm), length(var_history), length(bpm_deviation)]);

Epoch_ID        = (1:N)';
Time_Sec        = all_time(1:N)';
BPM             = all_bpm(1:N)';
Drop_Rate       = all_drop_rates(1:N)';
Rollover_Count  = all_motion_flags(1:N)';
BR_Variability  = var_history(1:N)'; 
BR_Deviation    = bpm_deviation(1:N)';

export_table = table(...
    Epoch_ID, Time_Sec, BPM, Drop_Rate, Rollover_Count, BR_Variability, BR_Deviation, ...
    'VariableNames', {'Epoch_ID', 'Time_Sec', 'BPM', 'Drop_Rate', 'Rollover_Count', 'BR_Variability', 'BR_Deviation'}...
);

output_csv_path = fullfile(data_folder, 'sleep_features_for_colab3.csv');
writetable(export_table, output_csv_path);

fprintf(' Success! Feature File Exported To:\n %s\n\n', output_csv_path);

