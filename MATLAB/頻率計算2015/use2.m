clear; clc; close all;

% 1. 路徑設定
data_folder = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\Frequency_Calculation_2015\sleep002_200hz_120min_0425\';

file_pattern = fullfile(data_folder, '*seg*.dat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0, error('找不到檔案！'); end

% 自然排序
seg_numbers = zeros(1, num_files);
for k = 1:num_files
    tokens = regexp(file_list(k).name, 'seg(\d+)', 'tokens');
    if ~isempty(tokens), seg_numbers(k) = str2double(tokens{1}{1}); end
end
[~, sort_idx] = sort(seg_numbers);
file_list = file_list(sort_idx);

all_bpm = []; all_time = []; 
all_motion_flags = []; all_motion_time = [];
current_offset = 0;

set(0, 'DefaultFigureVisible', 'off'); 
fprintf('開始處理 %d 個檔案 (加入體動偵測)...\n', num_files);

for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);
    try
        [csi_matrix, ~, ~] = read_intel5300_dat(filename);
        [amp_f, phase_f] = process_csi_signal(csi_matrix);
        [~, best_sig, ~] = select_respiration_stream(amp_f, phase_f);
        [peak_idx, ~] = detect_respiration_peaks(best_sig);
        
        % --- 【關鍵新增】呼叫體動偵測 ---
        [~, m_flags, m_time] = detect_body_motion(amp_f, 200);
        
        total_samples = length(best_sig);
        [bpm_seg, time_seg] = calculate_dynamic_bpm(peak_idx, total_samples);
        
        all_bpm = [all_bpm, bpm_seg];
        all_time = [all_time, time_seg + current_offset];
        
        % 儲存體動數據
        all_motion_flags = [all_motion_flags, m_flags];
        all_motion_time = [all_motion_time, m_time + current_offset];
        
        current_offset = current_offset + (total_samples / 200);
        clear csi_matrix amp_f phase_f best_sig;
    catch
    end
end
set(0, 'DefaultFigureVisible', 'on');

% 3. 特徵提取
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 300, 30);
[dev_history, baseline_bpm] = calculate_bpm_deviation(all_bpm);

% 4. 睡眠分期預測 (傳入體動數據)
[sleep_stages, stage_time] = predict_sleep_stages(all_bpm, all_time, var_history, var_time, baseline_bpm, all_motion_flags, all_motion_time);

% --- 圖表 1：SMARS 核心特徵分析 ---
figure('Name', 'SMARS Features Analysis', 'Position', [50, 50, 1000, 850]);
subplot(3, 1, 1);
plot(all_time/60, all_bpm, 'b'); hold on;
plot(xlim, [baseline_bpm baseline_bpm], 'g-', 'LineWidth', 2);
title(sprintf('Feature 0: Respiration Rate (Baseline = %.1f BPM)', baseline_bpm));
ylabel('BPM'); grid on; axis tight;

subplot(3, 1, 2);
plot(var_time, var_history, 'r', 'LineWidth', 1.5); hold on;
plot(xlim, [0.8 0.8], 'k--'); % REM 門檻
plot(xlim, [0.45 0.45], 'b--'); % Deep 門檻
title('Feature 1: Breathing Variability (Variance)');
ylabel('Variance'); grid on; axis tight;

subplot(3, 1, 3);
plot(all_time/60, dev_history, 'm'); hold on;
plot(xlim, [2.0 2.0], 'k--');
title('Feature 2: Breathing Deviation');
ylabel('Deviation'); xlabel('Time (min)'); grid on; axis tight;

% --- 圖表 2：睡眠階梯圖 ---
figure('Name', 'Hypnogram ', 'Position', [100, 100, 1000, 350]);
stairs(stage_time, sleep_stages, 'LineWidth', 2.5, 'Color', [0.1 0.4 0.8]);

set(gca, 'YTick', [0 1 2 3]);
set(gca, 'YTickLabel', {'Deep', 'Core', 'REM', 'Awake'});
ylim([-0.5 3.5]);
title('sleep002: Predicted Hypnogram (With Motion Detection)');
xlabel('Time (min)'); grid on;

% --- 5. 輸出統計比例 ---
total_ep = length(sleep_stages);
fprintf('\n===== 睡眠階段比例統計 =====\n');
fprintf('Awake: %.1f%%\n', sum(sleep_stages==3)/total_ep*100);
fprintf('REM:   %.1f%%\n', sum(sleep_stages==2)/total_ep*100);
fprintf('Core:  %.1f%%\n', sum(sleep_stages==1)/total_ep*100);
fprintf('Deep:  %.1f%%\n', sum(sleep_stages==0)/total_ep*100);