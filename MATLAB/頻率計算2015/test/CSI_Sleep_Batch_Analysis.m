% ===== 針對所有 seg 區段檔案的連續睡眠監測 =====
clear; clc; close all;

% 1. 環境設定與參數初始化
% 設定資料路徑
data_folder = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\Frequency_Calculation_2015\sleep002_200hz_120min_0425\';

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
current_offset = 0; % 時間偏移量（秒）

set(0, 'DefaultFigureVisible', 'off'); % 迴圈中不顯示圖像以加速處理
fprintf('開始處理 %d 個檔案區段 (加入體動偵測與訊號處理)...\n', num_files);

%% 2. 核心訊號處理迴圈
for i = 1:num_files
    filename = fullfile(data_folder, file_list(i).name);
    try
        % 讀取 Intel 5300 CSI 原始數據
        [csi_matrix, ~, ~] = read_intel5300_dat(filename);
        
        % 訊號預處理：計算幅度 (Amplitude) 與相位 (Phase)
        [amp_f, phase_f] = process_csi_signal(csi_matrix);
        
        % 串流選擇：挑選呼吸特徵最明顯的子載波 (Subcarrier)
        [~, best_sig, ~] = select_respiration_stream(amp_f, phase_f);
        
        % 呼吸峰值檢測
        [peak_idx, ~] = detect_respiration_peaks(best_sig);
        
        % 體動偵測：識別受試者是否有大幅度翻身或動作
        [~, m_flags, m_time] = detect_body_motion(amp_f, 200);
        
        % 計算動態呼吸率 (BPM)
        total_samples = length(best_sig);
        [bpm_seg, time_seg] = calculate_dynamic_bpm(peak_idx, total_samples);
        
        % 合併數據：將當前區段結果加入全局陣列
        all_bpm = [all_bpm, bpm_seg];
        all_time = [all_time, time_seg + current_offset];
        
        % 合併體動偵測結果
        all_motion_flags = [all_motion_flags, m_flags];
        all_motion_time = [all_motion_time, m_time + current_offset];
        
        % 更新下一區段的起始時間偏移 (假設採樣率 200Hz)
        current_offset = current_offset + (total_samples / 200);
        clear csi_matrix amp_f phase_f best_sig;
    catch
        fprintf('警告：處理檔案 %s 時發生錯誤，跳過該區段。\n', file_list(i).name);
    end
end
set(0, 'DefaultFigureVisible', 'on');

%% 3. 特徵提取與統計分析
% 計算呼吸變異度 (Breathing Variability)
[var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, 300, 30);

% 計算呼吸偏離度與基準線 (BPM Deviation & Baseline)
[dev_history, baseline_bpm] = calculate_bpm_deviation(all_bpm);

%% 4. 睡眠階段預測 (SMARS 演算法模型)
% 綜合 BPM、變異度、偏離度與體動資訊，預測四個階段 (Deep, Core, REM, Awake)
[sleep_stages, stage_time] = predict_sleep_stages(all_bpm, all_time, var_history, var_time, baseline_bpm, all_motion_flags, all_motion_time);

%% 5. 結果視覺化
% 繪製圖表 1：SMARS 特徵分析
figure('Name', 'SMARS Features Analysis', 'Position', [50, 50, 1000, 850]);

subplot(3, 1, 1);
plot(all_time/60, all_bpm, 'b'); hold on;
plot(xlim, [baseline_bpm baseline_bpm], 'g-', 'LineWidth', 2);
title(sprintf('Feature 0: Respiration Rate (Baseline = %.1f BPM)', baseline_bpm));
ylabel('BPM'); grid on; axis tight;

subplot(3, 1, 2);
plot(var_time, var_history, 'r', 'LineWidth', 1.5); hold on;
plot(xlim, [0.8 0.8], 'k--'); % REM 門檻線
plot(xlim, [0.45 0.45], 'b--'); % Deep 門檻線
title('Feature 1: Breathing Variability (Variance)');
ylabel('Variance'); grid on; axis tight;

subplot(3, 1, 3);
plot(all_time/60, dev_history, 'm'); hold on;
plot(xlim, [2.0 2.0], 'k--');
title('Feature 2: Breathing Deviation');
ylabel('Deviation'); xlabel('Time (min)'); grid on; axis tight;

% 繪製圖表 2：睡眠圖 (Hypnogram)
figure('Name', 'Hypnogram', 'Position', [100, 100, 1000, 350]);
stairs(stage_time, sleep_stages, 'LineWidth', 2.5, 'Color', [0.1 0.4 0.8]);

set(gca, 'YTick', [0 1 2 3]);
set(gca, 'YTickLabel', {'Deep', 'Core', 'REM', 'Awake'});
ylim([-0.5 3.5]);
title('Sleep Prediction: Hypnogram (Integrated with Motion Detection)');
xlabel('Time (min)'); grid on;

%% 6. 統計報告輸出
total_ep = length(sleep_stages);
fprintf('\n===== 睡眠階段分析報告 =====\n');
fprintf('清醒 (Awake): %.1f%%\n', sum(sleep_stages==3)/total_ep*100);
fprintf('快速動眼期 (REM): %.1f%%\n', sum(sleep_stages==2)/total_ep*100);
fprintf('淺眠 (Core): %.1f%%\n', sum(sleep_stages==1)/total_ep*100);
fprintf('深眠 (Deep): %.1f%%\n', sum(sleep_stages==0)/total_ep*100);