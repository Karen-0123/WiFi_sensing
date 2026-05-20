% ===== CSI 處理主程式 (main.m) =====
clear; clc; close all;

% 1. 設定你的 .dat 檔案路徑 (請替換為你實際的檔案名稱)
filename = 'C:\Users\fupei\Desktop\csi\data\sleep\sleep002_200hz_120min_0425\real_breathe_20260425_020647_seg4.dat'; 

% 2. 讀取與過濾原始資料
% 使用我們寫好的 get_clean_csi_5300 函式
fprintf('步驟 1: 讀取資料...\n');
[csi_matrix, timestamp, rssi] = read_intel5300_dat(filename);

% 3. 執行 PCA 與 基礎處理
fprintf('步驟 2: 執行 PCA 降維與預處理...\n');
[amp_pcs_norm, phase_pcs_norm] = process_csi_signal(csi_matrix);

% 4. 從 6 個 PCA 串流中自動選擇最佳呼吸波形
fprintf('步驟 3: 自動選擇最佳 PCA 串流...\n');
[best_name, best_sig, breathing_rate_hz] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm);

% 5. 執行精細峰值偵測 (使用選出的最尖銳波形)
fprintf('步驟 4: 執行精細峰值偵測...\n');
[peak_indices, peak_values] = detect_respiration_peaks(best_sig);

if length(peak_indices) < 2
    warning('未偵測到穩定呼吸訊號 (有效頂點不足)');
end

subplot(2,1,1);
fs = 200;
t = (0:length(best_sig)-1)/fs;
plot(t, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(peak_indices/fs, peak_values, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
title(['最佳呼吸串流: ', best_name, ' (紅點為校正後峰值)']);
xlabel('時間 (s)'); ylabel('相對強度');
grid on; axis tight;

% 6. 計算動態 BPM
fprintf('步驟 5: 計算動態呼吸率 (BPM)...\n');
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(peak_indices, length(best_sig));

% 最終視覺化輸出
figure('Name', 'CSI 呼吸監測最終結果', 'Position', [100, 100, 1000, 700]);

% 子圖 1: 呼吸波形與峰值
subplot(2,1,1);
fs = 200;
t = (0:length(best_sig)-1)/fs;
plot(t, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(peak_indices/fs, peak_values, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
title(['最佳呼吸串流: ', best_name, ' (紅點為校正後峰值)']);
xlabel('時間 (s)'); ylabel('相對強度');
grid on; axis tight;

% 子圖 2: 動態 BPM 曲線
subplot(2,1,2);
plot(time_axis_bpm, bpm_timeline, 'm-s', 'LineWidth', 2, 'MarkerSize', 4);
avg_bpm = nanmean(bpm_timeline);   % 忽略 NaN 計算平均
xl = xlim;   % 取得 x 軸範圍
line(xl, [avg_bpm avg_bpm], ...
    'Color', 'k', ...
    'LineStyle', '--');
text(xl(2), avg_bpm, '平均呼吸率', ...
    'HorizontalAlignment', 'right');
ylim([5 45]);
title(['動態呼吸率變化 (BPM) | 當前估計: ', ...
       sprintf('%.1f', breathing_rate_hz*60), ...
       ' bpm']);
xlabel('時間 (s)'); ylabel('BPM');
grid on;

fprintf('所有流程執行完畢！\n');