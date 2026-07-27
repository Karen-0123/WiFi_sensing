clear; clc; close all;

% 設定基本參數
filename = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\MATLAB\linux-80211n-csitool-supplementary-master\test_video003.dat';
Fs_orig = 200;
Fs_target = 40;                     % 目標均勻採樣率 (Hz)，若未指定則預設為 20Hz

fprintf('====== 系統啟動：開始生理訊號萃取量測 ======\n');
fprintf('====== 系統初始化：開始生理訊號萃取與量測 ======\n');

% 步驟 1: 讀取原始底層資料 (自動換算為秒數)
[csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename);

% 步驟 2: 抗混疊低通濾波與均勻重採樣 (獨立模組)
[csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);

% 步驟 3: 訊號特徵提取 (共軛相乘 + PCA 降維 + 自適應 SG 濾波)
[amp_pcs_norm, phase_pcs_norm] = process_csi_signal(csi_resampled, Fs_target);

% =========================================================================
% 步驟 4: 翻身動作偵測
% =========================================================================
[events, var_feat] = detect_rollover(amp_pcs_norm, Fs_target, 'WinSec', 3, 'ThreshStd', 0.6);
% =========================================================================

% 步驟 5: 6路 PCA 串流最佳呼吸特徵自動選擇
[best_name, best_sig, best_fpsd] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm, Fs_target);

% 步驟 6: 精細峰值偵測 (結合自適應標準差閾值與丟包遮罩聯防)
best_sig_column = best_sig(:);
if any(isnan(best_sig_column)), best_sig_column(isnan(best_sig_column)) = 0; end

[true_peak_idx, true_peak_vals] = detect_respiration_peaks(best_sig_column, gap_mask, Fs_target);

% --- 異常處理：有效頂點不足判定 ---
if length(true_peak_idx) < 2
    warning('【系統告警】未偵測到穩定呼吸訊號！(原因：有效呼吸頂點過少，無法計算呼吸率)');
    return;
end

% 步驟 7: 動態呼吸率計算 (20秒滑動窗口 + 丟包佔比熔斷)
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, length(best_sig_column), gap_mask, Fs_target);

% =========================================================================
% 步驟 8: 30秒自適應 FFT 滑動視窗頻域估計 (15% 熔斷機制)
% =========================================================================
fprintf('\n====== 執行 30 秒自適應 FFT 滑動視窗頻域估算 ======\n');

window_sec = 30;                     
step_sec   = 30;                       
window_len = window_sec * Fs_target; 
step_len   = step_sec * Fs_target;

total_points  = length(best_sig_column);
epoch_summary = []; 
epoch_idx     = 1;

for i = 1:step_len:(total_points - window_len + 1)
    
    current_sig  = best_sig_column(i : i + window_len - 1);
    current_gap  = gap_mask(i : i + window_len - 1);
    current_time = t_uniform(i) + window_sec/2; 
    
    epoch_drop_rate = (sum(current_gap) / window_len) * 100;
    
    % 15% 熔斷機制
    if epoch_drop_rate > 15.0
        fprintf('[區段 %2d] 時間: %5.1f 秒 | 丟包率: %5.2f%% -> 觸發熔斷！數據受損嚴重，跳過本區段。\n', ...
            epoch_idx, current_time, epoch_drop_rate);
        epoch_idx = epoch_idx + 1;
        continue; 
    end
    
    Nfft = 2^nextpow2(window_len * 4);   
    Y = fft(detrend(current_sig), Nfft); 
    P2 = abs(Y / window_len); P1 = P2(1 : Nfft/2 + 1); P1(2:end-1) = 2 * P1(2:end-1);
    f = Fs_target * (0 : (Nfft/2)) / Nfft; 
    
    breath_idx = find(f >= 0.15 & f <= 0.5);
    [~, max_idx] = max(P1(breath_idx));
    
    best_breath_freq = f(breath_idx(max_idx));
    epoch_fft_bpm = best_breath_freq * 60;
    
    epoch_summary = [epoch_summary; epoch_idx, current_time, epoch_fft_bpm, epoch_drop_rate];
    
    fprintf('[區段 %2d] 時間: %5.1f 秒 | 丟包率: %5.2f%% (安全) -> FFT 估算呼吸率: %.2f BPM\n', ...
        epoch_idx, current_time, epoch_drop_rate, epoch_fft_bpm);
    
    epoch_idx = epoch_idx + 1;
end

if ~isempty(epoch_summary)
    fft_mean_bpm = mean(epoch_summary(:, 3));
else
    fft_mean_bpm = NaN;
end

% =========================================================================
% 步驟 9: 最終生理量測綜合視覺化
% =========================================================================
figure('Name', 'CSI 呼吸生理監測最終成果報告', 'Position', [100, 100, 1000, 600]);

% 計算總丟包率
total_gap_ratio = (sum(gap_mask) / length(gap_mask)) * 100;

% -------------------------------------------------------------------------
% 子圖 1: 呼吸時域訊號波形與精準峰值標記
% -------------------------------------------------------------------------
subplot(2,1,1);

yl_sig = [-3, 3]; 
if ~isempty(best_sig)
    yl_sig = [min(best_sig) - 0.5, max(best_sig) + 0.5]; 
end

gap_diff = diff([0; gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends = find(gap_diff == -1) - 1;

for g = 1:length(gap_starts)
    patch([t_uniform(gap_starts(g)) t_uniform(gap_ends(g)) t_uniform(gap_ends(g)) t_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
end

if ~isempty(timestamp_sec)
    plot(timestamp_sec, zeros(size(timestamp_sec)) + yl_sig(1) + 0.1, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 4);
    hold on;
end

plot(t_uniform, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(true_peak_idx/Fs_target, true_peak_vals, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

title(['時域波形: ', best_name, ' (紅點: 波峰 | 綠虛線框: 翻身 | 粉紅陰影: 丟包區間)'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('標準化幅值');
grid on; axis tight;
ylim(yl_sig);

% -------------------------------------------------------------------------
% 子圖 2: 動態呼吸率變動趨勢圖 (BPM Timeline)
% -------------------------------------------------------------------------
subplot(2,1,2);
plot(time_axis_bpm, bpm_timeline, 'm-s', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'm'); hold on;

mean_bpm = mean(bpm_timeline, 'omitnan');
ax = gca; 
x_lim = get(ax, 'XLim');
line(x_lim, [mean_bpm mean_bpm], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

text(x_lim(2), mean_bpm, [' 觀測期平均呼吸率: ', num2str(mean_bpm, '%.1f'), ' bpm'], ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

ylim([5 45]);

title(['動態生理監測: 實時呼吸率走勢 | 數據總體丟包率: ', num2str(total_gap_ratio, '%.1f'), '% | 當前頻域估計: ', num2str(best_fpsd*60, '%.1f'), ' bpm'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('呼吸率 (BPM)');
grid on;

fprintf('====== 系統分析完成。選定通路: %s | 總體丟包率: %.2f%% | 平均呼吸率: %.2f BPM ======\n', best_name, total_gap_ratio, mean_bpm);

% =========================================================================
% 步驟 10: 呼吸變異性指標報告輸出
% =========================================================================
rrv_data = calculate_rrv_metrics(true_peak_idx, Fs_target);

% 計算總翻身次數
total_rollovers = length(events);

num_epochs = size(epoch_summary, 1);
rollover_counts = zeros(num_epochs, 1);

for e = 1:num_epochs
    ep_start = epoch_summary(e, 2) - 15; 
    ep_end   = epoch_summary(e, 2) + 15; 
    
    count = 0;
    for ev = 1:length(events)
        if events(ev).start_time >= ep_start && events(ev).start_time < ep_end
            count = count + 1;
        end
    end
    rollover_counts(e) = count;
end

colab_table = table(...
    epoch_summary(:, 1), ... 
    epoch_summary(:, 2), ... 
    epoch_summary(:, 3), ... 
    epoch_summary(:, 4), ... 
    rollover_counts,     ... 
    'VariableNames', {'Epoch_ID', 'Time_Sec', 'BPM', 'Drop_Rate', 'Rollover_Count'}...
);

if ~isnan(rrv_data.SDNN)
    fprintf('================== 呼吸變異性 (RRV) 分析報告 ==================\n');
    fprintf(' 時域波峰平均呼吸率   : %.2f BPM\n', mean_bpm);
    fprintf(' 30秒 FFT 滑動平均呼吸率: %.2f BPM\n', fft_mean_bpm);
    fprintf(' ---------------------------------------------------\n');
    fprintf(' 偵測到的總呼吸次數   : %d 次\n', rrv_data.Total_Breaths);
    fprintf(' 偵測到的總翻身次數   : %d 次\n', total_rollovers);
    fprintf(' 平均單次呼吸時間     : %.2f 秒 (Mean BB)\n', rrv_data.Mean_BB);
    fprintf(' 總呼吸變異性指標     : %.4f 秒 (SDNN)\n', rrv_data.SDNN);
    fprintf(' 短期快速呼吸變異性   : %.4f 秒 (RMSSD)\n', rrv_data.RMSSD);
    fprintf('===================================================\n');
end

% =========================================================================
% 自動打包特徵導出 CSV (專供 Google Colab / Python SVM 使用)
% =========================================================================
fprintf('\n====== 開始打包特徵資料集至 CSV ======\n');

csv_filename = 'sleep_features_for_colab.csv';
writetable(colab_table, csv_filename);
fprintf('成功產出 CSV 檔案：%s！請將此檔案上傳至 Google Colab。\n', csv_filename);