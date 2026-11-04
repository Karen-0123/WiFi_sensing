clear; clc; close all;

% 設定基本參數
filename = "C:\Users\fupei\Desktop\csi\data\0629_test_video\test_video008.dat"%"C:\Users\fupei\Desktop\csi\data\sleep\sleep002_200hz_120min_0425\real_breathe_20260425_024249_seg16.dat";
Fs_orig = 200;
Fs_target = 40;                     % 目標均勻採樣率 (Hz)

fprintf('====== 系統啟動：開始 WiFi-Sleep 生理訊號與體動萃取量測 ======\n');

% 步驟 1: 讀取原始底層資料 (自動換算為秒數)
[csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename);

% 步驟 2: 抗混疊低通濾波與均勻重採樣 (獨立模組)
[csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);

% 步驟 3: 論文 CSI Ratio + MRC-PCA 訊號特徵提取
[amp_pcs_norm, phase_pcs_norm, csi_ratio_complex, mrc_pca_details] = process_csi_signal(csi_resampled, Fs_target);

% =========================================================================
% 步驟 4: 基於 Doppler-MUSIC 速度譜能量變異數之體動偵測 (區分 MBM 翻身與 PLM 微動)
% =========================================================================
[num_events, events, var_feat, motion_info] = detect_rollover(csi_ratio_complex, Fs_target, 'GapMask', gap_mask);

% =========================================================================
% 步驟 5: 清醒 / 睡眠階段區分
% =========================================================================
actual_duration_sec = t_uniform(end) - t_uniform(1);
if actual_duration_sec <= 0, actual_duration_sec = 180; end
[state, sleep_features] = detect_sleep_state(events, actual_duration_sec);
disp(state);

% 步驟 6: 最佳 MRC-PCA 呼吸特徵串流選擇 (排除體動干擾區段評估 PSD)
[best_name, best_sig, best_fpsd, best_sg_sig] = select_respiration_stream( ...
    amp_pcs_norm, phase_pcs_norm, Fs_target, motion_info.motion_mask, mrc_pca_details);

% 步驟 7: 精細呼吸波峰與波谷偵測 + 論文強化呼吸波形特徵提取 (深度變異、FIT、I/E Ratio)
[true_peak_idx, true_peak_vals, valley_idx, valley_vals, waveform_features] = detect_respiration_peaks( ...
    best_sig, gap_mask, Fs_target, motion_info.motion_mask, best_sg_sig);

% --- 異常處理：有效頂點不足判定 ---
if length(true_peak_idx) < 2
    warning('【系統告警】未偵測到穩定呼吸訊號！(原因：有效呼吸頂點過少，無法計算呼吸率)');
    return;
end

% 步驟 8: 動態呼吸率計算 (論文 ACF 自相關函數 + 體動區間切除與線性插值修補)
[seg_90th, bpm_timeline, time_axis_bpm, bpm_stats] = calculate_dynamic_bpm( ...
    true_peak_idx, length(best_sig), gap_mask, Fs_target, best_sig, motion_info.motion_mask);


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

% 尋找 gap_mask 的連續區間
gap_diff = diff([0; gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends = find(gap_diff == -1) - 1;

% 將所有大小掉包區間塗上淺粉紅色陰影
for g = 1:length(gap_starts)
    patch([t_uniform(gap_starts(g)) t_uniform(gap_ends(g)) t_uniform(gap_ends(g)) t_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
end

% 在時域圖最底部，點出「實際收到的網卡封包時間點」（灰色點）
if ~isempty(timestamp_sec)
    plot(timestamp_sec, zeros(size(timestamp_sec)) + yl_sig(1) + 0.1, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 4);
    hold on;
end

% 繪製主呼吸波形、紅點吸氣波峰與綠點吐氣波谷
plot(t_uniform, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(t_uniform(true_peak_idx), true_peak_vals, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
if ~isempty(valley_idx)
    plot(t_uniform(valley_idx), valley_vals, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 5);
end

% 更新標題
title(['【時域波形】通道: ', best_name, ' (紅點:呼吸頂點 | 粉紅陰影:掉包區間)'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('標準化幅值');
grid on; axis tight;
ylim(yl_sig);

% -------------------------------------------------------------------------
% 子圖 2: 動態呼吸率變動趨勢圖 (ACF + 體動線性插值修補後之 BPM Timeline)
% -------------------------------------------------------------------------
subplot(2,1,2);
plot(time_axis_bpm, bpm_timeline, 'm-s', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'm'); hold on;

% 繪製平均呼吸率虛線
mean_bpm = mean(bpm_timeline, 'omitnan');
ax = gca;
x_lim = get(ax, 'XLim');
line(x_lim, [mean_bpm mean_bpm], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

% 添加右側標籤文字註解
text(x_lim(2), mean_bpm, [' 觀測期平均呼吸率: ', num2str(mean_bpm, '%.1f'), ' bpm'], ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

ylim([5 45]);

title(['【動態生理監測】實時呼吸率走勢 | 數據總體掉包率: ', num2str(total_gap_ratio, '%.1f'), '% | 當前頻域估計: ', num2str(best_fpsd*60, '%.1f'), ' bpm'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('呼吸率 (BPM)');
grid on;

% 終端機摘要輸出
fprintf('====== 系統分析完成。選定通路: %s | 總體掉包率: %.2f%% | 平均呼吸率: %.2f BPM (90th: %.2f, IQR: %.2f) ======\n', ...
        best_name, total_gap_ratio, mean_bpm, seg_90th, bpm_stats.iqr);
fprintf('====== 強化波形特徵：深度變異=%.4f | FIT=%.3f (var=%.4f) | I/E比=%.3f (var=%.4f) ======\n', ...
        waveform_features.resp_depth_var, waveform_features.fit_mean, waveform_features.fit_var, ...
        waveform_features.ie_ratio_mean, waveform_features.ie_ratio_var);
