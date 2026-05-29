clear; clc; close all;

% 設定基本參數
filename = 'C:\Users\fupei\Desktop\csi\data\0515_test_video\test_video002.dat';
Fs_orig = 200;
Fs_target = 30;                   % 目標均勻採樣率 (Hz)，不寫預設20Hz

fprintf('====== 系統啟動：開始生理訊號萃取量測 ======\n');

% 步驟 1: 讀取原始底層資料 (自動換算為秒數)
[csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename);

% 步驟 2: 抗混疊低通濾波與 20Hz 均勻重採樣 (獨立模組)
% 此處調用上一步建立的獨立重採樣模組，生成均勻矩陣與丟包遮罩
[csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);

% 步驟 3: 訊號特徵提取 (共軛相乘 + PCA 降維 + 自適應 SG 濾波)
[amp_pcs_norm, phase_pcs_norm] = process_csi_signal(csi_resampled, Fs_target);

% 步驟 4: 6路 PCA 串流最佳呼吸特徵自動選擇
[best_name, best_sig, best_fpsd] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm, Fs_target);

% --- 異常處理：全成分熔斷機制 ---
% if contains(best_name, 'None') || best_fpsd == 0
%     warning('【系統告警】未偵測到穩定呼吸訊號！(原因：所有 PCA 成分皆未通過 10~37 bpm 頻率篩選)');
%     return; % 優雅終止程式，不進行後續錯誤計算
% end

% 步驟 5: 精細峰值偵測 (結合自適應標準差閾值與丟包遮罩聯防)
[true_peak_idx, true_peak_vals] = detect_respiration_peaks(best_sig, gap_mask);

% --- 異常處理：有效頂點不足判定 ---
if length(true_peak_idx) < 2
    warning('【系統告警】未偵測到穩定呼吸訊號！(原因：有效呼吸頂點過少，無法計算呼吸率)');
    return;
end

% 步驟 6: 動態呼吸率計算 (20秒滑動窗口 + 丟包佔比熔斷)
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, length(best_sig), gap_mask);

% ==================== 步驟 7: 最終生理量測綜合視覺化 ====================
figure('Name', 'CSI 呼吸生理監測最終成果報告', 'Position', [100, 100, 1000, 600]);

% 子圖 1: 呼吸時域訊號波形與精準峰值標記
subplot(2,1,1);
plot(t_uniform, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(true_peak_idx/Fs_target, true_peak_vals, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

% 將丟包區間在圖表上以淺紅色區塊標出，方便肉眼比對
yl_sig = ylim;
gap_diff = diff([0; gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends = find(gap_diff == -1) - 1;
for g = 1:length(gap_starts)
    patch([t_uniform(gap_starts(g)) t_uniform(gap_ends(g)) t_uniform(gap_ends(g)) t_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
end

title(['【時域波形】最終選定呼吸主成分: ', best_name, ' (紅點為驗證呼吸頂點，陰影為硬體丟包區)'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('標準化幅值');
grid on; axis tight;

% 子圖 2: 動態呼吸率變動趨勢圖 (BPM Timeline)
subplot(2,1,2);
plot(time_axis_bpm, bpm_timeline, 'm-s', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'm'); hold on;

% ==================== R2015b 相容改寫：使用 line() 替代 yline() ====================
% 繪製整段觀測的平均呼吸率虛線 (排除不可信的 NaN 點)
mean_bpm = mean(bpm_timeline, 'omitnan');

% 方法 1: 使用 line() 函數繪製水平虛線
ax = gca;
x_lim = get(ax, 'XLim');
line(x_lim, [mean_bpm mean_bpm], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

% 方法 2: 添加標籤文字註解 (位置在右側)
text(x_lim(2), mean_bpm, [' 觀測期平均呼吸率: ', num2str(mean_bpm, '%.1f'), ' bpm'], ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

% 設定軸線範圍
ylim([5 45]);

title(['【動態生理監測】實時呼吸率變化趨勢 (20秒滑動觀測) | 當前頻域估計: ', num2str(best_fpsd*60, '%.1f'), ' bpm'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('呼吸率 (BPM)');
grid on;

fprintf('====== 系統分析完成。選定通路: %s | 平均呼吸率: %.2f BPM ======\n', best_name, mean_bpm);