clear; clc; close all;

% 設定基本參數 (已修正檔案路徑開頭引號)
filename = 'C:\Users\fupei\Desktop\csi\data\0515_test_video\test_video002.dat';
Fs_orig = 200;
Fs_target = 40;                    % 目標均勻採樣率 (Hz)，不寫預設20Hz

fprintf('====== 系統啟動：開始生理訊號萃取量測 ======\n');

% 步驟 1: 讀取原始底層資料 (自動換算為秒數)
[csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename);

% 步驟 2: 抗混疊低通濾波與均勻重採樣 (獨立模組)
% 此處調用上方建立的獨立重採樣模組，生成均勻矩陣與地毯式丟包遮罩
[csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig);

% 步驟 3: 訊號特徵提取 (共軛相乘 + PCA 降維 + 自適應 SG 濾波)
[amp_pcs_norm, phase_pcs_norm] = process_csi_signal(csi_resampled, Fs_target);

% 步驟 4: 6路 PCA 串流最佳呼吸特徵自動選擇
[best_name, best_sig, best_fpsd] = select_respiration_stream(amp_pcs_norm, phase_pcs_norm, Fs_target);

% 步驟 5: 精細峰值偵測 (結合自適應標準差閾值與丟包遮罩聯防)
[true_peak_idx, true_peak_vals] = detect_respiration_peaks(best_sig, gap_mask, Fs_target);

% --- 異常處理：有效頂點不足判定 ---
if length(true_peak_idx) < 2
    warning('【系統告警】未偵測到穩定呼吸訊號！(原因：有效呼吸頂點過少，無法計算呼吸率)');
    return;
end

% 步驟 6: 動態呼吸率計算 (20秒滑動窗口 + 丟包佔比熔斷)
[bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, length(best_sig), gap_mask);


% =========================================================================
% ? 步驟 7: 最終生理量測綜合視覺化 (完整修復修補版，全面顯示所有掉包)
% =========================================================================
figure('Name', 'CSI 呼吸生理監測最終成果報告', 'Position', [100, 100, 1000, 600]);

% 計算總丟包率
total_gap_ratio = (sum(gap_mask) / length(gap_mask)) * 100;

% -------------------------------------------------------------------------
% 子圖 1: 呼吸時域訊號波形與精準峰值標記 (全面強化掉包顯示)
% -------------------------------------------------------------------------
subplot(2,1,1);

% 【核心修復】：先動態算出訊號的真實範圍，用來當作粉紅背景陰影的高度與 Y 軸極限
yl_sig = [-3, 3]; 
if ~isempty(best_sig)
    yl_sig = [min(best_sig) - 0.5, max(best_sig) + 0.5]; 
end

% 尋找 gap_mask 的連續區間（從 false 變 true 代表掉包開始，true 變 false 代表結束）
gap_diff = diff([0; gap_mask; 0]);
gap_starts = find(gap_diff == 1);
gap_ends = find(gap_diff == -1) - 1;

% ? 【標示優化 1】：將所有大小掉包區間塗上淺粉紅色陰影
for g = 1:length(gap_starts)
    patch([t_uniform(gap_starts(g)) t_uniform(gap_ends(g)) t_uniform(gap_ends(g)) t_uniform(gap_starts(g))], ...
          [yl_sig(1) yl_sig(1) yl_sig(2) yl_sig(2)], [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
end

% ? 【標示優化 2】：在時域圖最底部，點出「實際收到的網卡封包時間點」（灰色點）
% 點點密的地方代表沒掉包；點點出現空白斷層的地方，上方就會剛好對準粉紅陰影
if ~isempty(timestamp_sec)
    plot(timestamp_sec, zeros(size(timestamp_sec)) + yl_sig(1) + 0.1, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 4);
    hold on;
end

% 繪製主呼吸波形與紅點波峰
plot(t_uniform, best_sig, 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(true_peak_idx/Fs_target, true_peak_vals, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

title(['【時域波形】通道: ', best_name, ' (紅點:呼吸頂點 | 粉紅陰影:所有大小掉包區間 | 底部灰點:實際收包點)'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('標準化幅值');
grid on; axis tight;
ylim(yl_sig);

% -------------------------------------------------------------------------
% 子圖 2: 動態呼吸率變動趨勢圖 (BPM Timeline)
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

% ? 【標示優化 3】：把量化的「總數據掉包率」直接大字秀在 Title 上
title(['【動態生理監測】實時呼吸率走勢 | 數據總體掉包率: ', num2str(total_gap_ratio, '%.1f'), '% | 當前頻域估計: ', num2str(best_fpsd*60, '%.1f'), ' bpm'], 'FontSize', 12);
xlabel('時間 (秒)'); ylabel('呼吸率 (BPM)');
grid on;

% 終端機摘要輸出
fprintf('====== 系統分析完成。選定通路: %s | 總體掉包率: %.2f%% | 平均呼吸率: %.2f BPM ======\n', best_name, total_gap_ratio, mean_bpm);