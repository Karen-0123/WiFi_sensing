function [num_events, events, var_signal] = detect_rollover(amp_signal, Fs, varargin)
% detect_rollover   基於 CSI 振幅 PC1 的翻身/大動作偵測程式 (相容 MATLAB R2015b)
%
% 算法原理：
%   結合「滑動標準差」與「一階差分能量」組成複合衝擊特徵，
%   並搭配「MAD 自適應動態閾值」，精準捕捉翻身時造成的階躍與衝擊，
%   同時避免因全域背景呼吸/噪音過大造成的誤判。
%
% 輸入:
%   amp_signal - 預處理後的振幅訊號 (如 process_csi_signal 輸出的 amp_pc1_norm) [N x 1]
%   Fs         - 採樣頻率 (Hz)
%   varargin   - 可選參數 (以鍵值對輸入，例如 'WinSec', 3)
%
% 輸出:
%   events     - 偵測到的翻身事件結構體陣列 (包含開始時間、結束時間、持續時間、強度)
%   var_signal - 計算出的動態複合特徵曲線 [N x 1]（維持格式不變，供觀察與調參）

%% 1. 參數解析與預設值 (R2015b 相容)
p = inputParser;
addRequired(p, 'amp_signal');
addRequired(p, 'Fs');
% 滑動窗口長度 (秒)：翻身一般持續 1.5 ~ 5 秒，預設用 2.5 秒
addParameter(p, 'WinSec', 5); 
% 自適應閾值靈敏度倍數 K (預設 2.5，越高越嚴格，可防止高噪聲誤判)
addParameter(p, 'ThreshStd', 6); 
% 最小事件間隔 (秒)：小於此間隔的兩個動作會被合併為同一次翻身
addParameter(p, 'MinMergeSec', 3);
% 最小事件持續時間 (秒)：過短的突波（如 < 0.8 秒）視為雜訊剔除
addParameter(p, 'MinDurationSec', 0.8);
% 最大事件持續時間 (秒)：合併後若超過此長度則視為長時間干擾，直接刪除
addParameter(p, 'MaxDurationSec', 20);

parse(p, amp_signal, Fs, varargin{:});
win_sec = p.Results.WinSec;
k_factor = p.Results.ThreshStd; % 將 ThreshStd 當做自適應倍數 K 使用
min_merge_sec = p.Results.MinMergeSec;
min_dur_sec = p.Results.MinDurationSec;
max_dur_sec = p.Results.MaxDurationSec;

N = length(amp_signal);
time_axis = (0:N-1)' / Fs;

%% 2. 複合特徵提取：滑動標準差 (STD) + 一階差分衝擊能量 (Diff)
win_len = round(win_sec * Fs);
if mod(win_len, 2) == 0
    win_len = win_len + 1;
end
half_w = floor(win_len / 2);

% 計算一階差分絕對值 (捕捉瞬間突變邊緣斜率)
diff_sig = [0; abs(diff(amp_signal))];

std_feat = zeros(N, 1);
diff_feat = zeros(N, 1);

for i = 1:N
    idx_start = max(1, i - half_w);
    idx_end = min(N, i + half_w);
    % 特徵 1: 滑動標準差
    std_feat(i) = std(amp_signal(idx_start:idx_end));
    % 特徵 2: 滑動平均一階差分能量
    diff_feat(i) = mean(diff_sig(idx_start:idx_end));
end

% 輸出變數維持 var_signal 格式 [N x 1]，但升級為兩者相乘的複合特徵能量
var_signal = std_feat .* diff_feat;

%% 3. MAD 動態自適應閾值計算 (Robust Adaptive Thresholding)
% 使用中位數與中位絕對偏差 (MAD) 估算該筆數據的環境背景噪音基底
bg_median = median(var_signal);
bg_mad = median(abs(var_signal - bg_median));

% 動態自適應閾值公式：背景基底 + K * 正規化 MAD
adaptive_thresh = bg_median + k_factor * (1.4826 * bg_mad + eps);

%% 4. 閾值判決與二值化
binary_detection = var_signal > adaptive_thresh;

%% 5. 偵測事件後處理 (邊緣搜尋、合併相鄰事件、長度過濾)
diff_bin = diff([0; binary_detection; 0]);
starts = find(diff_bin == 1);
ends = find(diff_bin == -1) - 1;

if isempty(starts)
    num_events = 0;
    events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {});
    fprintf('=== [偵測結束] 背景噪音基底: %.3f | 自適應閾值: %.3f | 未偵測到翻身 ===\n', bg_median, adaptive_thresh);
    plot_results(time_axis, amp_signal, var_signal, adaptive_thresh, events);
    return;
end

% 步驟 A: 合併間隔過近的事件
merged_starts = starts(1);
merged_ends = [];
for i = 1:length(starts)-1
    gap = (starts(i+1) - ends(i)) / Fs;
    if gap < min_merge_sec
        % 間隔太近，合併
    else
        merged_ends = [merged_ends; ends(i)];
        merged_starts = [merged_starts; starts(i+1)];
    end
end
merged_ends = [merged_ends; ends(end)];

% 步驟 B: 過濾事件長度 (MinDurationSec <= duration <= MaxDurationSec)
final_starts = [];
final_ends = [];

for i = 1:length(merged_starts)

    % 若事件起始於 0 秒，直接刪除
    if merged_starts(i) == 1
        fprintf('刪除事件：起始時間為 0 秒\n');
        continue;
    end

    dur = (merged_ends(i) - merged_starts(i)) / Fs;

    if dur >= min_dur_sec && dur <= max_dur_sec
        final_starts = [final_starts; merged_starts(i)];
        final_ends = [final_ends; merged_ends(i)];

    elseif dur > max_dur_sec
        fprintf('刪除持續過長事件：%.1f 秒 (超過 %.1f 秒，視為環境干擾)\n', ...
                dur, max_dur_sec);
    end

end

%% 6. 整理輸出事件
num_events = length(final_starts);
events = struct('start_time', cell(num_events, 1), ...
                'end_time', cell(num_events, 1), ...
                'duration', cell(num_events, 1), ...
                'intensity', cell(num_events, 1));

fprintf('=== [自適應翻身偵測報告] 背景基底: %.3f | 計算動態閾值: %.3f ===\n', bg_median, adaptive_thresh);
fprintf('共偵測到 %d 個翻身/大動作事件：\n', num_events);
for i = 1:num_events
    events(i).start_time = time_axis(final_starts(i));
    events(i).end_time = time_axis(final_ends(i));
    events(i).duration = events(i).end_time - events(i).start_time;
    % 以區間內的最大複合特徵值作為動作強度指標
    events(i).intensity = max(var_signal(final_starts(i):final_ends(i)));
    
    fprintf('事件 #%d: [%.1f秒 -> %.1f秒] | 持續: %.1f 秒 | 複合強度: %.2f\n', ...
            i, events(i).start_time, events(i).end_time, events(i).duration, events(i).intensity);
end

%% 7. 繪製偵測結果圖
plot_results(time_axis, amp_signal, var_signal, adaptive_thresh, events);

end

function plot_results(time_axis, amp_signal, var_signal, adaptive_thresh, events)
% 輔助繪圖函數
figure('Name', 'CSI Adaptive Rollover Detection Results', 'NumberTitle', 'off', 'Position', [150, 150, 1000, 550]);

% Subplot 1: 原始預處理後的 PC1 訊號與標註區間
subplot(2, 1, 1);
plot(time_axis, amp_signal, 'LineWidth', 1.2, 'Color', [0.15 0.15 0.15]); hold on;
grid on;
title('CSI Amplitude PC1 & 偵測到的翻身區間');
xlabel('時間 (秒)'); ylabel('Z-score 振幅');
axis tight;

% Subplot 2: 動態複合特徵與自適應閾值
subplot(2, 1, 2);
plot(time_axis, var_signal, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]); hold on;
plot([time_axis(1), time_axis(end)], [adaptive_thresh, adaptive_thresh], 'r--', 'LineWidth', 1.5);
grid on;
title('動態特徵：STD \times 一階差分衝擊 (Composite Feature)');
xlabel('時間 (秒)'); ylabel('特徵能量');
legend('複合特徵 (var\_signal)', '自適應動態閾值', 'Location', 'best');
axis tight;

% 在兩個子圖中用淡紅色區塊標記出偵測到的事件
for i = 1:length(events)
    t_start = events(i).start_time;
    t_end = events(i).end_time;
    
    % 子圖 1 標註
    subplot(2, 1, 1);
    ylim_curr = ylim;
    fill([t_start t_end t_end t_start], [ylim_curr(1) ylim_curr(1) ylim_curr(2) ylim_curr(2)], ...
         [1 0.7 0.7], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
     
    % 子圖 2 標註
    subplot(2, 1, 2);
    ylim_curr = ylim;
    fill([t_start t_end t_end t_start], [0 0 ylim_curr(2) ylim_curr(2)], ...
         [1 0.7 0.7], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
end

% 為了避免 fill 覆蓋掉原本的曲線，重新將線條與文字畫在最上層
subplot(2, 1, 1);
plot(time_axis, amp_signal, 'LineWidth', 1.2, 'Color', [0.15 0.15 0.15]);
for i = 1:length(events)
    text(events(i).start_time, max(amp_signal)*0.8, sprintf('Event %d', i), ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.6 0 0]);
end

end