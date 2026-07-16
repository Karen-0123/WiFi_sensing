function [events, var_signal] = detect_rollover(amp_signal, Fs, varargin)
% detect_rollover   基於 CSI 振幅 PC1 的翻身/大動作偵測程式 (相容 MATLAB R2015b)
%
% 算法原理：
%   結合滑動標準差 (Sliding Standard Deviation) 與一階差分能量，
%   捕捉翻身時造成的「基線突變」與「高頻劇烈干擾」。
%
% 輸入:
%   amp_signal - 預處理後的振幅訊號 (如 process_csi_signal 輸出的 amp_pc1_norm) [N x 1]
%   Fs         - 採樣頻率 (Hz)
%   varargin   - 可選參數 (以鍵值對輸入，例如 'WinSec', 3)
%
% 輸出:
%   events     - 偵測到的翻身事件結構體陣列 (包含開始時間、結束時間、持續時間、強度)
%   var_signal - 計算出的滑動標準差特徵曲線（便於觀察調參）

%% 1. 參數解析與預設值 (R2015b 相容)
p = inputParser;
addRequired(p, 'amp_signal');
addRequired(p, 'Fs');
% 滑動窗口長度 (秒)：翻身一般持續 1.5 ~ 5 秒，預設用 3 秒
addParameter(p, 'WinSec', 3); 
% 判定為翻身的標準差閾值 (因輸入已 z-score，預設大於 0.8 代表劇烈波動)
addParameter(p, 'ThreshStd', 0.8); 
% 最小事件間隔 (秒)：小於此間隔的兩個動作會被合併為同一次翻身
addParameter(p, 'MinMergeSec', 4);
% 最小事件持續時間 (秒)：過短的突波（如 < 0.8 秒）視為雜訊剔除
addParameter(p, 'MinDurationSec', 0.8);

parse(p, amp_signal, Fs, varargin{:});
win_sec = p.Results.WinSec;
thresh_std = p.Results.ThreshStd;
min_merge_sec = p.Results.MinMergeSec;
min_dur_sec = p.Results.MinDurationSec;

N = length(amp_signal);
time_axis = (0:N-1)' / Fs;

%% 2. 特徵提取：滑動標準差 (Sliding STD)
% R2015b 中不支援 movstd，我們使用實用的 filter 技巧或 R2015b 自帶的 sliding 實作
win_len = round(win_sec * Fs);
if mod(win_len, 2) == 0
    win_len = win_len + 1;
end

% 使用 standard sliding window 計算標準差 (R2015b 適用方法)
var_signal = zeros(N, 1);
half_w = floor(win_len / 2);
for i = 1:N
    idx_start = max(1, i - half_w);
    idx_end = min(N, i + half_w);
    var_signal(i) = std(amp_signal(idx_start:idx_end));
end

%% 3. 閾值判決與二值化
% 找出所有標準差超過閾值的點
binary_detection = var_signal > thresh_std;

%% 4. 偵測事件後處理 (合併相鄰事件、剔除過短雜訊)
% 尋找邊緣
diff_bin = diff([0; binary_detection; 0]);
starts = find(diff_bin == 1);
ends = find(diff_bin == -1) - 1;

if isempty(starts)
    events = struct('start_time', {}, 'end_time', {}, 'duration', {}, 'intensity', {});
    fprintf('=== [偵測結束] 未偵測到任何翻身事件 ===\n');
    plot_results(time_axis, amp_signal, var_signal, thresh_std, events);
    return;
end

% 步驟 A: 合併間隔過近的事件
merged_starts = starts(1);
merged_ends = [];
for i = 1:length(starts)-1
    gap = (starts(i+1) - ends(i)) / Fs;
    if gap < min_merge_sec
        % 間隔太近，合併：不記錄當前的 end，讓它繼續往後延伸
    else
        merged_ends = [merged_ends; ends(i)];
        merged_starts = [merged_starts; starts(i+1)];
    end
end
merged_ends = [merged_ends; ends(end)];

% 步驟 B: 過濾持續時間太短的事件
final_starts = [];
final_ends = [];
for i = 1:length(merged_starts)
    dur = (merged_ends(i) - merged_starts(i)) / Fs;
    if dur >= min_dur_sec
        final_starts = [final_starts; merged_starts(i)];
        final_ends = [final_ends; merged_ends(i)];
    end
end

%% 5. 整理輸出事件
num_events = length(final_starts);
events = struct('start_time', cell(num_events, 1), ...
                'end_time', cell(num_events, 1), ...
                'duration', cell(num_events, 1), ...
                'intensity', cell(num_events, 1));

fprintf('=== [翻身偵測報告] 共偵測到 %d 個翻身/大動作事件 ===\n', num_events);
for i = 1:num_events
    events(i).start_time = time_axis(final_starts(i));
    events(i).end_time = time_axis(final_ends(i));
    events(i).duration = events(i).end_time - events(i).start_time;
    % 以區間內的最大標準差作為動作強度指標
    events(i).intensity = max(var_signal(final_starts(i):final_ends(i)));
    
    fprintf('事件 #%d: [%.1f秒 -> %.1f秒] | 持續: %.1f 秒 | 動作強度: %.2f\n', ...
            i, events(i).start_time, events(i).end_time, events(i).duration, events(i).intensity);
end

%% 6. 繪製偵測結果圖
plot_results(time_axis, amp_signal, var_signal, thresh_std, events);

end

function plot_results(time_axis, amp_signal, var_signal, thresh_std, events)
% 輔助繪圖函數
figure('Name', 'CSI Rollover Detection Results', 'NumberTitle', 'off', 'Position', [150, 150, 1000, 550]);

% Subplot 1: 原始預處理後的 PC1 訊號與標註區間
subplot(2, 1, 1);
plot(time_axis, amp_signal, 'LineWidth', 1.2, 'Color', [0.15 0.15 0.15]); hold on;
grid on;
title('CSI Amplitude PC1 & 偵測到的翻身區間');
xlabel('時間 (秒)'); ylabel('Z-score 振幅');
axis tight;

% Subplot 2: 滑動標準差特徵與閾值
subplot(2, 1, 2);
plot(time_axis, var_signal, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]); hold on;
plot([time_axis(1), time_axis(end)], [thresh_std, thresh_std], 'r--', 'LineWidth', 1.5);
grid on;
title('動態特徵：滑動標準差 (Sliding STD)');
xlabel('時間 (秒)'); ylabel('標準差強度');
legend('Sliding STD', '偵測閾值 (Threshold)', 'Location', 'best');
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

% 為了避免 fill 覆蓋掉原本的曲線，重新強行將線條畫在最上層
subplot(2, 1, 1);
plot(time_axis, amp_signal, 'LineWidth', 1.2, 'Color', [0.15 0.15 0.15]);
for i = 1:length(events)
    text(events(i).start_time, max(amp_signal)*0.8, sprintf('Event %d', i), ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.6 0 0]);
end

end