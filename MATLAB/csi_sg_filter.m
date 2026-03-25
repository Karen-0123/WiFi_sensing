clear; clc; close all;
feature('DefaultCharacterSet', 'UTF-8'); 

%% 1. 數據讀取與 4D 矩陣架構
filename = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\data\breathe\breathe_200hz_015.dat';
if ~exist(filename, 'file'), error('找不到檔案，請檢查路徑！'); end

csi_trace = read_bf_file(filename);
Fs = 200; 
N_total = length(csi_trace);
fprintf('1. 讀取封包總數: %d\n', N_total);

% 預配置 (N, 30, 2, 3)
csi_matrix_temp = zeros(N_total, 30, 2, 3);
valid_count = 0;
for i = 1:N_total
    if ~isempty(csi_trace{i})
        csi = get_scaled_csi(csi_trace{i});
        if size(csi, 1) == 2 && size(csi, 2) == 3 && size(csi, 3) == 30
            valid_count = valid_count + 1;
            csi_matrix_temp(valid_count, :, :, :) = permute(csi, [3, 1, 2]);
        end
    end
end
csi_matrix = csi_matrix_temp(1:valid_count, :, :, :);
N = valid_count;
t_full = (0:N-1) / Fs; 

%% 2. 共軛乘法 (CM) 與 複數 S-G 濾波
H1 = csi_matrix(:, :, 1, :); 
H2 = csi_matrix(:, :, 2, :); 
csi_cm = H1 .* conj(H2); 

sg_order = 3;
sg_framelen = 41; 
csi_cm_smoothed = zeros(size(csi_cm));
for tx = 1:3
    temp_data = squeeze(csi_cm(:, :, 1, tx));
    csi_cm_smoothed(:, :, 1, tx) = sgolayfilt(temp_data, sg_order, sg_framelen, [], 1);
end

%% 3. 提取振幅 (Amplitude) 與 帶通濾波 (核心修改：0.1~0.2Hz)
amp_data = abs(csi_cm_smoothed(:, :, 1, :));
amp_flatten = reshape(amp_data, N, 90);

% 修改濾波範圍為 0.1 ~ 0.2 Hz (對應 6 ~ 12 BPM)
f_low = 0.1; 
f_high = 0.2;
[b, a] = butter(3, [f_low f_high] / (Fs / 2), 'bandpass');
amp_filtered = filtfilt(b, a, amp_flatten);

% PCA 提取呼吸主成分
[~, score, ~] = pca(zscore(amp_filtered));
breathing_signal_raw = score(:, 1);

%% 4. 數值計算預處理 (裁切邊緣效應與 Detrend)
clip = 15 * Fs; 
t_clean = t_full(clip:end-clip);
sig_clean = detrend(breathing_signal_raw(clip:end-clip)); 
N_clean = length(sig_clean);

%% 5. 計算 BPM
% --- 方法一：FFT 頻譜分析 ---
L = N_clean;
Y = fft(sig_clean);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

% 同步修改搜尋索引範圍，確保與濾波器一致
res_idx = (f >= f_low & f <= f_high); 
[~, max_idx] = max(P1(res_idx));
peak_freq_range = f(res_idx);
BPM_fft = peak_freq_range(max_idx) * 60;

% --- 方法二：方波計數 ---
square_wave = sig_clean > 0;
raw_edges = find(diff(square_wave) > 0); 
% 最小間隔設為 2.5s (對應 0.2Hz 的半週期) 是合理的，能防止雜訊重複計數
min_dist = 2.5 * Fs; 
valid_edges = [];
if ~isempty(raw_edges)
    valid_edges = raw_edges(1);
    for k = 2:length(raw_edges)
        if (raw_edges(k) - valid_edges(end)) > min_dist
            valid_edges = [valid_edges; raw_edges(k)];
        end
    end
end
BPM_square = length(valid_edges) / ((t_clean(end)-t_clean(1))/60);

%% ---------------------------------------------------------
%% 6. 圖表1：訊號提取驗證
%% ---------------------------------------------------------
figure('Color', 'k', 'Name', '訊號提取過程驗證', 'Position', [100 100 900 600]);

subplot(2,1,1);
plot(t_full, amp_filtered, 'LineWidth', 0.5); 
title(['帶通濾波後的振幅通道 (', num2str(f_low), '-', num2str(f_high), ' Hz)'], 'Color', 'w');
xlabel('時間 (秒)', 'Color', 'w'); ylabel('幅度', 'Color', 'w');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'GridColor', 'w');
grid on; xlim([0 t_full(end)]);

subplot(2,1,2);
plot(t_full, breathing_signal_raw, 'Color', [0 0.8 0], 'LineWidth', 2);
title('PCA 提取出的呼吸波形 (第一主成分)', 'Color', 'w');
xlabel('時間 (秒)', 'Color', 'w'); ylabel('PCA Score', 'Color', 'w');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'GridColor', 'w');
grid on; xlim([0 t_full(end)]);

%% ---------------------------------------------------------
%% 7. 圖表2：BPM 報告
%% ---------------------------------------------------------
figure('Color', 'w', 'Name', 'BPM 數值分析報告', 'Position', [1000 100 800 900]);

subplot(3,1,1);
plot(t_clean, sig_clean, 'b', 'LineWidth', 1.5); hold on;
plot(t_clean(valid_edges), sig_clean(valid_edges), 'ro', 'MarkerFaceColor', 'r');
title(['呼吸波形校正 (BPM_Square: ', num2str(round(BPM_square,2)), ')']);
ylabel('幅度'); grid on;

subplot(3,1,2);
stairs(t_clean, square_wave, 'r', 'LineWidth', 1.2);
title('呼吸方波邏輯 (抗噪計數)');
ylim([-0.5 1.5]); grid on;

subplot(3,1,3);
plot(f, P1, 'k', 'LineWidth', 1.2); hold on;
plot(peak_freq_range(max_idx), P1(f==peak_freq_range(max_idx)), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlim([0 0.5]); % 縮小顯示範圍至 0.5Hz，看得更清楚
title(['FFT 頻譜分析 (BPM_FFT: ', num2str(round(BPM_fft,2)), ')']);
xlabel('頻率 (Hz)'); grid on;

fprintf('\n--- 最終驗證報告 ---\n');
fprintf('濾波器設定: %.2f - %.2f Hz\n', f_low, f_high);
fprintf('FFT 計算結果: %.2f BPM\n', BPM_fft);
fprintf('方波計數結果: %.2f BPM\n', BPM_square);
fprintf('絕對誤差: %.2f BPM\n', abs(BPM_fft - BPM_square));