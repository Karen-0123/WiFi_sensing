
clear; clc; close all;
feature('DefaultCharacterSet', 'UTF-8'); 

%% 1. 讀取與數據解析
filename = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\data\breathe\breathe_200hz_015.dat';

if ~exist(filename, 'file')
    error('找不到檔案，請確認路徑是否正確！');
end

csi_trace = read_bf_file(filename);
Fs = 200; 
N_total = length(csi_trace);

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

%% 2. 共軛乘法 (CM) 與 S-G 濾波
H1 = csi_matrix(:, :, 1, :); 
H2 = csi_matrix(:, :, 2, :); 
amp_cm = abs(H1 .* conj(H2)); 

sg_order = 3;
sg_framelen = 31; 
amp_smoothed = zeros(size(amp_cm));
for tx = 1:3
    temp = squeeze(amp_cm(:, :, 1, tx));
    amp_smoothed(:, :, 1, tx) = sgolayfilt(temp, sg_order, sg_framelen, [], 1);
end

%% 3. 帶通濾波與 PCA 提取
[b, a] = butter(3, [0.1 0.5] / (Fs / 2), 'bandpass');
amp_flatten = reshape(amp_smoothed, N, 90);
amp_filtered = filtfilt(b, a, amp_flatten);

[~, score, ~] = pca(zscore(amp_filtered));
breathing_signal = score(:, 1);

%% 4. 數據切邊 (解決邊緣突波問題)
clip = 5 * Fs; 
t = (0:N-1) / Fs;

t_clean = t(clip:end-clip);
sig_clean = breathing_signal(clip:end-clip);
N_clean = length(sig_clean);

%% 5. 計算 BPM
% FFT 法
L = N_clean;
Y = fft(sig_clean);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
res_idx = (f >= 0.1 & f <= 0.5);
[~, max_idx] = max(P1(res_idx));
f_res = f(res_idx);
peak_freq = f_res(max_idx);
BPM_fft = peak_freq * 60;

% 方波法
square_wave = sig_clean > 0;
rising_edges = find(diff(square_wave) > 0);
BPM_square = length(rising_edges) / ((t_clean(end)-t_clean(1))/60);

%% 6. 繪圖結果 (移除 yyaxis，改用三個 Subplot)
figure('Color', 'w', 'Name', 'CSI 呼吸監測終極分析', 'Position', [100 100 800 900]);

% 子圖 1: 原始波形
subplot(3,1,1);
plot(t_clean, sig_clean, 'b', 'LineWidth', 1);
hold on;
plot(t_clean(rising_edges), sig_clean(rising_edges), 'ro', 'MarkerSize', 4);
title(['呼吸 PCA 波形 (估計: ', num2str(round(BPM_square,1)), ' BPM)']);
ylabel('幅度'); grid on;

% 子圖 2: 方波 (單獨畫一欄)
subplot(3,1,2);
stairs(t_clean, square_wave, 'r', 'LineWidth', 1.5);
title('轉換後的呼吸方波 (0/1)');
ylabel('狀態'); ylim([-0.5 1.5]); grid on;

% 子圖 3: 頻譜分析
subplot(3,1,3);
plot(f, P1, 'k', 'LineWidth', 1.2);
hold on;
plot(peak_freq, P1(f==peak_freq), 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlim([0 1]); 
title(['FFT 頻譜分析 (最高峰: ', num2str(round(BPM_fft,1)), ' BPM)']);
xlabel('頻率 (Hz)'); ylabel('強度'); grid on;

fprintf('--- 呼吸分析報告 ---\n');
fprintf('FFT 頻譜計算法: %.2f BPM\n', BPM_fft);
fprintf('方波計算法: %.2f BPM\n', BPM_square);