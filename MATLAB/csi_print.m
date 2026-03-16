%% 1. 讀取數據與參數設定
% 確保 read_bf_file.m 與 get_scaled_csi.m 在目前資料夾中
filename = 'C:\Users\Admin\OneDrive\Documents\MATLAB\WiFi_sensing\data\breathe\breathe_200hz_015.dat';
csi_trace = read_bf_file(filename);

N_total = length(csi_trace);
Fs = 200; % 對應 Python 中的 fs=200

fprintf('總封包數量: %d\n', N_total);

% 提取振幅 (Amplitude)
% 對應 Python: amp = np.abs(csi[:, :, 0, 0])
% 我們預先分配一個 (N x 30) 的矩陣
amp = zeros(N_total, 30);

for i = 1:N_total
    if ~isempty(csi_trace{i})
        csi_entry = csi_trace{i};
        csi = get_scaled_csi(csi_entry); % 獲取 [Rx, Tx, Subcarrier] 的複數矩陣
        % 提取第 1 根 Rx, 第 1 根 Tx 的 30 個子載波
        % squeeze 會將 1x1x30 變成 30x1
        amp(i, :) = abs(squeeze(csi(1, 1, :))); 
    end
end

%% 2. 低通濾波 (Lowpass Filter)
% 對應 Python: cutoff=0.1, fs=200, order=5
cutoff = 0.1;
order = 5;
Wn = cutoff / (Fs/2); % 正規化頻率
[b, a] = butter(order, Wn, 'low');

% 使用 filtfilt 進行零相位濾波，這與 Python 的 scipy.signal.filtfilt 效果完全一致
% 它能確保濾波後的波形不會產生時間位移（相位延遲）
amp_filtered = filtfilt(b, a, amp);

%% 3. PCA 降維 (Principal Component Analysis)
% 對應 Python: sklearn.decomposition.PCA(n_components=3)
% sklearn 的 PCA 預設會減去平均值（Center），但不做標準化（Scale）
% MATLAB 的 pca 函數預設行為與之一致
[coeff, score, latent] = pca(amp_filtered);

% 提取第一主成分
pc1 = score(:, 1);

%% 4. 繪圖視覺化 
% 圖表一：第一主成分 (Motion Feature)
figure('Name', 'First Principal Component', 'Color', 'w');
plot(pc1, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
title('First Principal Component (Motion Feature)');
xlabel('Packets');
ylabel('Score');
grid on;

% 圖表二：所有子載波濾波後的結果
figure('Name', 'All Filtered Subcarriers', 'Color', 'w', 'Position', [100 100 1000 600]);
plot(amp_filtered);
title('All Filtered Subcarriers');
xlabel('Packets');
ylabel('Amplitude');
grid on;

