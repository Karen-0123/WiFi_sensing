%% 0. 讀取官方 .dat 檔與資料重塑
% 確保 read_bf_file.m 與 get_scaled_csi.m 在你的 MATLAB 路徑中
filename = 'C:\Users\fupei\Desktop\csi\data\breathe\breathe_200hz_003.dat';
csi_trace = read_bf_file(filename);

N_total = length(csi_trace);
Fs = 200; % 採樣頻率 200 Hz

% 預配置記憶體，形狀為 (N, 30, 2, 3)
csi_matrix_temp = zeros(N_total, 30, 2, 3);
valid_count = 0;

% 解析每一筆封包
for i = 1:N_total
    csi_entry = csi_trace{i};
    
    % 排除掉包或損毀的空資料
    if isempty(csi_entry)
        continue;
    end
    
    % 將原始資料轉為實際複數 CSI，輸出維度為 (Nrx, Ntx, 30) = (2, 3, 30)
    csi = get_scaled_csi(csi_entry);
    
    % 確保該封包確實包含 3x2 的天線對
    if size(csi, 1) == 2 && size(csi, 2) == 3 && size(csi, 3) == 30
        valid_count = valid_count + 1;
        % 使用 permute 將 (2, 3, 30) 轉置為 (30, 2, 3)
        % 並存入對應的時間索引 valid_count 中
        csi_matrix_temp(valid_count, :, :, :) = permute(csi, [3, 1, 2]);
    end
end

% 截去未使用的預配置空間，得到最終的 4D 矩陣 (N, 30, 2, 3)
csi_matrix = csi_matrix_temp(1:valid_count, :, :, :);
N = valid_count;

% 提取原始相位 (結果為 -pi 到 pi)
raw_phase_4d = angle(csi_matrix); 

disp(['成功讀取的有效封包數量 N = ', num2str(N)]);
% 找出第一個有效封包的真實形狀
for i = 1:N_total
    if ~isempty(csi_trace{i})
        csi_test = get_scaled_csi(csi_trace{i});
        disp('真實的天線維度 (Nrx, Ntx, Subcarriers):');
        disp(size(csi_test));
        break;
    end
end

%% 1. 相位校準 (Phase Sanitization)
sanitized_phase_4d = zeros(N, 30, 2, 3);

for rx = 1:2
    for tx = 1:3
        for i = 1:N
            % 取出單一時間點、單一天線對的 30 個子載波
            temp_phase = squeeze(raw_phase_4d(i, :, rx, tx)); 
            
            % 跨子載波解纏繞並移除硬體時鐘造成的線性偏移
            sanitized_phase_4d(i, :, rx, tx) = detrend(unwrap(temp_phase));
        end
    end
end

%% 1.5 資料壓平 (Flattening)
% 將 4D 轉為 2D 時間序列：(N, 180)
sanitized_phase_2d = reshape(sanitized_phase_4d, N, 180);

%% 2. 相位解纏繞 (時間維度)
% 還原隨時間變化的物理位移
unwrapped_phase = unwrap(sanitized_phase_2d, [], 1);

%% 3. 帶通濾波 (Bandpass Filtering)
low_cutoff = 0.1;
high_cutoff = 0.5;
[b, a] = butter(3, [low_cutoff high_cutoff] / (Fs / 2), 'bandpass');

% 零相位濾波 180 個通道
filtered_phase = filtfilt(b, a, unwrapped_phase);

%% 4. 子載波融合 (PCA)
% 標準化並提取第一主成分
normalized_phase = zscore(filtered_phase);
[coeff, score, latent] = pca(normalized_phase);
breathing_signal = score(:, 1);

%% 繪圖檢視
figure;
t = (0:N-1) / Fs; % 建立時間軸 (秒)

subplot(2,1,1);
plot(t, filtered_phase(:, 1:5)); 
title('帶通濾波後的部分通道');
xlabel('時間 (秒)'); ylabel('相位');

subplot(2,1,2);
plot(t, breathing_signal, 'LineWidth', 1.5, 'Color', 'r');
title('PCA 提取出之最終呼吸訊號');
xlabel('時間 (秒)'); ylabel('PCA 振幅');