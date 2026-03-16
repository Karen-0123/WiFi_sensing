%% 0. 讀取官方 .dat 檔與資料重塑
% 確保 read_bf_file.m 與 get_scaled_csi.m 在你的 MATLAB 路徑中
filename = 'D:\大學資料\WiFi_sensing\data\breathe\breathe_200hz_015.dat';
csi_trace = read_bf_file(filename);

N_total = length(csi_trace);
Fs = 200; % 採樣頻率 200 Hz

disp(['總封包數量 N = ', num2str(N_total)]);

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

%% 0. 共軛乘法 (Conjugate Multiplication)

% 提取兩根接收天線的資料
% H1 尺寸: (N, 30, 1, 3) -> 第一根接收天線的所有資料
H1 = csi_matrix(:, :, 1, :); 
% H2 尺寸: (N, 30, 1, 3) -> 第二根接收天線的所有資料
H2 = csi_matrix(:, :, 2, :); 

% 執行共軛乘法：Hcm = H1 .* conj(H2)
% 注意：這裡使用點乘 (.*) 確保對應的子載波與發射天線進行運算
csi_cm = H1 .* conj(H2);

% 根據要求，輸出需與 input 有一樣的 shape (N, 30, 2, 3)
% 這裡我們將 CM 的結果填入新矩陣的第一個 RX 通道，或依需求決定放置位置
% 通常 CM 後空間維度會減半（因為兩根變一根），若要維持 4D，我們可以做如下處理：
csi_matrix_cm = zeros(size(csi_matrix));
csi_matrix_cm(:, :, 1, :) = csi_cm; 
% 註：csi_matrix_cm(:, :, 2, :) 預設為 0，代表 RX2 已被融合進 RX1

disp('共軛乘法運算完成。');

%% Savitzky-Golay 濾波器
% S-G 濾波器參數設定
% order: 多項式擬合階數 (通常 2-4)
% framelen: 視窗長度 (必須為奇數，長度愈大愈平滑，但太長會導致訊號失真)
sg_order = 3;
sg_framelen = 11; % 根據 200Hz 採樣率，11 點約為 0.05 秒，適合濾除微小抖動

% 對 csi_matrix_cm 的實部與虛部分別進行平滑 (避免直接對相位平滑造成的跳變問題)
% 這樣可以確保複數平面的軌跡更穩定
csi_matrix_cm_smoothed = csi_matrix_cm;

for tx = 1:3
    % 提取 RX1 的複數資料 (N, 30)
    temp_data = squeeze(csi_matrix_cm(:, :, 1, tx));
    
    % 對每一個子載波進行 S-G 濾波
    % sgolayfilt 作用於維度 1 (時間維度)
    temp_smoothed = sgolayfilt(temp_data, sg_order, sg_framelen, [], 1);
    
    % 放回平滑後的矩陣
    csi_matrix_cm_smoothed(:, :, 1, tx) = temp_smoothed;
end

disp('Savitzky-Golay 濾波處理完成。');

%% 提取原始相位 (結果為 -pi 到 pi)
cm_phase_3d = angle(csi_matrix_cm_smoothed(:, :, 1, :));
N_cm_total = size(cm_phase_3d, 1);

disp(['成功處理 CM 訊號，有效封包數量 N = ', num2str(N)]);

%% 1. 相位校準 (Phase Sanitization)
sanitized_phase_3d = zeros(N, 30, 3);

for tx = 1:3
    for i = 1:N
        % 取出單一時間點、單一 TX 的 30 個子載波
        temp_phase = squeeze(cm_phase_3d(i, :, tx)); 
        
        % 跨子載波解纏繞 (Unwrap across subcarriers)
        % 建議：如果 CM 效果好，這裡只需 unwrap，不一定要 detrend
        sanitized_phase_3d(i, :, tx) = unwrap(temp_phase);
    end
end

%% 1.5 資料壓平 (Flattening)
% 將 3D (N, 30, 3) 轉為 2D (N, 90)
% 因為現在只有一組有效 RX，所以總通道數從 180 變為 90
sanitized_phase_2d = reshape(sanitized_phase_3d, N, 90);

%% 2. 相位解纏繞 (時間維度)
% 還原隨時間變化的物理位移
unwrapped_phase = unwrap(sanitized_phase_2d, [], 1);

%% 3. 帶通濾波 (Bandpass Filtering)
low_cutoff = 0.1;
high_cutoff = 0.5;
[b, a] = butter(3, [low_cutoff high_cutoff] / (Fs / 2), 'bandpass');

% 零相位濾波
filtered_phase = filtfilt(b, a, unwrapped_phase);

%% 4. 子載波融合 (PCA)
% 排除掉可能的全零通道或異常值
valid_channels = ~all(filtered_phase == 0);
processed_data = filtered_phase(:, valid_channels);

% 標準化並提取第一主成分
normalized_phase = zscore(processed_data);
[coeff, score, latent] = pca(normalized_phase);
breathing_signal = score(:, 1);

%% 繪圖檢視
figure;
t = (0:N-1) / Fs; % 建立時間軸 (秒)

subplot(2,1,1);
plot(t, filtered_phase(:, 1:min(10, size(filtered_phase,2)))); 
title('CM 處理 + 帶通濾波後的差分相位通道');
xlabel('時間 (秒)'); ylabel('相位 (rad)');

subplot(2,1,2);
plot(t, breathing_signal, 'LineWidth', 1.5, 'Color', [0 0.5 0]);
title('從 CM 訊號中經 PCA 提取之呼吸波形');
xlabel('時間 (秒)'); ylabel('PCA 第一主成分');