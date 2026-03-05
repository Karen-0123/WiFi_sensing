% 1. 讀取 .dat 檔案 (請替換為你的檔名)
filename = 'C:\Users\fupei\Desktop\csi\data\flip50hzR_20s_001.dat';
fprintf('正在讀取檔案: %s\n', filename);
csi_trace = read_bf_file(filename);

% 檢查是否成功讀取
if isempty(csi_trace)
    error('未讀取到任何資料，請檢查檔案路徑或檔案是否損壞。');
end

num_packets = length(csi_trace);
fprintf('成功讀取 %d 個封包。\n', num_packets);

% 2. 選擇要觀察的封包 (這裡取第 1 個封包)
packet_idx = 1;
csi_entry = csi_trace{packet_idx};

% 3. 取得經過縮放的 CSI 矩陣
% 回傳的形狀通常是 [Nrx, Ntx, 30]
csi_matrix = get_scaled_csi(csi_entry);
[Nrx, Ntx, num_tones] = size(csi_matrix);

fprintf('天線配置: Nrx=%d, Ntx=%d\n', Nrx, Ntx);

% 4. 繪製相位圖
figure('Name', 'CSI Phase Analysis', 'Color', 'w');
hold on;

% 走訪所有的傳送與接收天線對
for tx = 1:Ntx
    for rx = 1:Nrx
        % 提取特定天線對的 30 個子載波 (使用 squeeze 轉為 1D 陣列)
        subcarrier_csi = squeeze(csi_matrix(rx, tx, :));
        
        % 計算原始相位 (弧度)
        raw_phase = angle(subcarrier_csi);
        
        % 相位展開 (Unwrap)，避免 -pi 到 pi 的跳變
        unwrapped_phase = unwrap(raw_phase);
        
        % 繪製曲線
        plot_label = sprintf('RX%d - TX%d', rx, tx);
        plot(1:num_tones, unwrapped_phase, '-o', 'LineWidth', 1.5, 'DisplayName', plot_label);
    end
end

% 設定圖表外觀
title(sprintf('CSI Unwrapped Phase (Packet %d)', packet_idx), 'FontSize', 14);
xlabel('Subcarrier Index (1-30)', 'FontSize', 12);
ylabel('Phase (Radians)', 'FontSize', 12);
grid on;
legend('Location', 'best');
hold off;