function [csi_matrix, timestamp, rssi] = read_intel5300_dat(filename)
    % 基於官方 read_bf_file 的資料提取函式
    % 輸出:
    %   csi_matrix - 4D 矩陣: [有效封包數量, 30, 2 (Tx), 3 (Rx)]
    %   timestamp  - 1D 陣列: 時間戳記
    %   rssi       - 2D 矩陣: [有效封包數量, 3] 包含 RSSI_A, B, C

    % 1. 呼叫官方工具讀取所有封包
    try
        fprintf('正在使用 read_bf_file 讀取底層資料...\n');
        csi_trace = read_bf_file(filename);
    catch
        error('找不到 read_bf_file.m，請確認 Intel 5300 CSI Tool 是否已加入 MATLAB 路徑中！');
    end

    total_pkts = length(csi_trace);
    
    % 預先配置記憶體 (最大數量就是 total_pkts)
    csi_matrix = zeros(total_pkts, 30, 2, 3);
    timestamp = zeros(total_pkts, 1);
    rssi = zeros(total_pkts, 3);
    
    valid_idx = 0;

    % 2. 迭代處理並過濾空封包
    for i = 1:total_pkts
        entry = csi_trace{i};
        
        % 防呆機制：檢查封包是否損毀或為空
        if isempty(entry) || ~isfield(entry, 'csi') || isempty(entry.csi)
            continue;
        end
        
        % 取得此封包的天線數量
        [Ntx, Nrx, Nsc] = size(entry.csi);
        
        % 只提取符合 2 Tx, 3 Rx 且 30 個 Subcarriers 的封包
        if Ntx == 2 && Nrx == 3 && Nsc == 30
            valid_idx = valid_idx + 1;
            
            % 官方解出的 csi 維度是 [Tx, Rx, Subcarriers]
            % 使用 permute 將其轉置為你要求的 [Subcarriers, Tx, Rx]
            csi_matrix(valid_idx, :, :, :) = permute(entry.csi, [3, 1, 2]);
            
            % 提取時間戳與 RSSI
            timestamp(valid_idx) = entry.timestamp_low;
            rssi(valid_idx, :) = [entry.rssi_a, entry.rssi_b, entry.rssi_c];
        end
    end

    % 3. 裁減掉沒有用到的記憶體空間
    csi_matrix = csi_matrix(1:valid_idx, :, :, :);
    timestamp = timestamp(1:valid_idx);
    rssi = rssi(1:valid_idx, :);
    
    fprintf('轉換完成！原始封包數: %d, 成功提取的 2x3 封包數: %d\n', total_pkts, valid_idx);
end