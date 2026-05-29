function [csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename)
    % 基於官方 read_bf_file 的資料提取函式 (時間已優化為秒數)
    
    fprintf('正在使用 read_bf_file 讀取底層資料...\n');
    csi_trace = read_bf_file(filename);

    total_pkts = length(csi_trace);
    csi_matrix = zeros(total_pkts, 30, 2, 3);
    timestamp = zeros(total_pkts, 1);
    rssi = zeros(total_pkts, 3);     % 訊號強度
    valid_idx = 0;

   for i = 1:total_pkts
        entry = csi_trace{i};
        [Ntx, Nrx, Nsc] = size(entry.csi);
        if Ntx == 2 && Nrx == 3 && Nsc == 30
            valid_idx = valid_idx + 1;  % 記錄合法封包數
            csi_matrix(valid_idx, :, :, :) = permute(entry.csi, [3, 1, 2]); % 重新排列維度
            timestamp(valid_idx) = entry.timestamp_low; % 記錄時間
            rssi(valid_idx, :) = [entry.rssi_a, entry.rssi_b, entry.rssi_c];    % 記錄三根接收天線的訊號強度
        end
   end
   
    csi_matrix = csi_matrix(1:valid_idx, :, :, :);
    timestamp = timestamp(1:valid_idx);
    rssi = rssi(1:valid_idx, :);
    
    % === 新增：轉換時間戳記為秒數 (相對時間) ===
    if ~isempty(timestamp)
        % Intel 5300 內建時鐘通常為 10MHz 或 1MHz 核心，基礎單位為微秒
        timestamp_sec = (timestamp - timestamp(1)) / 1e6; 
    else
        timestamp_sec = [];
    end
    
    fprintf('轉換完成！原始封包數: %d, 成功提取的 2x3 封包數: %d\n', total_pkts, valid_idx);
end