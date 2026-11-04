function [csi_matrix, timestamp_sec, rssi] = read_intel5300_dat(filename)
    % read_intel5300_dat  基於官方 read_bf_file 的 Intel 5300 CSI 資料提取函式
    %
    % 輸入:
    %   filename      - Intel 5300 CSI .dat 檔案路徑
    %
    % 輸出:
    %   csi_matrix    - 複數 CSI 矩陣 [N, 30, 2, 3] (封包數 x 30子載波 x 2Tx x 3Rx)
    %   timestamp_sec - 展開溢位後的相對時間向量 (秒) [N, 1]
    %   rssi          - 三根接收天線的 RSSI 訊號強度 [N, 3]

    fprintf('正在使用 read_bf_file 讀取底層資料: %s ...\n', filename);
    csi_trace = read_bf_file(filename);

    total_pkts = length(csi_trace);
    csi_matrix = zeros(total_pkts, 30, 2, 3);
    timestamp  = zeros(total_pkts, 1);
    rssi       = zeros(total_pkts, 3);
    valid_idx  = 0;

    for i = 1:total_pkts
        entry = csi_trace{i};
        if isempty(entry) || ~isfield(entry, 'csi')
            continue;
        end
        [Ntx, Nrx, Nsc] = size(entry.csi);
        if Ntx == 2 && Nrx == 3 && Nsc == 30
            valid_idx = valid_idx + 1;
            csi_matrix(valid_idx, :, :, :) = permute(entry.csi, [3, 1, 2]);
            timestamp(valid_idx) = entry.timestamp_low;
            rssi(valid_idx, :)   = [entry.rssi_a, entry.rssi_b, entry.rssi_c];
        elseif Ntx == 1 && Nrx == 3 && Nsc == 30
            % 相容 1Tx x 3Rx 設定 (複製為 2Tx 維度以維持下游維度一致)
            valid_idx = valid_idx + 1;
            csi_1tx = permute(entry.csi, [3, 1, 2]); % [30, 1, 3]
            csi_matrix(valid_idx, :, 1, :) = csi_1tx;
            csi_matrix(valid_idx, :, 2, :) = csi_1tx;
            timestamp(valid_idx) = entry.timestamp_low;
            rssi(valid_idx, :)   = [entry.rssi_a, entry.rssi_b, entry.rssi_c];
        end
    end

    csi_matrix = csi_matrix(1:valid_idx, :, :, :);
    timestamp  = timestamp(1:valid_idx);
    rssi       = rssi(1:valid_idx, :);

    % 轉換 32-bit 微秒時間戳記為相對秒數 (處理 2^32 溢位)
    if ~isempty(timestamp)
        timestamp        = double(timestamp);
        wrap_value       = 2^32;
        timestamp_unwrap = timestamp;
        offset           = 0;

        for i = 2:length(timestamp)
            if timestamp(i) < timestamp(i-1)
                offset = offset + wrap_value;
            end
            timestamp_unwrap(i) = timestamp(i) + offset;
        end
        timestamp_sec = (timestamp_unwrap - timestamp_unwrap(1)) / 1e6;

        fprintf('duration = %.3f sec\n', timestamp_sec(end));
        if timestamp_sec(end) > 0
            fprintf('Fs estimated = %.3f Hz\n', (length(timestamp_sec)-1) / timestamp_sec(end));
        end
    else
        timestamp_sec = [];
        warning('未提取到任何有效的 CSI 封包！');
    end

    fprintf('轉換完成！原始封包數: %d, 成功提取的有效封包數: %d\n', total_pkts, valid_idx);
end
