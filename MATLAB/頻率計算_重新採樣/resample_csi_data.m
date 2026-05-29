function [csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig)
    % 輸出:
    %   csi_resampled - 重採樣後的 4D CSI 矩陣 [N_uniform, 30, 2, 3]
    %   t_uniform     - 均勻時間向量 [N_uniform, 1]
    %   gap_mask      - 丟包標記遮罩 [N_uniform, 1] (true 代表該處有嚴重丟包)

    %==============================================================================
    % 1. 輸入採樣率參數
    if nargin < 3
        Fs_orig = 200;  % 原始頻率不傳入參數預設為 200 Hz
        Fs_target = 20; % 目標頻率不傳入參數預設為 20 Hz
    end
    %==============================================================================
    
    [N_orig, Nsc, Ntx, Nrx] = size(csi_matrix);
    
    fprintf('原始平均採樣率: %.2f Hz -> 目標重採樣率: %d Hz\n', Fs_orig, Fs_target);

    % 2. 矩陣降維 (4D -> 2D) 以利於批量訊號處理
    csi_2d = reshape(csi_matrix, N_orig, []);

    % 3. 反混疊低通濾波 (Anti-aliasing)
    % 截止頻率設定為目標奈奎斯特頻率 (Fs_target / 2 = 10 Hz)
    cutoff = Fs_target / 2; 
    [b, a] = butter(4, cutoff / (Fs_orig / 2), 'low');
    csi_lp = filtfilt(b, a, real(csi_2d)) + 1i * filtfilt(b, a, imag(csi_2d));  % 對實部和虛部低通濾波

    % 4. 產生均勻時間向量(重新採樣時間軸)
    t_uniform = (timestamp_sec(1) : 1/Fs_target : timestamp_sec(end))';
    N_uniform = length(t_uniform);

    % 5. 一維線性插值對齊
    % 使用 'linear' 能在遇到大面積丟包時保持穩定的線性過渡，避免 pchip 的邊緣超調
    csi_resampled_2d = interp1(timestamp_sec, csi_lp, t_uniform, 'linear'); % 填入重新採樣的時間點的數據

    % 6. 丟包補償與精確標記 (間隔大於 0.5 秒)
    gap_mask = false(N_uniform, 1); % 建立N_uniform x 1的0矩陣(丟包時間軸)
    gap_threshold = 0.5; % 秒
    gap_indices = find(diff(timestamp_sec) > gap_threshold);    % 找到丟包大於gap_threshold秒的時間

    for g = 1:length(gap_indices)
        t_start = timestamp_sec(gap_indices(g));    % 丟包起始時間
        t_end = timestamp_sec(gap_indices(g) + 1);    % 丟包結束時間
        % 將重採樣後落在該斷層區間內的所有採樣點標記為無效
        gap_mask(t_uniform >= t_start & t_uniform <= t_end) = true;
    end

    % 7. 還原為 4D 矩陣
    csi_resampled = reshape(csi_resampled_2d, N_uniform, Nsc, Ntx, Nrx);    % 重新採樣後的數據
    fprintf('重採樣完成。均勻採樣點數: %d (偵測到 %d 處嚴重丟包斷層)\n', N_uniform, length(gap_indices));
end