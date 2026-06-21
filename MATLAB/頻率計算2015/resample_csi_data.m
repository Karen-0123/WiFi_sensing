function [csi_resampled, t_uniform, gap_mask] = resample_csi_data(csi_matrix, timestamp_sec, Fs_target, Fs_orig)
    % resample_csi_data  抗混疊低通濾波與均勻重採樣模組 (全面偵測所有大小掉包)
    %
    % 輸入:
    %   csi_matrix    - 原始 4D CSI 矩陣 [N_orig, 30, 2, 3]
    %   timestamp_sec - 原始相對時間向量 (秒) [N_orig, 1]
    %   Fs_target     - 目標重採樣率 (預設 30 Hz)
    %   Fs_orig       - 原始採樣率 (預設 200 Hz)
    %
    % 輸出:
    %   csi_resampled - 重採樣與插值後的 4D CSI 矩陣 [N_uniform, 30, 2, 3]
    %   t_uniform      - 均勻的時間軸向量 [N_uniform, 1]
    %   gap_mask       - 丟包標記遮罩 [N_uniform, 1] (true 代表該處有任何微小或嚴重掉包)

    % 1. 輸入參數防呆與預設值設定
    if nargin < 4, Fs_orig = 200; end
    if nargin < 3, Fs_target = 40; end
    
    [N_orig, Nsc, Ntx, Nrx] = size(csi_matrix);
    if N_orig == 0, error('輸入的 CSI 矩陣為空！無法進行重採樣。'); end
    
    fprintf('====== [重採樣模組] 啟動 ======\n');
    fprintf('原始平均採樣率: %.2f Hz -> 目標均勻重採樣率: %d Hz\n', Fs_orig, Fs_target);

    % 2. 矩陣降維 (4D -> 2D) 以利於批量的插值與濾波處理
    csi_2d = reshape(csi_matrix, N_orig, []);

    % 3. 反混疊低通濾波 (Anti-aliasing)
    cutoff = Fs_target / 2; 
    [b, a] = butter(4, cutoff / (Fs_orig / 2), 'low');
    
    % 對實部和虛部獨立進行雙向低通濾波 (避免相位失真)
    csi_lp = filtfilt(b, a, real(csi_2d)) + 1i * filtfilt(b, a, imag(csi_2d));

    % 4. 產生均勻時間向量 (重新採樣的時間軸)
    t_uniform = (timestamp_sec(1) : 1/Fs_target : timestamp_sec(end))';
    N_uniform = length(t_uniform);

    % 5. 一維線性插值對齊
    csi_resampled_2d = interp1(timestamp_sec, csi_lp, t_uniform, 'linear');

    % =========================================================================
    % 地毯式偵測所有掉包：有掉的就都顯示出來s
    % =========================================================================
    % 初始化與均勻時間軸等長的布林遮罩 (預設 false 代表沒掉包)
    gap_mask = false(N_uniform, 1); 
    
    % 動態精確閾值：理論上每秒有 Fs_orig 個封包，前後間隔應為 1/Fs_orig 秒。
    % 只要原始兩個封包的時間差大於理論間隔的 1.5 倍，就判定為「硬體出現掉包」！
    expected_interval = 1 / Fs_orig;
    gap_threshold = expected_interval * 1.5; 
    
    % 找出原始時間軸中所有掉包的區間索引
    diff_t = diff(timestamp_sec);
    gap_indices = find(diff_t > gap_threshold);

    % 遍歷所有掉包斷層，將時間精準映射到重採樣後的均勻時間軸上
    for g = 1:length(gap_indices)
        t_start = timestamp_sec(gap_indices(g));     % 掉包起始時間 (秒)
        t_end = timestamp_sec(gap_indices(g) + 1);    % 掉包結束時間 (秒)
        
        % 只要 t_uniform 落在這個原始斷層區間內，全部強制標記為無效丟包點 (true)
        gap_mask(t_uniform >= t_start & t_uniform <= t_end) = true;
    end
    
    % 計算整段觀測時間中，丟包時間佔總時間的百分比
    lost_ratio = (sum(gap_mask) / N_uniform) * 100;
    % =========================================================================

    % 6. 還原為 4D 矩陣 [N_uniform, 30, 2, 3]
    csi_resampled = reshape(csi_resampled_2d, N_uniform, Nsc, Ntx, Nrx);
    
    fprintf('重採樣完成！均勻點數: %d | 共捕捉到 %d 處大小掉包斷層 | 總數據丟包率: %.2f%%\n', ...
            N_uniform, length(gap_indices), lost_ratio);
    fprintf('====== [重採樣模組] 處理完畢 ======\n\n');
end