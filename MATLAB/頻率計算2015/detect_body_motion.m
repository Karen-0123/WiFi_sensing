function [motion_stat, motion_flags, time_axis_motion] = detect_body_motion(amp_filtered, fs)
    % 功能: 計算 CSI 振幅的移動方差 (Moving Variance) 來偵測大體動
    % 輸入: 
    %   amp_filtered - 濾波後的 CSI 振幅矩陣 [N, 30]
    %   fs - 採樣頻率 (通常為 200)

    % 1. 降維：將 30 個子載波取平均，得到一維的總體趨勢
    amp_1d = mean(amp_filtered, 2);
    
    % 2. 設定視窗：動作偵測需要高靈敏度，所以視窗設很短 (例如 2 秒)
    window_sec = 2;
    win_samples = round(window_sec * fs);
    step_samples = round(1 * fs); % 每 1 秒滑動一次
    
    total_samples = length(amp_1d);
    num_steps = floor((total_samples - win_samples) / step_samples) + 1;
    
    if num_steps <= 0
        error('CSI 訊號太短，無法偵測動作。');
    end
    
    motion_stat = zeros(1, num_steps);
    time_axis_motion = zeros(1, num_steps);
    
    % 3. 計算每個短視窗內的振幅方差
    for i = 1:num_steps
        start_idx = (i-1) * step_samples + 1;
        end_idx = start_idx + win_samples - 1;
        
        segment = amp_1d(start_idx:end_idx);
        % 去趨勢後算方差，排除設備本身的漂移
        motion_stat(i) = var(detrend(segment)); 
        time_axis_motion(i) = (start_idx + win_samples/2) / fs;
    end
    
    % 4. 自適應門檻判定 (Adaptive Thresholding)
    % 取整段訊號的中位數作為「安靜時的基準雜訊」，乘上一個倍數作為大動作門檻
    % 論文中通常使用 MAD (中位數絕對偏差) 或經驗倍數，這裡使用 5 倍中位數
    baseline_noise = median(motion_stat);
    threshold = baseline_noise * 5; 
    
    % 產出 0 與 1 的二值化陣列 (1 代表有大動作)
    motion_flags = motion_stat > threshold;
end