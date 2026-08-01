function [var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, window_sec, step_sec, target_epochs)
    % 呼吸頻率變異性計算 (4階 Butter 濾波去趨勢 + 視窗歸一化)
    % 輸入:
    %    all_bpm       - 瞬時呼吸率數據 (BPM 序列)
    %    all_time      - 對應的時間軸 (秒)
    %    window_sec    - 視窗長度 (預設 180 秒)
    %    step_sec      - 滑動步長 (預設 180 秒)
    %    target_epochs - 目標 Epoch 數量 (可傳入數字如 102，或傳入檔案/特徵陣列)
    
    if nargin < 3 || isempty(window_sec), window_sec = 180; end
    if nargin < 4 || isempty(step_sec), step_sec = 180; end
    
    % 強制將輸入轉為列向量 (N x 1)
    all_bpm = all_bpm(:);
    all_time = all_time(:);
    
    % 1. 數據插值處理 (填補 NaN 以便進行 Butterworth 濾波)
    valid_idx = ~isnan(all_bpm) & all_bpm >= 5 & all_bpm <= 40;
    if sum(valid_idx) < 10
        var_history = [];
        var_time = [];
        return;
    end
    bpm_interp = interp1(all_time(valid_idx), all_bpm(valid_idx), all_time, 'pchip', 'extrap');
    bpm_interp = bpm_interp(:);
    
    % 2. 巴特沃斯低通濾波器去趨勢 (fc = 0.1 Hz)
    dt = mean(diff(all_time)); % 平均採樣間隔 (秒)
    if dt <= 0, dt = 1; end
    fs_bpm = 1 / dt; % 採樣率 (Hz)
    
    Wn = 0.1 / (fs_bpm / 2);
    if Wn >= 1, Wn = 0.99; end
    
    % 使用 padarray 做邊界反射擴充，防止邊界失真
    pad_len = min(100, length(bpm_interp)-1);
    bpm_padded = padarray(bpm_interp, pad_len, 'replicate', 'both');
    
    % 使用 4 階低通濾波提取趨勢
    [b, a] = butter(4, Wn, 'low');
    trend_padded = filtfilt(b, a, bpm_padded);
    
    % 裁切回原始長度並去除趨勢
    trend = trend_padded(pad_len+1 : end-pad_len);
    detrended_bpm = bpm_interp - trend;
    
    % 3. 判斷總 Epoch 數量 (支援傳入數字純量或陣列/Cell)
    if isscalar(target_epochs)
        num_epochs = target_epochs;
    else
        num_epochs = length(target_epochs);
    end
    
    var_history = zeros(num_epochs, 1);
    var_time = zeros(num_epochs, 1);
    t_start = all_time(1);
    
    % 4. 針對每個 Epoch 計算方差與歸一化
    for k = 1:num_epochs
        epoch_t_start = t_start + (k-1) * step_sec;
        epoch_t_end   = t_start + k * step_sec;
        
        % 記錄該 Epoch 的中心時間戳 (方便繪圖對齊)
        var_time(k) = (epoch_t_start + epoch_t_end) / 2;
        
        % 抓取該 180 秒 Epoch 區間內的數據點
        in_win = (all_time >= epoch_t_start) & (all_time < epoch_t_end);
        segment = detrended_bpm(in_win);
        
        if length(segment) >= 2
            raw_variance = var(segment);
            var_history(k) = raw_variance / window_sec; % 除以 180 歸一化
        else
            var_history(k) = 0;
        end
    end
end