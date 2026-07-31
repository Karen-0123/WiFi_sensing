function [var_history, var_time] = calculate_breathing_variability(all_bpm, all_time, window_sec, step_sec)
    % 呼吸頻率變異性計算 (10階 Butter 濾波去趨勢 + 視窗歸一化)
    % 輸入:
    %   all_bpm     - 瞬時呼吸率數據 (BPM 序列)
    %   all_time    - 對應的時間軸 (秒)
    %   window_sec  - 視窗長度 (300 秒)
    %   step_sec    - 滑動步長 (30 秒)
    
    if nargin < 3, window_sec = 300; end
    if nargin < 4, step_sec = 30; end
    
    num_pts = length(all_bpm);
    var_history = zeros(size(all_bpm));
    var_time = all_time;
    
    % 1. 數據插值處理 (填補 NaN 以便進行 Butterworth 濾波)
    valid_idx = ~isnan(all_bpm) & all_bpm >= 5 & all_bpm <= 40;
    if sum(valid_idx) < 10
        return;
    end
    bpm_interp = interp1(all_time(valid_idx), all_bpm(valid_idx), all_time, 'pchip', 'extrap');
    
    % 2. 10 階巴特沃斯低通濾波器去趨勢 (fc = 0.1 Hz)】
    dt = mean(diff(all_time));
    if dt <= 0, dt = 1; end
    fs_bpm = 1 / dt; % bpm_timeline 的採樣率 (Hz)
    
    Wn = 0.1 / (fs_bpm / 2);
    if Wn >= 1, Wn = 0.99; end
    
    % 使用 padarray 做邊界反射擴充，防止 10 階濾波器在開頭產生邊界失真
    pad_len = min(100, length(bpm_interp)-1);
    bpm_padded = padarray(bpm_interp', pad_len, 'replicate', 'both')';
    
    [b, a] = butter(10, Wn, 'low');
    trend_padded = filtfilt(b, a, bpm_padded);
    
    % 裁切回原始長度
    trend = trend_padded(pad_len+1 : end-pad_len);
    
    % 原始值減去趨勢值 (去趨勢後的呼吸率波動)
    detrended_bpm = bpm_interp - trend;
    
    % 3.300 秒自適應視窗切分、方差計算與歸一化】
    half_win = window_sec / 2;
    for k = 1:num_pts
        t_curr = all_time(k);
        
        % 自適應邊界保護：即便在開頭 (如 t = 15s)，也抓取現有的所有點進行計算，絕不給 0
        in_win = (all_time >= max(all_time(1), t_curr - half_win)) & ...
                 (all_time <= min(all_time(end), t_curr + half_win));
             
        segment = detrended_bpm(in_win);
        
        if length(segment) >= 2
            % 計算方差並除以時段長度 (300) 進行歸一化
            raw_variance = var(segment);
            var_history(k) = raw_variance / window_sec;
        else
            var_history(k) = var(detrended_bpm) / window_sec;
        end
    end
end