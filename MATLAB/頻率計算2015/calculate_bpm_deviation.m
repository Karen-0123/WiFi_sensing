function [var_timeline, time_axis_var] = calculate_breathing_variability(bpm_timeline, time_axis_bpm, window_size_sec, step_size_sec)
    % 1. 計算 BPM 資料的採樣間隔
    dt = time_axis_bpm(2) - time_axis_bpm(1);
    fs_bpm = 1 / dt;
    
    % 2. 轉換視窗樣本數
    win_samples = round(window_size_sec * fs_bpm);
    step_samples = round(step_size_sec * fs_bpm);
    
    total_len = length(bpm_timeline);
    num_steps = floor((total_len - win_samples) / step_samples) + 1;

    var_timeline = zeros(1, num_steps);
    time_axis_var = zeros(1, num_steps);

    % 3. 滑動視窗計算
    for i = 1:num_steps
        start_idx = (i-1) * step_samples + 1;
        end_idx = start_idx + win_samples - 1;
        
        current_segment = bpm_timeline(start_idx:end_idx);
        valid_data = current_segment(~isnan(current_segment));
        
        if length(valid_data) < (win_samples * 0.6)
            var_timeline(i) = NaN;
        else
            % 去趨勢處理 (SMARS 論文核心)
            detrended_data = detrend(valid_data);
            var_timeline(i) = var(detrended_data);
        end
        time_axis_var(i) = time_axis_bpm(start_idx + floor(win_samples/2)) / 60;
    end
end