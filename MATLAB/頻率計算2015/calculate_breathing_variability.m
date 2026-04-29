function [var_timeline, time_axis_var] = calculate_breathing_variability(bpm_timeline, time_axis_bpm, window_size_sec, step_size_sec)
    % 檢查數據量
    if length(time_axis_bpm) < 2
        var_timeline = []; time_axis_var = []; return;
    end

    % 安全計算採樣間隔
    dt = mean(diff(time_axis_bpm));
    if dt <= 0, dt = 1; end % 防止無窮大
    fs_bpm = 1 / dt;
    
    win_samples = round(window_size_sec * fs_bpm);
    step_samples = round(step_size_sec * fs_bpm);
    
    total_samples = length(bpm_timeline);
    num_steps = floor((total_samples - win_samples) / step_samples) + 1;
    
    if num_steps <= 0, var_timeline = []; time_axis_var = []; return; end
    
    var_timeline = zeros(1, num_steps);
    time_axis_var = zeros(1, num_steps);
    
    for i = 1:num_steps
        start_idx = (i-1) * step_samples + 1;
        end_idx = start_idx + win_samples - 1;
        
        segment = bpm_timeline(start_idx:end_idx);
        valid_segment = segment(~isnan(segment));
        
        if length(valid_segment) > 5
            var_timeline(i) = var(detrend(valid_segment));
        else
            var_timeline(i) = NaN;
        end
        % 轉換為分鐘座標
        time_axis_var(i) = time_axis_bpm(round((start_idx + end_idx)/2)) / 60;
    end
end