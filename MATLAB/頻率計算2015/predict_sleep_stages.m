function [stages, epoch_time] = predict_sleep_stages(all_bpm, all_time, var_history, var_time, baseline_bpm)
    epoch_sec = 30; 
    total_duration = max(all_time);
    num_epochs = floor(total_duration / epoch_sec);
    
    stages = zeros(1, num_epochs); 
    epoch_time = (1:num_epochs) * epoch_sec / 60; 
    
    for i = 1:num_epochs
        t_start = (i-1) * epoch_sec;
        t_end = i * epoch_sec;
        
        idx = (all_time >= t_start) & (all_time < t_end);
        epoch_bpm = all_bpm(idx);
        valid_bpm = epoch_bpm(~isnan(epoch_bpm));
        
        [~, v_idx] = min(abs(var_time - (t_start + epoch_sec/2)/60));
        current_var = var_history(v_idx);
        
        % 醫學標準數值：Wake=3 (最高), REM=2, Light=1, Deep=0 (最沉)
        if length(valid_bpm) < 3 || isnan(current_var)
            stages(i) = 3; % Wake (清醒/大動作導致訊號遺失)
        else
            epoch_dev = mean(abs(valid_bpm - baseline_bpm));
            if current_var > 0.8 || epoch_dev > 2.5  
                stages(i) = 2; % REM (做夢)
            else
                % 將深睡的門檻放寬 (原本是 0.2，現在改成 0.35)
                if current_var < 0.35 && epoch_dev < 1.5 
                    stages(i) = 0; % Deep (深睡)
                else
                    stages(i) = 1; % Light (淺睡)
            end
        end
    end
end