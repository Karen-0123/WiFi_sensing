function [stages, epoch_time] = predict_sleep_stages(all_bpm, all_time, var_h, var_t, baseline_bpm, all_motion, all_m_time)
  
    % 參數調優區 (Parameter Tuning) 
    REM_VAR_LIMIT  = 0.7;   % 變異數高於此值 -> REM (做夢)
    DEEP_VAR_LIMIT = 0.65;  % 變異數低於此值 -> Deep (深睡) [原本 0.45]
    DEEP_DEV_LIMIT = 2.0;   % 偏差值低於此值 -> Deep (深睡) [原本 1.8]
    
    epoch_sec = 30;
    num_epochs = floor(max(all_time) / epoch_sec);
    stages = zeros(1, num_epochs);
    epoch_time = (1:num_epochs) * epoch_sec / 60;
    
    last_valid_stage = 1; % 預設為 Core
    
    for i = 1:num_epochs
        t_start = (i-1) * epoch_sec;
        t_end = i * epoch_sec;
        
        % 1. 檢查這 30 秒內有沒有大動作 (從 detect_body_motion 來的)
        m_idx = (all_m_time >= t_start) & (all_m_time < t_end);
        if any(all_motion(m_idx) == 1)
            stages(i) = 3; % Awake (大動作優先判斷)
            last_valid_stage = 1;
            continue;
        end
        
        % 2. 如果沒動作，再檢查呼吸數據
        idx = (all_time >= t_start) & (all_time < t_end);
        v_bpm = all_bpm(idx);
        v_bpm = v_bpm(~isnan(v_bpm));
        
        if isempty(v_bpm)
            
            stages(i) = 1; 
        else
            % 抓取對應時間點的變異數
            [~, v_idx] = min(abs(var_t - (t_start + 15)/60));
            curr_var = var_h(v_idx);
            epoch_dev = mean(abs(v_bpm - baseline_bpm));
            
            % --- 判定邏輯 (參考 SMARS 論文) ---
            if curr_var > REM_VAR_LIMIT || epoch_dev > 2.5
                stages(i) = 2; % REM
            elseif curr_var < DEEP_VAR_LIMIT && epoch_dev < DEEP_DEV_LIMIT
                stages(i) = 0; % Deep
            else
                stages(i) = 1; % Core
            end
            last_valid_stage = stages(i);
        end
    end
    

    stages = medfilt1(stages, 3);
end