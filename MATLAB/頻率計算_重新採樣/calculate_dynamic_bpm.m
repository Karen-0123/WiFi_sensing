function [bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, total_samples, gap_mask)
    % 透過 20 秒滑動窗口與 P2P 間隔計算動態呼吸頻率 (BPM)

    fs = 20; 
    peak_times = true_peak_idx / fs;
    total_time = total_samples / fs;

    % 20秒滑動窗口，1秒步長
    window_size = 20; 
    step_size = 1;    
    t_starts = 0:step_size:(total_time - window_size);
    
    bpm_timeline = NaN(1, length(t_starts)); 
    time_axis_bpm = t_starts + (window_size / 2);

    for i = 1:length(t_starts)
        t_s = t_starts(i);
        t_e = t_s + window_size;
        
        % 換算當前 20 秒窗口對應的重採樣點索引
        idx_start = max(1, round(t_s * fs) + 1);
        idx_end = min(total_samples, round(t_e * fs));
        
        % 熔斷機制：若此窗口內超過 25% 的時間屬於大面積丟包，該時段放棄計算 (保持 NaN)
        if mean(gap_mask(idx_start:idx_end)) > 0.25
            continue; 
        end
        
        % 提取落在此時間窗口內的頂點時間
        p_in_w = peak_times(peak_times >= t_s & peak_times <= t_e);
        
        % 必須具備至少 2 個波峰才能計算 P2P 間隔
        if length(p_in_w) >= 2
            p2p_intervals = diff(p_in_w);
            Tp2p = mean(p2p_intervals);
            bpm_timeline(i) = 60 / Tp2p;
        end
    end
    fprintf('動態 BPM 計算完畢。\n');
end