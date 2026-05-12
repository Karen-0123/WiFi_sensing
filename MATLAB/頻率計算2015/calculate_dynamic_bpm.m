function [bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, total_samples)
    % 透過滑動窗口與 P2P 間隔計算動態呼吸頻率 (BPM)
    % 輸入:
    %   true_peak_idx - 有效呼吸頂點的索引值陣列
    %   total_samples - 原始訊號的總採樣點數
    % 輸出:
    %   bpm_timeline  - 隨時間變化的 BPM 陣列
    %   time_axis_bpm - 對應的時間軸 (以窗口中心點為準)

    fs = 200;
    peak_times = true_peak_idx / fs;
    total_time = total_samples / fs;

    % 1. 20秒滑動窗口，1秒步長
    window_size = 20; 
    step_size = 1;    
    t_starts = 0:step_size:(total_time - window_size);
    
    bpm_timeline = NaN(1, length(t_starts)); % 預設為 NaN
    time_axis_bpm = t_starts + (window_size / 2);

    % 2. 遍歷窗口計算
    for i = 1:length(t_starts)
        t_s = t_starts(i);
        t_e = t_s + window_size;
        
        % 提取窗口內的峰值
        p_in_w = peak_times(peak_times >= t_s & peak_times <= t_e);
        
        % 3. 異常處理：至少需要 2 個峰值才能計算間隔
        if length(p_in_w) >= 2
            p2p_intervals = diff(p_in_w);
            Tp2p = mean(p2p_intervals);
            bpm_timeline(i) = 60 / Tp2p;
        end
    end
    
    fprintf('BPM 計算完成！成功產出隨時間變化的呼吸率曲線。\n');
end