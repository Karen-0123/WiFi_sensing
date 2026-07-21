function [seg_90th, bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, total_samples, gap_mask, Fs_target)
    % 透過滑動窗口與 P2P 間隔計算動態呼吸頻率 (BPM)
    % 輸入:
    %   true_peak_idx - 有效呼吸頂點的索引值陣列
    %   total_samples - 原始訊號的總採樣點數
    % 輸出:
    %   bpm_timeline  - 隨時間變化的 BPM 陣列
    %   time_axis_bpm - 對應的時間軸 (以窗口中心點為準)

    if nargin < 4, Fs_target = 40; end
    peak_times = true_peak_idx / Fs_target;
    total_time = total_samples / Fs_target;

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
        
        % 換算當前 20 秒窗口對應的重採樣點索引
        idx_start = max(1, round(t_s * Fs_target) + 1);
        idx_end = min(total_samples, round(t_e * Fs_target));
        
        % 熔斷機制：若此窗口內超過 25% 的時間屬於大面積丟包，該時段放棄計算 (保持 NaN)
        if mean(gap_mask(idx_start:idx_end)) > 0.25
            continue; 
        end
        
        % 提取落在此時間窗口內的頂點時間
        p_in_w = peak_times(peak_times >= t_s & peak_times <= t_e);
        
        % 3. 異常處理：至少需要 2 個峰值才能計算間隔
        if length(p_in_w) >= 2
            p2p_intervals = diff(p_in_w);
            Tp2p = mean(p2p_intervals);
            bpm_timeline(i) = 60 / Tp2p;
        end
    end
    
%     disp(bpm_timeline)
    
    % 計算當前區段的 90th 百分位
    valid_bpm_seg = bpm_timeline(bpm_timeline >= 5 & bpm_timeline <= 40);
    if ~isempty(valid_bpm_seg)
        seg_90th = prctile(valid_bpm_seg, 90);
    else
        seg_90th = NaN; % 若該區段無有效資料則給予 NaN
    end
    
    fprintf('動態 BPM 計算完畢。\n');
end