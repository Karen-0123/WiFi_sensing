function [seg_90th, bpm_timeline, time_axis_bpm] = calculate_dynamic_bpm(true_peak_idx, total_samples, gap_mask, Fs_target)
    % 利用滑動視窗與 P2P 間隔計算動態呼吸率 (BPM)
    % 輸入:
    %   true_peak_idx - 檢測到的呼吸波峰在重採樣陣列中的索引 (點數)
    %   total_samples - 原始訊號的總採樣點數
    %   gap_mask      - 斷訊品質遮罩陣列 (1 表示斷訊/缺漏資料)
    %   Fs_target     - 訊號重採樣目標頻率 (Hz, 預設為 40 Hz)
    % 輸出:
    %   seg_90th      - 當前區段有效 BPM 的第 90 百分位數 (90th percentile)
    %   bpm_timeline  - 隨時間變化的動態 BPM 陣列
    %   time_axis_bpm - BPM 時間軸 (以滑動視窗中心點為基準)

    if nargin < 4, Fs_target = 40; end
    peak_times = true_peak_idx / Fs_target;
    total_time = total_samples / Fs_target;

    % 1. 20 秒滑動視窗，1 秒步長
    window_size = 20; 
    step_size = 1;    
    t_starts = 0:step_size:(total_time - window_size);
    
    bpm_timeline = NaN(1, length(t_starts)); % 預設填入 NaN
    time_axis_bpm = t_starts + (window_size / 2);

    % 2. 遍歷滑動視窗
    for i = 1:length(t_starts)
        t_s = t_starts(i);
        t_e = t_s + window_size;
        
        % 計算當前 20 秒視窗對應的採樣點索引
        idx_start = max(1, round(t_s * Fs_target) + 1);
        idx_end = min(total_samples, round(t_e * Fs_target));
        
        % 斷訊品質評估：若此視窗內斷訊比例超過 25%，則視為無效區段，不進行計算 (保持 NaN)
        if mean(gap_mask(idx_start:idx_end)) > 0.25
            continue; 
        end
        
        % 找出位於當前視窗內的波峰時間點
        p_in_w = peak_times(peak_times >= t_s & peak_times <= t_e);
        
        % 3. 正常處理：必須包含至少 2 個波峰才能計算 P2P 間隔
        if length(p_in_w) >= 2
            p2p_intervals = diff(p_in_w);
            Tp2p = mean(p2p_intervals);
            bpm_timeline(i) = 60 / Tp2p;
        end
    end
    
%   disp(bpm_timeline)
    
    % 計算當前區段的 90th 百分位數
    valid_bpm_seg = bpm_timeline(bpm_timeline >= 5 & bpm_timeline <= 40);
    if ~isempty(valid_bpm_seg)
        seg_90th = prctile(valid_bpm_seg, 90);
    else
        seg_90th = NaN; % 若區段無有效資料則返回 NaN
    end
    
    fprintf('動態 BPM 計算完成！\n');
end