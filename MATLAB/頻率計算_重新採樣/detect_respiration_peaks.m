function [true_peak_idx, true_peak_vals] = detect_respiration_peaks(signal, gap_mask)
    % 呼吸訊號的精細峰值偵測 (自適應閾值 + 丟包遮罩聯防)
    
    fs = 20; 
    N = length(signal);
    if N == 0, true_peak_idx = []; true_peak_vals = []; return; end

    % 1. 適應性閾值：動態跟隨 20Hz 訊號標準差
    adaptive_threshold = 0.5 * std(signal); 
    [candidate_pks, candidate_locs] = findpeaks(signal, 'MinPeakHeight', adaptive_threshold);

    % 2. 1.5 秒驗證窗口 (1.5 * 20 = 30 個採樣點)
    window_samples = round(1.5 * fs); 
    half_window = floor(window_samples / 2); 

    true_peak_idx = [];
    true_peak_vals = [];

    % 3. 虛假峰值與丟包點剔除
    for i = 1:length(candidate_locs)
        curr_idx = candidate_locs(i);
        curr_val = candidate_pks(i);

        % 強力攔截：若此點落在 Wi-Fi 嚴重丟包斷層區，直接判定為無效偽峰
        if gap_mask(curr_idx)
            continue; 
        end

        % 1.5 秒窗口內最大值鄰域校驗
        start_idx = max(1, curr_idx - half_window);
        end_idx = min(N, curr_idx + half_window);

        if curr_val >= max(signal(start_idx:end_idx))
            true_peak_idx(end+1) = curr_idx;
            true_peak_vals(end+1) = curr_val;
        end
    end
    
    fprintf('峰值偵測完成：候選峰值 %d 個 -> 校正後保留 %d 個有效呼吸頂點。\n', ...
            length(candidate_locs), length(true_peak_idx));
end