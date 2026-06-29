function [true_peak_idx, true_peak_vals] = detect_respiration_peaks(signal, gap_mask, Fs_target)
    % 呼吸訊號的精細峰值偵測 (自適應相對突起度 + 丟包遮罩聯防)
    
    % 【BUG 修正】調整為正確的參數數量檢查
    if nargin < 3, Fs_target = 40; end  
    N = length(signal);
    if N == 0, true_peak_idx = []; true_peak_vals = []; return; end

    % 【策略優化】改用相對突起度 (Prominence)，只要高出局部低谷一定比例就納入候選
    % 這樣可以完美保留絕對幅值低、但波形明顯的綠點波峰
    adaptive_prominence = 0.2 * std(signal); 
    [candidate_pks, candidate_locs] = findpeaks(signal, 'MinPeakProminence', adaptive_prominence);

    % 2. 自適應驗證窗口 (縮短為 1 秒的半窗口，避免被相鄰大波峰的半山腰邊緣誤殺)
    window_samples = round(3 * Fs_target);
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

        % 鄰域最大值校驗
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