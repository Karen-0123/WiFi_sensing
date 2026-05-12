function [true_peak_idx, true_peak_vals] = detect_respiration_peaks(signal)
    % 呼吸訊號的精細峰值偵測
    % 輸入: signal - 1D 時域訊號 (PCA 後的最佳串流)
    
    fs = 200;
    N = length(signal);
    if N == 0, true_peak_idx = []; true_peak_vals = []; return; end

    % 1. 適應性閾值設定
    % 設定最小峰值高度為訊號標準差的 0.5 倍，有效過濾低位雜訊
    adaptive_threshold = 0.5 * std(signal); 

    % 2. 初步偵測 (使用動態閾值)
    % MinPeakHeight 確保只抓取具有一定強度的波動
    [candidate_pks, candidate_locs] = findpeaks(signal, 'MinPeakHeight', adaptive_threshold);

    %=============================================================================
    % 3. 虛假峰值剔除 (1.5 秒驗證窗口)
    % 設定 1.5 秒的驗證窗口 (包含當前點，前後共涵蓋 1.5 秒) 1.5*200=300個採樣點
    window_samples = round(1.5 * fs); 
    half_window = floor(window_samples / 2);
    %=============================================================================

    true_peak_idx = [];
    true_peak_vals = [];

    for i = 1:length(candidate_locs)
        curr_idx = candidate_locs(i);
        curr_val = candidate_pks(i);

        % 定義窗口範圍
        start_idx = max(1, curr_idx - half_window);
        end_idx = min(N, curr_idx + half_window);

        % 邏輯：窗口內是否有比目前更強的峰值？
        if curr_val >= max(signal(start_idx:end_idx))
            true_peak_idx(end+1) = curr_idx;
            true_peak_vals(end+1) = curr_val;
        end
    end

    % 在終端機印出處理結果
    fprintf('峰值偵測完成！初步找到 %d 個候選峰值，經 1.5 秒窗口剔除後，保留 %d 個有效呼吸頂點。\n', ...
            length(candidate_locs), length(true_peak_idx));
end