function baseline_bpm = calculate_nrem_baseline(all_bpm)
    % calculate_nrem_baseline  統計整晚有效 BPM (5-40 bpm) 之直方圖眾數作為 NREM 呼吸基線
    %
    % 輸入:
    %   all_bpm      - 整晚所有區段串接之動態呼吸率向量
    %
    % 輸出:
    %   baseline_bpm - 整晚呼吸基線 (BPM)

    valid_bpm = all_bpm(~isnan(all_bpm) & all_bpm >= 5 & all_bpm <= 40);

    if isempty(valid_bpm)
        baseline_bpm = 15; % 若無有效數據，給予預設值
        return;
    end

    % 使用 histcounts 直接計算 35 個 bins (涵蓋 5-40 bpm) 之眾數，無需開關隱形 Figure
    edges = linspace(5, 40, 36);
    counts = histcounts(valid_bpm, edges);
    [~, max_idx] = max(counts);
    bin_width = edges(2) - edges(1);
    baseline_bpm = edges(max_idx) + (bin_width / 2);

    fprintf('計算完成。整晚呼吸基線: %.2f bpm\n', baseline_bpm);
end
