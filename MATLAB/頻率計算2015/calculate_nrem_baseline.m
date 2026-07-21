function baseline_bpm = calculate_nrem_baseline(all_bpm)
    % 過濾範圍 5-40 bpm
    valid_bpm = all_bpm(all_bpm >= 5 & all_bpm <= 40);
    
    if isempty(valid_bpm)
        baseline_bpm = 15; % 若無有效數據，給予預設值
        return;
    end
    
    % 繪製直方圖 (在背景進行)
    figure('Visible', 'off');
    h = histogram(valid_bpm, 35); % 35 個 bins 涵蓋 5-40 bpm
    
    % 找出出現頻率最高的點 (Mode)
    [~, max_idx] = max(h.Values);
    baseline_bpm = h.BinEdges(max_idx) + (h.BinWidth / 2);
    
    fprintf('計算完成。整晚呼吸基線: %.2f bpm\n', baseline_bpm);
    
    close(gcf);
end