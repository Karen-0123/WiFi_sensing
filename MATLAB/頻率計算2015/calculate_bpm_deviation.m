function [deviation_timeline, baseline_bpm] = calculate_bpm_deviation(bpm_timeline)
    valid_bpm = bpm_timeline(~isnan(bpm_timeline));
    
    if isempty(valid_bpm)
        baseline_bpm = 12; 
        deviation_timeline = zeros(size(bpm_timeline)); 
        return; 
    end
    
    % 使用直方圖尋找最常出現的呼吸率作為 Baseline
    edges = 8:0.5:18; 
    [counts, ~] = histcounts(valid_bpm, edges);
    [~, max_idx] = max(counts);
    baseline_bpm = edges(max_idx) + 0.25; 
    
    deviation_timeline = abs(bpm_timeline - baseline_bpm);
end